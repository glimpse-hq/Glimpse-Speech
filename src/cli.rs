use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::{Parser, Subcommand};
use directories::ProjectDirs;

use crate::{
    models::{InstallSpec, ModelEngine, ModelInstallManager, ModelStorage, RemoteFile},
    service::{AudioInput, SpeechService, TranscribeRequest},
    Transcription,
};

fn default_cache_dir() -> PathBuf {
    if let Ok(value) = std::env::var("GLIMPSE_SPEECH_CACHE_DIR") {
        return PathBuf::from(value);
    }

    #[cfg(target_os = "macos")]
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("com.glimpse.data")
            .join("models");
    }

    if let Some(project_dirs) = ProjectDirs::from("com", "Glimpse", "glimpse-speech") {
        return project_dirs.data_local_dir().join("models");
    }

    PathBuf::from("glimpse-speech").join("models")
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum CliEngine {
    Whisper,
    Parakeet,
    Nemotron,
}

impl From<CliEngine> for ModelEngine {
    fn from(engine: CliEngine) -> Self {
        match engine {
            CliEngine::Whisper => ModelEngine::Whisper,
            CliEngine::Parakeet => ModelEngine::Parakeet,
            CliEngine::Nemotron => ModelEngine::Nemotron,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "glimpse")]
#[command(about = "Local Glimpse transcription from the terminal")]
struct Cli {
    #[arg(long, global = true)]
    cache_dir: Option<PathBuf>,
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    Transcribe {
        audio: PathBuf,
        #[arg(long)]
        model: String,
        #[arg(long, value_enum, default_value_t = CliEngine::Whisper)]
        engine: CliEngine,
        #[arg(long)]
        language: Option<String>,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, default_value = "text")]
        response_format: String,
        #[arg(long)]
        timestamps: bool,
        #[arg(long = "dictionary")]
        dictionary: Vec<String>,
    },
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 11435)]
        port: u16,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, value_enum, default_value_t = CliEngine::Whisper)]
        engine: CliEngine,
        #[arg(long)]
        api_key: Option<String>,
        /// Upstream OpenAI-compatible speech endpoint. When set, transcriptions proxy remotely.
        #[arg(long)]
        remote_endpoint: Option<String>,
        #[arg(long)]
        remote_api_key: Option<String>,
        #[arg(long)]
        remote_model: Option<String>,
        /// Enable permissive CORS headers for browser clients.
        #[arg(long)]
        cors: bool,
        /// Deprecated compatibility flag. CORS is disabled by default.
        #[arg(long = "no-cors")]
        no_cors: bool,
    },
}

#[derive(Debug, Subcommand)]
enum ModelsCommand {
    List,
    Install {
        id: String,
        #[arg(long)]
        url: String,
        #[arg(long)]
        artifact: Option<String>,
        #[arg(long)]
        size: Option<u64>,
        #[arg(long)]
        sha256: Option<String>,
    },
    Delete {
        id: String,
    },
}

pub fn run_blocking() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run())
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cache_dir = cli.cache_dir.unwrap_or_else(default_cache_dir);

    match cli.command {
        Command::Models { command } => handle_models(command, cache_dir, cli.json).await,
        Command::Transcribe {
            audio,
            model,
            engine,
            language,
            prompt,
            response_format,
            timestamps,
            dictionary,
        } => {
            let service = SpeechService::new_loose_with_engine(cache_dir, engine.into());
            let response = service.transcribe(TranscribeRequest {
                audio: AudioInput::WavPath(audio),
                model_id: model,
                language,
                prompt,
                dictionary,
                timestamps,
                timestamp_granularity: timestamps.then_some(crate::TimestampGranularity::Segment),
            })?;
            print_transcription_response(response, &response_format, cli.json)?;
            Ok(())
        }
        Command::Serve {
            host,
            port,
            model,
            engine,
            api_key,
            remote_endpoint,
            remote_api_key,
            remote_model,
            cors,
            no_cors,
        } => {
            let cors_enabled = cors && !no_cors;
            let remote_enabled = remote_endpoint
                .as_deref()
                .is_some_and(|endpoint| !endpoint.trim().is_empty());
            let event_sink = serve_event_sink(
                cache_dir.clone(),
                model.clone(),
                engine,
                remote_enabled,
                api_key.as_deref().is_some_and(|key| !key.trim().is_empty()),
                cors_enabled,
            );
            let service = std::sync::Arc::new(
                crate::service::SpeechService::new_loose_with_engine(cache_dir, engine.into()),
            );
            if !remote_enabled {
                if let Some(model_id) = model.as_deref() {
                    let warm = std::sync::Arc::clone(&service);
                    let model_id = model_id.to_string();
                    let label = model_id.clone();
                    tokio::task::spawn_blocking(move || warm.preload_and_warm(&model_id))
                        .await
                        .map_err(|err| anyhow::anyhow!("warm model task failed: {err}"))?
                        .map_err(|err| anyhow::anyhow!("warm model `{label}`: {err}"))?;
                }
            }
            #[cfg(feature = "remote")]
            let transcription_provider = if remote_enabled {
                let endpoint = remote_endpoint
                    .as_deref()
                    .map(str::trim)
                    .unwrap_or_default()
                    .to_string();
                let remote_model = remote_model
                    .or(model.clone())
                    .filter(|value| !value.trim().is_empty());
                Some(std::sync::Arc::new(crate::provider::build_remote_provider(
                    reqwest::Client::new(),
                    crate::remote::RemoteConfig {
                        endpoint,
                        api_key: remote_api_key.unwrap_or_default(),
                        model: remote_model,
                    },
                    std::sync::Arc::clone(&service),
                )))
            } else {
                None
            };
            #[cfg(not(feature = "remote"))]
            let _ = (&remote_api_key, &remote_model);
            #[cfg(not(feature = "remote"))]
            let transcription_provider: Option<
                std::sync::Arc<crate::provider::SpeechProvider>,
            > = None;
            if remote_enabled {
                #[cfg(not(feature = "remote"))]
                {
                    return Err(anyhow::anyhow!(
                        "Remote speech requires the `remote` feature"
                    ));
                }
            }
            crate::api::serve(crate::api::ApiConfig {
                host,
                port,
                service,
                api_key,
                event_sink: Some(event_sink),
                cors: cors_enabled,
                transcription_provider,
                local_models: Vec::new(),
                local_model_source: None,
            })
            .await
        }
    }
}

fn serve_event_sink(
    model_cache_dir: PathBuf,
    warm_model: Option<String>,
    engine: CliEngine,
    remote_enabled: bool,
    api_key_required: bool,
    cors_enabled: bool,
) -> crate::api::ApiEventSink {
    Arc::new(move |event| {
        if let Some(base_url) = event.message.strip_prefix("Local API listening on ") {
            println!(
                "{}",
                serve_banner(
                    base_url,
                    &model_cache_dir,
                    warm_model.as_deref(),
                    engine,
                    remote_enabled,
                    api_key_required,
                    cors_enabled,
                )
            );
        }
    })
}

fn serve_banner(
    base_url: &str,
    model_cache_dir: &Path,
    warm_model: Option<&str>,
    engine: CliEngine,
    remote_enabled: bool,
    api_key_required: bool,
    cors_enabled: bool,
) -> String {
    let auth = if api_key_required {
        "API key required"
    } else {
        "none"
    };
    let cors = if cors_enabled { "enabled" } else { "disabled" };
    let backend = if remote_enabled {
        "remote proxy"
    } else {
        "local"
    };
    let warm = if remote_enabled {
        warm_model.unwrap_or("configured remote model")
    } else {
        warm_model.unwrap_or("none")
    };

    format!(
        "Now serving Glimpse Speech API\n\
         Base URL: {base_url}\n\
         Serving:\n\
         - Models: GET {base_url}/v1/models\n\
         - Transcriptions: POST {base_url}/v1/audio/transcriptions\n\
         Model cache: {}\n\
         Backend: {backend}\n\
         Engine: {}\n\
         Model: {warm}\n\
         Auth: {auth}\n\
         CORS: {cors}",
        model_cache_dir.display(),
        ModelEngine::from(engine),
    )
}

async fn handle_models(
    command: ModelsCommand,
    cache_dir: PathBuf,
    json: bool,
) -> anyhow::Result<()> {
    let manager = ModelInstallManager::new(cache_dir);
    match command {
        ModelsCommand::List => {
            let ids = installed_model_ids(manager.cache_dir());
            if json {
                println!("{}", serde_json::to_string_pretty(&ids)?);
            } else if ids.is_empty() {
                println!("No models installed in {}", manager.cache_dir().display());
            } else {
                for id in ids {
                    println!("{id}");
                }
            }
        }
        ModelsCommand::Install {
            id,
            url,
            artifact,
            size,
            sha256,
        } => {
            let artifact = artifact.unwrap_or_else(|| filename_from_url(&url));
            let spec = InstallSpec {
                id: id.clone(),
                engine: ModelEngine::Whisper,
                layout: None,
                storage: ModelStorage::File {
                    artifact: artifact.clone(),
                },
                files: vec![RemoteFile {
                    url,
                    path: artifact,
                    size_bytes: size,
                    sha256,
                    extract: false,
                }],
                variant: None,
            };
            let status = manager.install(&spec, Default::default()).await?;
            print_status(status, json)?;
        }
        ModelsCommand::Delete { id } => {
            let status = manager.delete(&id)?;
            print_status(status, json)?;
        }
    }
    Ok(())
}

fn installed_model_ids(cache_dir: &Path) -> Vec<String> {
    let mut ids = std::fs::read_dir(cache_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn filename_from_url(url: &str) -> String {
    url.rsplit('/')
        .find(|segment| !segment.is_empty())
        .map(|segment| segment.split(['?', '#']).next().unwrap_or(segment))
        .filter(|name| !name.is_empty())
        .unwrap_or("model.bin")
        .to_string()
}

fn print_status(status: crate::models::ModelStatus, json: bool) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else if status.installed {
        println!("{} installed at {}", status.id, status.directory);
    } else {
        println!(
            "{} not installed; missing: {}",
            status.id,
            status.missing_files.join(", ")
        );
    }
    Ok(())
}

fn print_transcription_response(
    response: Transcription,
    response_format: &str,
    json: bool,
) -> anyhow::Result<()> {
    let response_format = if json && response_format == "text" {
        "json"
    } else {
        response_format
    };

    match response_format {
        "json" => println!("{}", serde_json::json!({ "text": response.text })),
        "verbose_json" => println!("{}", verbose_json(response)?),
        "text" => println!("{}", response.text),
        "srt" => print!("{}", format_srt(&response)),
        "vtt" => print!("{}", format_vtt(&response)),
        other => anyhow::bail!("Unsupported response_format `{other}`"),
    }
    Ok(())
}

use crate::api::{format_srt, format_vtt, verbose_response};

fn verbose_json(response: Transcription) -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(&verbose_response(
        response,
        &[],
    ))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbose_json_falls_back_to_full_text_segment() {
        let response = Transcription {
            text: "hello world".to_string(),
            segments: None,
            words: None,
            model_id: "nemotron_streaming_en".to_string(),
            language: Some("en".to_string()),
            duration_ms: 1_500,
        };

        let json: serde_json::Value =
            serde_json::from_str(&verbose_json(response).unwrap()).unwrap();
        assert_eq!(json["text"], "hello world");
        assert_eq!(json["segments"][0]["text"], "hello world");
        assert_eq!(json["segments"][0]["start"], 0.0);
        assert_eq!(json["segments"][0]["end"], 1.5);
    }
}
