use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, anyhow};
use clap::builder::TypedValueParser;
use clap::{Parser, Subcommand};
use directories::ProjectDirs;

use crate::{
    TimestampGranularity, Transcription,
    api::{
        ApiConfig, ApiEventSink, ApiModelInfo, ResponseFormat, format_srt, format_vtt,
        verbose_response,
    },
    models::{
        InstallSpec, ModelEngine, ModelInstallManager, ModelStatus, ModelStorage, RemoteFile,
    },
    service::{AudioInput, SpeechService, TranscribeRequest},
};

const CACHE_DIR_ENV: &str = "GLIMPSE_SPEECH_CACHE_DIR";
const LISTENING_PREFIX: &str = "Local API listening on ";

fn default_cache_dir() -> PathBuf {
    if let Ok(value) = std::env::var(CACHE_DIR_ENV) {
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

    ProjectDirs::from("com", "Glimpse", "glimpse-speech")
        .map_or_else(
            || PathBuf::from("glimpse-speech"),
            |dirs| dirs.data_local_dir().to_path_buf(),
        )
        .join("models")
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
        #[arg(long, value_enum, default_value_t = ModelEngine::Whisper)]
        engine: ModelEngine,
        #[arg(long)]
        language: Option<String>,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(
            long,
            default_value = "text",
            value_parser = clap::builder::PossibleValuesParser::new(["text", "json", "verbose_json", "srt", "vtt"])
                .try_map(|value| value.parse::<ResponseFormat>()),
        )]
        response_format: ResponseFormat,
        #[arg(long)]
        timestamps: bool,
        #[arg(long = "dictionary")]
        dictionary: Vec<String>,
    },
    Serve(ServeArgs),
}

#[derive(Debug, clap::Args)]
struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 11435)]
    port: u16,
    #[arg(long)]
    model: Option<String>,
    #[arg(long, value_enum, default_value_t = ModelEngine::Whisper)]
    engine: ModelEngine,
    #[arg(long, env = "GLIMPSE_SPEECH_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    /// Upstream speech endpoint (OpenAI-compatible, ElevenLabs, or Deepgram). When set, transcriptions proxy remotely.
    #[arg(long)]
    remote_endpoint: Option<String>,
    #[arg(long, env = "GLIMPSE_SPEECH_REMOTE_API_KEY", hide_env_values = true)]
    remote_api_key: Option<String>,
    #[arg(long)]
    remote_model: Option<String>,
    /// Enable permissive CORS headers for browser clients.
    #[arg(long)]
    cors: bool,
    #[arg(long = "no-cors", hide = true)]
    no_cors: bool,
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
            let format = match response_format {
                ResponseFormat::Text if cli.json => ResponseFormat::Json,
                format => format,
            };
            let timestamps = timestamps || format.needs_timestamps();
            let service = SpeechService::new_loose_with_engine(cache_dir, engine);
            let response = service.transcribe(TranscribeRequest {
                audio: AudioInput::WavPath(audio),
                model_id: model,
                language,
                prompt,
                dictionary,
                timestamps,
                timestamp_granularity: timestamps.then_some(TimestampGranularity::Segment),
            })?;
            print_transcription_response(response, format)
        }
        Command::Serve(args) => serve(args, cache_dir).await,
    }
}

async fn serve(args: ServeArgs, cache_dir: PathBuf) -> anyhow::Result<()> {
    let ServeArgs {
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
    } = args;
    let cors_enabled = cors && !no_cors;
    let remote_endpoint = remote_endpoint
        .as_deref()
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty())
        .map(str::to_string);
    let remote_enabled = remote_endpoint.is_some();
    let api_key_required = api_key.as_deref().is_some_and(|key| !key.trim().is_empty());

    let event_sink = serve_event_sink(ServeBanner {
        model_cache_dir: cache_dir.clone(),
        warm_model: model.clone(),
        engine,
        remote_enabled,
        api_key_required,
        cors_enabled,
    });
    let installed_models_dir = cache_dir.clone();
    let service = Arc::new(SpeechService::new_loose_with_engine(cache_dir, engine));

    if !remote_enabled && let Some(model_id) = model.clone() {
        let warm = Arc::clone(&service);
        let label = model_id.clone();
        tokio::task::spawn_blocking(move || warm.preload_and_warm(&model_id))
            .await
            .map_err(|err| anyhow!("warm model task failed: {err}"))?
            .with_context(|| format!("warm model `{label}`"))?;
    }

    let transcription_provider = match remote_endpoint {
        #[cfg(feature = "remote")]
        Some(endpoint) => Some(Arc::new(crate::provider::build_remote_provider(
            reqwest::Client::new(),
            crate::remote::RemoteConfig {
                endpoint,
                api_key: remote_api_key.unwrap_or_default(),
                model: remote_model
                    .or(model)
                    .filter(|value| !value.trim().is_empty()),
            },
            Arc::clone(&service),
        ))),
        #[cfg(not(feature = "remote"))]
        Some(_) => {
            let _ = (remote_api_key, remote_model);
            anyhow::bail!("Remote speech requires the `remote` feature");
        }
        None => None,
    };

    crate::api::serve(ApiConfig {
        host,
        port,
        service,
        api_key,
        event_sink: Some(event_sink),
        cors: cors_enabled,
        transcription_provider,
        local_models: Vec::new(),
        local_model_source: Some(Arc::new(move || {
            installed_model_ids(&installed_models_dir)
                .into_iter()
                .map(|id| {
                    ApiModelInfo::new(
                        id.clone(),
                        id,
                        "Installed in the local model cache.".to_string(),
                        vec!["Local".to_string()],
                        Vec::new(),
                    )
                })
                .collect()
        })),
    })
    .await
}

struct ServeBanner {
    model_cache_dir: PathBuf,
    warm_model: Option<String>,
    engine: ModelEngine,
    remote_enabled: bool,
    api_key_required: bool,
    cors_enabled: bool,
}

impl ServeBanner {
    fn render(&self, base_url: &str) -> String {
        let auth = if self.api_key_required {
            "API key required"
        } else {
            "none"
        };
        let cors = if self.cors_enabled {
            "enabled"
        } else {
            "disabled"
        };
        let backend = if self.remote_enabled {
            "remote proxy"
        } else {
            "local"
        };
        let warm = self
            .warm_model
            .as_deref()
            .unwrap_or(if self.remote_enabled {
                "configured remote model"
            } else {
                "none"
            });

        format!(
            "Now serving Glimpse Speech API\n\
             Base URL: {base_url}\n\
             Serving:\n\
             - Models: GET {base_url}/v1/models\n\
             - Transcriptions: POST {base_url}/v1/audio/transcriptions\n\
             - Health: GET {base_url}/health\n\
             Model cache: {}\n\
             Backend: {backend}\n\
             Engine: {}\n\
             Model: {warm}\n\
             Auth: {auth}\n\
             CORS: {cors}",
            self.model_cache_dir.display(),
            self.engine,
        )
    }
}

fn serve_event_sink(banner: ServeBanner) -> ApiEventSink {
    Arc::new(move |event| {
        if let Some(base_url) = event.message.strip_prefix(LISTENING_PREFIX) {
            println!("{}", banner.render(base_url));
        }
    })
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
                id,
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
            print_status(&status, json)?;
        }
        ModelsCommand::Delete { id } => {
            let status = manager.delete(&id)?;
            print_status(&status, json)?;
        }
    }
    Ok(())
}

fn installed_model_ids(cache_dir: &Path) -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(cache_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
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

fn print_status(status: &ModelStatus, json: bool) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(status)?);
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
    format: ResponseFormat,
) -> anyhow::Result<()> {
    match format {
        ResponseFormat::Json => println!("{}", serde_json::json!({ "text": response.text })),
        ResponseFormat::VerboseJson => println!(
            "{}",
            serde_json::to_string_pretty(&verbose_response(response, &[]))?
        ),
        ResponseFormat::Text => println!("{}", response.text),
        ResponseFormat::Srt => print!("{}", format_srt(&response)),
        ResponseFormat::Vtt => print!("{}", format_vtt(&response)),
    }
    Ok(())
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

        let json = serde_json::to_value(verbose_response(response, &[])).unwrap();
        assert_eq!(json["text"], "hello world");
        assert_eq!(json["segments"][0]["text"], "hello world");
        assert_eq!(json["segments"][0]["start"], 0.0);
        assert_eq!(json["segments"][0]["end"], 1.5);
    }

    #[test]
    fn filename_from_url_strips_query_and_fragment() {
        assert_eq!(
            filename_from_url("https://example.test/models/ggml-base.bin?download=1#top"),
            "ggml-base.bin"
        );
        assert_eq!(filename_from_url("https://example.test/"), "example.test");
        assert_eq!(filename_from_url(""), "model.bin");
    }
}
