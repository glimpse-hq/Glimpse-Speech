use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::{Parser, Subcommand};

use crate::{
    models::{default_model_cache_dir, ModelInstallManager},
    service::{AudioInput, SpeechConfig, SpeechService, TranscribeRequest, TranscribeResponse},
};

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
        #[arg(long)]
        api_key: Option<String>,
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
    Install { id: String },
    Status { id: String },
    Delete { id: String },
}

pub fn run_blocking() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run())
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cache_dir = cli.cache_dir.unwrap_or_else(default_model_cache_dir);

    match cli.command {
        Command::Models { command } => handle_models(command, cache_dir, cli.json).await,
        Command::Transcribe {
            audio,
            model,
            language,
            prompt,
            response_format,
            timestamps,
            dictionary,
        } => {
            let service = SpeechService::new(SpeechConfig {
                model_cache_dir: cache_dir,
            });
            let response = service.transcribe(TranscribeRequest {
                audio: AudioInput::WavPath(audio),
                model_id: model,
                language,
                prompt,
                dictionary,
                timestamps,
                timestamp_granularity: timestamps
                    .then_some(crate::service::TimestampGranularity::Segment),
            })?;
            print_transcription_response(response, &response_format, cli.json)?;
            Ok(())
        }
        Command::Serve {
            host,
            port,
            model,
            api_key,
            cors,
            no_cors,
        } => {
            let cors_enabled = cors && !no_cors;
            let event_sink = serve_event_sink(
                cache_dir.clone(),
                model.clone(),
                api_key.as_deref().is_some_and(|key| !key.trim().is_empty()),
                cors_enabled,
            );
            crate::api::serve(crate::api::ApiConfig {
                host,
                port,
                model_cache_dir: cache_dir,
                warm_model: model,
                api_key,
                event_sink: Some(event_sink),
                cors: cors_enabled,
            })
            .await
        }
    }
}

fn serve_event_sink(
    model_cache_dir: PathBuf,
    warm_model: Option<String>,
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
    api_key_required: bool,
    cors_enabled: bool,
) -> String {
    let auth = if api_key_required {
        "API key required"
    } else {
        "none"
    };
    let cors = if cors_enabled { "enabled" } else { "disabled" };

    format!(
        "Now serving Glimpse Speech API\n\
         Base URL: {base_url}\n\
         Serving:\n\
         - Models: GET {base_url}/v1/models\n\
         - Model install: POST {base_url}/v1/models/{{id}}/install\n\
         - Transcriptions: POST {base_url}/v1/audio/transcriptions\n\
         Model cache: {}\n\
         Warm model: {}\n\
         Auth: {auth}\n\
         CORS: {cors}",
        model_cache_dir.display(),
        warm_model.unwrap_or("none")
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
            let models = crate::models::list_models();
            if json {
                println!("{}", serde_json::to_string_pretty(models)?);
            } else {
                for model in models {
                    println!("{}\t{}\t{}", model.id, model.engine, model.variant);
                }
            }
        }
        ModelsCommand::Install { id } => {
            let status = manager.install_model(&id, Default::default()).await?;
            print_status(status, json)?;
        }
        ModelsCommand::Status { id } => {
            let status = manager.model_status(&id)?;
            print_status(status, json)?;
        }
        ModelsCommand::Delete { id } => {
            let status = manager.delete_model(&id)?;
            print_status(status, json)?;
        }
    }
    Ok(())
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
    response: TranscribeResponse,
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
        "verbose_json" => println!("{}", verbose_json(&response)?),
        "text" => println!("{}", response.text),
        "srt" => print!("{}", format_srt(&response)),
        "vtt" => print!("{}", format_vtt(&response)),
        other => anyhow::bail!("Unsupported response_format `{other}`"),
    }
    Ok(())
}

fn verbose_json(response: &TranscribeResponse) -> anyhow::Result<String> {
    let segments = caption_segments(response)
        .into_iter()
        .enumerate()
        .map(|(id, segment)| {
            serde_json::json!({
                "id": id,
                "seek": 0,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0
            })
        })
        .collect::<Vec<_>>();

    Ok(serde_json::to_string_pretty(&serde_json::json!({
        "task": "transcribe",
        "language": response.language,
        "duration": response.duration_ms as f32 / 1000.0,
        "text": response.text,
        "segments": segments
    }))?)
}

fn format_srt(response: &TranscribeResponse) -> String {
    caption_segments(response)
        .into_iter()
        .enumerate()
        .map(|(idx, segment)| {
            format!(
                "{}\n{} --> {}\n{}\n",
                idx + 1,
                format_timestamp(segment.start, ','),
                format_timestamp(segment.end, ','),
                segment.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_vtt(response: &TranscribeResponse) -> String {
    let cues = caption_segments(response)
        .into_iter()
        .map(|segment| {
            format!(
                "{} --> {}\n{}",
                format_timestamp(segment.start, '.'),
                format_timestamp(segment.end, '.'),
                segment.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    format!("WEBVTT\n\n{cues}\n")
}

fn caption_segments(response: &TranscribeResponse) -> Vec<crate::TranscriptionSegment> {
    let mut segments = response.segments.clone().unwrap_or_default();
    if segments.is_empty() && !response.text.is_empty() {
        segments.push(crate::TranscriptionSegment {
            start: 0.0,
            end: response.duration_ms as f32 / 1000.0,
            text: response.text.clone(),
        });
    }
    segments
}

fn format_timestamp(seconds: f32, decimal_separator: char) -> String {
    let millis = (seconds.max(0.0) * 1000.0).round() as u64;
    let hours = millis / 3_600_000;
    let minutes = (millis % 3_600_000) / 60_000;
    let secs = (millis % 60_000) / 1000;
    let ms = millis % 1000;
    format!("{hours:02}:{minutes:02}:{secs:02}{decimal_separator}{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbose_json_falls_back_to_full_text_segment() {
        let response = TranscribeResponse {
            text: "hello world".to_string(),
            segments: None,
            model_id: "nemotron_streaming_en".to_string(),
            language: Some("en".to_string()),
            duration_ms: 1_500,
        };

        let json: serde_json::Value =
            serde_json::from_str(&verbose_json(&response).unwrap()).unwrap();
        assert_eq!(json["text"], "hello world");
        assert_eq!(json["segments"][0]["text"], "hello world");
        assert_eq!(json["segments"][0]["start"], 0.0);
        assert_eq!(json["segments"][0]["end"], 1.5);
    }
}
