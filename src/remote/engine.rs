use std::{path::Path, time::Duration};

use reqwest::{header::RETRY_AFTER, multipart, Client};
use serde::Deserialize;
use tokio_util::io::ReaderStream;

use super::provider::{
    apply_auth, build_transcription_form, plan_request, resolve_profile, DurationSource,
    EndpointProfile, TranscriptionFormParams,
};
use super::{
    config_error, parse_retry_after, parse_upstream_error, transport_error, RemoteError,
    RemoteErrorKind, ResponseFormat,
};
use crate::{TimestampGranularity, Transcription};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const MODELS_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct RemoteConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RemoteRequestParams<'a> {
    pub model: &'a str,
    pub language: Option<&'a str>,
    pub dictionary: &'a [String],
    pub prompt: Option<&'a str>,
    pub timestamps: bool,
    pub timestamp_granularity: Option<TimestampGranularity>,
}

pub struct RemoteEngine {
    client: Client,
    config: RemoteConfig,
}

impl RemoteEngine {
    pub fn new(client: Client, config: RemoteConfig) -> Self {
        Self { client, config }
    }

    pub fn config(&self) -> &RemoteConfig {
        &self.config
    }

    pub async fn transcribe_file(
        &self,
        audio_path: &Path,
        params: RemoteRequestParams<'_>,
    ) -> Result<Transcription, RemoteError> {
        let endpoint = self.config.endpoint.trim();
        if endpoint.is_empty() {
            return Err(config_error("Remote speech endpoint is not configured"));
        }
        let model = params.model.trim();
        if model.is_empty() {
            return Err(config_error("Remote speech model is not configured"));
        }

        let profile = resolve_profile(endpoint);
        let plan = plan_request(&profile, params.timestamps, params.timestamp_granularity);
        let url = format!("{}/audio/transcriptions", api_base(endpoint));

        let extension = audio_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_ascii_lowercase);
        let mime_type = audio_mime_for_extension(extension.as_deref());
        let file_name = audio_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("recording.wav")
            .to_string();
        let api_key = self.config.api_key.trim();
        let language = params
            .language
            .map(str::trim)
            .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("auto"))
            .map(str::to_string);

        let mut effective_format = plan.response_format;
        let mut granularities = plan.timestamp_granularities.clone();
        let body = loop {
            let file = tokio::fs::File::open(audio_path).await.map_err(|err| {
                transport_error(format!(
                    "Failed to read recording at {}: {err}",
                    audio_path.display()
                ))
            })?;
            let stream = ReaderStream::new(file);
            let file_part = multipart::Part::stream(reqwest::Body::wrap_stream(stream))
                .file_name(file_name.clone())
                .mime_str(mime_type)
                .map_err(|err| transport_error(format!("Failed to prepare audio upload: {err}")))?;
            let form = build_transcription_form(
                &profile,
                file_part,
                TranscriptionFormParams {
                    model,
                    response_format: effective_format,
                    timestamp_granularities: &granularities,
                    language: language.as_deref(),
                    dictionary: params.dictionary,
                    prompt: params.prompt,
                },
            );

            let builder = apply_auth(
                self.client
                    .post(&url)
                    .multipart(form)
                    .timeout(DEFAULT_TIMEOUT),
                api_key,
            );

            let response = builder.send().await.map_err(|err| {
                transport_error(format!("Failed to reach remote speech endpoint: {err}"))
            })?;
            let status = response.status();
            let retry_after = parse_retry_after(response.headers().get(RETRY_AFTER));
            let body_text = response.text().await.map_err(|err| {
                transport_error(format!("Failed to read remote speech response: {err}"))
            })?;
            if status.is_success() {
                break body_text;
            }
            let err = parse_upstream_error(status, retry_after, &body_text);
            if profile.sends_response_format
                && effective_format == ResponseFormat::VerboseJson
                && is_verbose_unsupported(&err)
            {
                effective_format = ResponseFormat::Json;
                if profile.keep_timestamps_on_format_fallback {
                    granularities = vec!["segment"];
                } else {
                    granularities.clear();
                }
                continue;
            }
            return Err(err);
        };

        parse_transcription_body(&body, model, &profile)
    }

    pub async fn list_models(&self) -> Result<Vec<String>, RemoteError> {
        let endpoint = self.config.endpoint.trim();
        if endpoint.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/models", api_base(endpoint));
        let builder = apply_auth(
            self.client.get(url).timeout(MODELS_TIMEOUT),
            self.config.api_key.trim(),
        );

        let response = builder.send().await.map_err(|err| {
            transport_error(format!(
                "Failed to reach remote speech models endpoint: {err}"
            ))
        })?;
        let status = response.status();
        let retry_after = parse_retry_after(response.headers().get(RETRY_AFTER));
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(parse_upstream_error(status, retry_after, &body));
        }

        let parsed: ModelsResponse = response.json().await.map_err(|err| {
            transport_error(format!(
                "Failed to parse remote speech models response: {err}"
            ))
        })?;
        Ok(parsed.data.into_iter().map(|entry| entry.id).collect())
    }
}

#[derive(Debug, Deserialize)]
struct TranscriptionBody {
    #[serde(default)]
    text: String,
    #[serde(default)]
    segments: Option<Vec<UpstreamSegment>>,
    #[serde(default)]
    words: Option<Vec<UpstreamSegment>>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration: Option<f32>,
    #[serde(default)]
    usage: Option<UsageBody>,
}

#[derive(Debug, Deserialize)]
struct UsageBody {
    #[serde(default)]
    prompt_audio_seconds: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct UpstreamSegment {
    #[serde(default)]
    start: f32,
    #[serde(default)]
    end: f32,
    #[serde(default)]
    text: String,
    #[serde(default)]
    word: String,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
}

fn parse_transcription_body(
    body: &str,
    model: &str,
    profile: &EndpointProfile,
) -> Result<Transcription, RemoteError> {
    let parsed = serde_json::from_str::<TranscriptionBody>(body).map_err(|err| RemoteError {
        kind: RemoteErrorKind::Other,
        status: 200,
        message: format!("Failed to parse remote speech response: {err}"),
        error_type: None,
        code: None,
        param: None,
        retry_after: None,
    })?;
    let duration_seconds = match profile.duration_source {
        DurationSource::TopLevel => parsed.duration,
        DurationSource::UsagePromptAudioSeconds => parsed
            .usage
            .as_ref()
            .and_then(|usage| usage.prompt_audio_seconds),
    };
    let segments = map_timed_text(parsed.segments);
    let words = if profile.supports_word_timestamps {
        map_timed_text(parsed.words)
    } else {
        None
    };
    Ok(Transcription {
        text: parsed.text,
        segments,
        words,
        model_id: model.to_string(),
        language: parsed.language,
        duration_ms: duration_seconds
            .map(|seconds| (seconds.max(0.0) * 1000.0) as u128)
            .unwrap_or(0),
    })
}

fn map_timed_text(items: Option<Vec<UpstreamSegment>>) -> Option<Vec<crate::TranscriptionSegment>> {
    items.map(|items| {
        items
            .into_iter()
            .map(|item| crate::TranscriptionSegment {
                start: item.start,
                end: item.end,
                text: upstream_segment_text(&item),
            })
            .collect()
    })
}

fn upstream_segment_text(item: &UpstreamSegment) -> String {
    if !item.text.is_empty() {
        item.text.clone()
    } else {
        item.word.clone()
    }
}

fn is_verbose_unsupported(err: &RemoteError) -> bool {
    if err.kind != RemoteErrorKind::InvalidRequest {
        return false;
    }
    let needles = ["verbose_json", "response_format", "timestamp_granularit"];
    let haystack = [
        err.message.as_str(),
        err.param.as_deref().unwrap_or(""),
        err.code.as_deref().unwrap_or(""),
        err.error_type.as_deref().unwrap_or(""),
    ];
    haystack.iter().any(|field| {
        let lowered = field.to_ascii_lowercase();
        needles.iter().any(|needle| lowered.contains(needle))
    })
}

fn audio_mime_for_extension(extension: Option<&str>) -> &'static str {
    match extension {
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("m4a") | Some("mp4") => "audio/mp4",
        Some("aac") => "audio/aac",
        Some("flac") => "audio/flac",
        Some("ogg") | Some("oga") => "audio/ogg",
        Some("opus") => "audio/opus",
        Some("webm") => "audio/webm",
        Some("mpga") | Some("mpeg") => "audio/mpeg",
        _ => "application/octet-stream",
    }
}

fn api_base(endpoint: &str) -> String {
    let mut base = endpoint.trim().trim_end_matches('/').to_string();
    for suffix in ["/v1/audio/transcriptions", "/audio/transcriptions"] {
        if base.ends_with(suffix) {
            base.truncate(base.len() - suffix.len());
            break;
        }
    }
    let base = base.trim_end_matches('/').to_string();
    if base.is_empty() || base.ends_with("/v1") {
        base
    } else {
        format!("{base}/v1")
    }
}
