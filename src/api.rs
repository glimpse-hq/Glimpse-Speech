use std::{
    future::Future,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Context, Result};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, FromRequest, Multipart, Path, Request, State},
    http::{header::CONTENT_TYPE, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde::Serialize;
use tower_http::cors::CorsLayer;

use crate::{
    models::{InstallOptions, ModelDownloadProgress, ModelManifest},
    service::{
        AudioInput, SpeechConfig, SpeechService,
        TimestampGranularity as ServiceTimestampGranularity, TranscribeRequest, TranscribeResponse,
    },
};

#[derive(Clone)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub model_cache_dir: PathBuf,
    pub warm_model: Option<String>,
    pub api_key: Option<String>,
    pub event_sink: Option<ApiEventSink>,
    /// When true, responses include permissive CORS headers so browser-based
    /// clients on any origin can call the API.
    pub cors: bool,
}

#[derive(Clone)]
struct ApiState {
    service: Arc<SpeechService>,
    api_key: Option<Arc<str>>,
    event_sink: Option<ApiEventSink>,
}

pub type ApiEventSink = Arc<dyn Fn(ApiEvent) + Send + Sync + 'static>;

#[derive(Debug, Clone, Serialize)]
pub struct ApiEvent {
    pub level: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    param: Option<String>,
    code: Option<String>,
}

#[derive(Debug, Serialize)]
struct InstallResponse {
    status: crate::models::ModelStatus,
    progress: Vec<ModelDownloadProgress>,
}

struct ParsedTranscriptionRequest {
    request: TranscribeRequest,
    response_format: ResponseFormat,
    timestamp_granularities: Vec<TimestampGranularity>,
    uploaded_file: Option<PathBuf>,
}

struct TempUpload {
    path: PathBuf,
}

impl TempUpload {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }

    fn into_path(self) -> PathBuf {
        let path = self.path.clone();
        std::mem::forget(self);
        path
    }
}

impl Drop for TempUpload {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseFormat {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimestampGranularity {
    Segment,
    Word,
}

#[derive(Debug, Serialize)]
struct JsonTranscriptionResponse {
    text: String,
}

#[derive(Debug, Serialize)]
struct VerboseTranscriptionResponse {
    task: &'static str,
    language: Option<String>,
    duration: f32,
    text: String,
    segments: Vec<VerboseSegment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    words: Option<Vec<VerboseWord>>,
}

#[derive(Debug, Serialize)]
struct VerboseSegment {
    id: usize,
    seek: usize,
    start: f32,
    end: f32,
    text: String,
    tokens: Vec<i32>,
    temperature: f32,
    avg_logprob: f32,
    compression_ratio: f32,
    no_speech_prob: f32,
}

#[derive(Debug, Serialize)]
struct VerboseWord {
    word: String,
    start: f32,
    end: f32,
}

pub async fn serve(config: ApiConfig) -> Result<()> {
    serve_with_shutdown(config, std::future::pending::<()>()).await
}

pub async fn serve_with_shutdown(
    config: ApiConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let cors_enabled = config.cors;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let service = Arc::new(SpeechService::new(SpeechConfig {
        model_cache_dir: config.model_cache_dir,
    }));
    if let Some(model_id) = config.warm_model.as_deref() {
        let service = Arc::clone(&service);
        let model_id = model_id.to_string();
        let warm_label = model_id.clone();
        tokio::task::spawn_blocking(move || service.preload_and_warm(&model_id))
            .await
            .map_err(|err| anyhow!("warm model task failed: {err}"))?
            .with_context(|| format!("warm model `{warm_label}`"))?;
    }

    let state = ApiState {
        service,
        api_key: config
            .api_key
            .filter(|key| !key.trim().is_empty())
            .map(Arc::from),
        event_sink: config.event_sink,
    };
    state.log("info", format!("Local API listening on http://{addr}"));

    let mut app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/models/{id}/install", post(install_model))
        .route("/v1/models/{id}", delete(delete_model))
        .route("/v1/audio/transcriptions", post(transcribe))
        .layer(DefaultBodyLimit::max(1024 * 1024 * 1024))
        .with_state(state);

    if cors_enabled {
        app = app.layer(CorsLayer::permissive());
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

async fn list_models(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<&'static [ModelManifest]>, (StatusCode, Json<ErrorBody>)> {
    authorize(&state, &headers)?;
    state.log("info", "GET /v1/models".to_string());
    Ok(Json(crate::models::list_models()))
}

async fn install_model(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<InstallResponse>, (StatusCode, Json<ErrorBody>)> {
    authorize(&state, &headers)?;
    state.log("info", format!("POST /v1/models/{id}/install"));
    let progress = Arc::new(Mutex::new(Vec::new()));
    let progress_events = Arc::clone(&progress);
    let callback = |event: ModelDownloadProgress| {
        if let Ok(mut guard) = progress_events.lock() {
            guard.push(event);
        }
    };

    let status = state
        .service
        .model_manager()
        .install_model(
            &id,
            InstallOptions {
                cancel_token: None,
                progress: Some(&callback),
            },
        )
        .await
        .map_err(map_error)?;

    let progress = progress
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_default();
    Ok(Json(InstallResponse { status, progress }))
}

async fn delete_model(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<crate::models::ModelStatus>, (StatusCode, Json<ErrorBody>)> {
    authorize(&state, &headers)?;
    state.log("info", format!("DELETE /v1/models/{id}"));
    state
        .service
        .model_manager()
        .delete_model(&id)
        .map(Json)
        .map_err(map_error)
}

async fn transcribe(
    State(state): State<ApiState>,
    headers: HeaderMap,
    request: Request<Body>,
) -> Result<Response, (StatusCode, Json<ErrorBody>)> {
    authorize(&state, &headers)?;
    state.log("info", "POST /v1/audio/transcriptions".to_string());
    let content_type = headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();

    if !content_type.starts_with("multipart/form-data") {
        return Err(map_error(anyhow!(
            "Use multipart/form-data with a `file` field for transcription"
        )));
    }

    let parsed = transcribe_request_from_multipart(request, &state).await?;

    let response_format = parsed.response_format;
    let timestamp_granularities = parsed.timestamp_granularities.clone();
    let uploaded_file = parsed.uploaded_file.clone();
    let service = Arc::clone(&state.service);
    let result = tokio::task::spawn_blocking(move || service.transcribe(parsed.request))
        .await
        .map_err(|err| map_error(anyhow!("transcription task failed: {err}")))?
        .map(|response| {
            state.log("info", format!("Transcribed with {}", response.model_id));
            state.log_model(
                "info",
                format!("Loaded model {}", response.model_id),
                &response.model_id,
            );
            format_transcription_response(response, response_format, &timestamp_granularities)
        })
        .map_err(map_error);

    if let Some(path) = uploaded_file {
        let _ = std::fs::remove_file(path);
    }

    result
}

impl ApiState {
    fn log(&self, level: &'static str, message: String) {
        if let Some(sink) = &self.event_sink {
            sink(ApiEvent {
                level,
                message,
                model_id: None,
            });
        }
    }

    fn log_model(&self, level: &'static str, message: String, model_id: &str) {
        if let Some(sink) = &self.event_sink {
            sink(ApiEvent {
                level,
                message,
                model_id: Some(model_id.to_string()),
            });
        }
    }
}

fn authorize(state: &ApiState, headers: &HeaderMap) -> Result<(), (StatusCode, Json<ErrorBody>)> {
    let Some(expected) = &state.api_key else {
        return Ok(());
    };

    let bearer = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim);
    let api_key = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(str::trim);

    if bearer == Some(expected.as_ref()) || api_key == Some(expected.as_ref()) {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            Json(error_body("Missing or invalid API key")),
        ))
    }
}

fn build_transcription_request(
    model: String,
    audio: AudioInput,
    language: Option<String>,
    prompt: Option<String>,
    response_format: Option<&str>,
    timestamp_granularities: Vec<String>,
    dictionary: Vec<String>,
    timestamps: bool,
    stream: bool,
    uploaded_file: Option<PathBuf>,
) -> Result<ParsedTranscriptionRequest, (StatusCode, Json<ErrorBody>)> {
    if stream {
        return Err(map_error(anyhow!(
            "Streaming transcription responses are not supported"
        )));
    }

    let response_format = parse_response_format(response_format.unwrap_or("json"))?;
    let timestamp_granularities = parse_timestamp_granularities(timestamp_granularities)?;
    if !timestamp_granularities.is_empty() && response_format != ResponseFormat::VerboseJson {
        return Err(map_error(anyhow!(
            "`timestamp_granularities` requires response_format `verbose_json`"
        )));
    }

    Ok(ParsedTranscriptionRequest {
        request: TranscribeRequest {
            audio,
            model_id: model,
            language,
            prompt,
            dictionary,
            timestamps: timestamps || !timestamp_granularities.is_empty(),
            timestamp_granularity: service_timestamp_granularity(&timestamp_granularities),
        },
        response_format,
        timestamp_granularities,
        uploaded_file,
    })
}

fn service_timestamp_granularity(
    values: &[TimestampGranularity],
) -> Option<ServiceTimestampGranularity> {
    if values.contains(&TimestampGranularity::Word) {
        Some(ServiceTimestampGranularity::Word)
    } else if values.contains(&TimestampGranularity::Segment) {
        Some(ServiceTimestampGranularity::Segment)
    } else {
        None
    }
}

fn parse_response_format(value: &str) -> Result<ResponseFormat, (StatusCode, Json<ErrorBody>)> {
    match value {
        "json" => Ok(ResponseFormat::Json),
        "text" => Ok(ResponseFormat::Text),
        "verbose_json" => Ok(ResponseFormat::VerboseJson),
        "srt" => Ok(ResponseFormat::Srt),
        "vtt" => Ok(ResponseFormat::Vtt),
        other => Err(map_error(anyhow!("Unsupported response_format `{other}`"))),
    }
}

fn parse_timestamp_granularities(
    values: Vec<String>,
) -> Result<Vec<TimestampGranularity>, (StatusCode, Json<ErrorBody>)> {
    let mut parsed = Vec::new();
    for value in values {
        for entry in split_field_values(&value) {
            let granularity = match entry.as_str() {
                "segment" => TimestampGranularity::Segment,
                "word" => TimestampGranularity::Word,
                other => {
                    return Err(map_error(anyhow!(
                        "Unsupported timestamp granularity `{other}`"
                    )))
                }
            };
            if !parsed.contains(&granularity) {
                parsed.push(granularity);
            }
        }
    }
    Ok(parsed)
}

fn format_transcription_response(
    response: TranscribeResponse,
    format: ResponseFormat,
    timestamp_granularities: &[TimestampGranularity],
) -> Response {
    match format {
        ResponseFormat::Json => Json(JsonTranscriptionResponse {
            text: response.text,
        })
        .into_response(),
        ResponseFormat::VerboseJson => {
            Json(verbose_response(response, timestamp_granularities)).into_response()
        }
        ResponseFormat::Text => text_response(response.text, "text/plain; charset=utf-8"),
        ResponseFormat::Srt => text_response(format_srt(&response), "application/x-subrip"),
        ResponseFormat::Vtt => text_response(format_vtt(&response), "text/vtt; charset=utf-8"),
    }
}

fn verbose_response(
    response: TranscribeResponse,
    timestamp_granularities: &[TimestampGranularity],
) -> VerboseTranscriptionResponse {
    let mut segments = verbose_segments(&response);
    if segments.is_empty() && !response.text.is_empty() {
        segments.push(VerboseSegment {
            id: 0,
            seek: 0,
            start: 0.0,
            end: response.duration_ms as f32 / 1000.0,
            text: response.text.clone(),
            tokens: Vec::new(),
            temperature: 0.0,
            avg_logprob: 0.0,
            compression_ratio: 0.0,
            no_speech_prob: 0.0,
        });
    }

    let words = timestamp_granularities
        .contains(&TimestampGranularity::Word)
        .then(|| verbose_words(&response));
    let duration = segments
        .last()
        .map(|segment| segment.end)
        .unwrap_or(response.duration_ms as f32 / 1000.0);

    VerboseTranscriptionResponse {
        task: "transcribe",
        language: response.language,
        duration,
        text: response.text,
        segments,
        words,
    }
}

fn verbose_words(response: &TranscribeResponse) -> Vec<VerboseWord> {
    response
        .segments
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|segment| VerboseWord {
            word: segment.text.trim().to_string(),
            start: segment.start,
            end: segment.end,
        })
        .filter(|word| !word.word.is_empty())
        .collect()
}

fn verbose_segments(response: &TranscribeResponse) -> Vec<VerboseSegment> {
    response
        .segments
        .as_deref()
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(id, segment)| VerboseSegment {
            id,
            seek: 0,
            start: segment.start,
            end: segment.end,
            text: segment.text.clone(),
            tokens: Vec::new(),
            temperature: 0.0,
            avg_logprob: 0.0,
            compression_ratio: 0.0,
            no_speech_prob: 0.0,
        })
        .collect()
}

fn text_response(body: String, content_type: &'static str) -> Response {
    ([(CONTENT_TYPE, content_type)], body).into_response()
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

async fn transcribe_request_from_multipart(
    request: Request<Body>,
    state: &ApiState,
) -> Result<ParsedTranscriptionRequest, (StatusCode, Json<ErrorBody>)> {
    let mut multipart = Multipart::from_request(request, state)
        .await
        .map_err(|err| map_error(anyhow!(err.to_string())))?;
    let mut model = None;
    let mut language = None;
    let mut prompt = None;
    let mut response_format = None;
    let mut dictionary = Vec::new();
    let mut timestamp_granularities = Vec::new();
    let mut timestamps = false;
    let mut stream = false;
    let mut uploaded_file: Option<TempUpload> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| map_error(anyhow!(err.to_string())))?
    {
        let Some(name) = field.name().map(str::to_string) else {
            continue;
        };

        match name.as_str() {
            "file" | "audio" => {
                let extension = field
                    .file_name()
                    .and_then(|name| PathBuf::from(name).extension().map(|ext| ext.to_owned()));
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|err| map_error(anyhow!(err.to_string())))?;
                let path = temp_audio_path(extension.as_deref());
                tokio::fs::write(&path, &bytes)
                    .await
                    .with_context(|| format!("write uploaded audio to {}", path.display()))
                    .map_err(map_error)?;
                uploaded_file = Some(TempUpload::new(path));
            }
            "model" => model = Some(field_text(field).await?),
            "language" => {
                let value = field_text(field).await?;
                if !value.trim().is_empty() {
                    language = Some(value);
                }
            }
            "prompt" => {
                let value = field_text(field).await?;
                if !value.trim().is_empty() {
                    prompt = Some(value);
                }
            }
            "response_format" => response_format = Some(field_text(field).await?),
            "dictionary" | "dictionary[]" | "glimpse_dictionary" => {
                dictionary.extend(split_field_values(&field_text(field).await?));
            }
            "timestamp_granularities" | "timestamp_granularities[]" => {
                timestamp_granularities.extend(split_field_values(&field_text(field).await?));
            }
            "timestamps" => timestamps = parse_bool(&field_text(field).await?),
            "stream" => stream = parse_bool(&field_text(field).await?),
            "temperature" => {
                let _ = field_text(field).await?;
            }
            _ => {}
        }
    }

    let upload =
        uploaded_file.ok_or_else(|| map_error(anyhow!("Missing multipart file field `file`")))?;
    let audio_path = upload.path().clone();
    let model = model.ok_or_else(|| map_error(anyhow!("Missing multipart field `model`")))?;

    let mut parsed = build_transcription_request(
        model,
        AudioInput::WavPath(audio_path),
        language,
        prompt,
        response_format.as_deref(),
        timestamp_granularities,
        dictionary,
        timestamps,
        stream,
        Some(upload.path().clone()),
    )?;
    parsed.uploaded_file = Some(upload.into_path());
    Ok(parsed)
}

async fn field_text(
    field: axum::extract::multipart::Field<'_>,
) -> Result<String, (StatusCode, Json<ErrorBody>)> {
    field
        .text()
        .await
        .map_err(|err| map_error(anyhow!(err.to_string())))
}

fn split_field_values(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_bool(value: &str) -> bool {
    matches!(value.trim(), "true" | "1" | "yes" | "on")
}

fn map_error(error: anyhow::Error) -> (StatusCode, Json<ErrorBody>) {
    (StatusCode::BAD_REQUEST, Json(error_body(error.to_string())))
}

fn error_body(message: impl Into<String>) -> ErrorBody {
    ErrorBody {
        error: ErrorDetail {
            message: message.into(),
            error_type: "invalid_request_error",
            param: None,
            code: None,
        },
    }
}

fn temp_audio_path(extension: Option<&std::ffi::OsStr>) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let mut path = std::env::temp_dir().join(format!(
        "glimpse-speech-upload-{}-{timestamp}",
        std::process::id()
    ));
    if let Some(extension) = extension {
        path.set_extension(extension);
    } else {
        path.set_extension("wav");
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_response() -> TranscribeResponse {
        TranscribeResponse {
            text: "hello world".to_string(),
            segments: Some(vec![crate::TranscriptionSegment {
                start: 1.25,
                end: 2.5,
                text: "hello world".to_string(),
            }]),
            model_id: "whisper_small_q5".to_string(),
            language: Some("en".to_string()),
            duration_ms: 1250,
        }
    }

    #[test]
    fn parses_openai_timestamp_granularity_arrays() {
        let parsed =
            parse_timestamp_granularities(vec!["segment".to_string(), "word,segment".to_string()])
                .unwrap();
        assert_eq!(
            parsed,
            vec![TimestampGranularity::Segment, TimestampGranularity::Word]
        );
    }

    #[test]
    fn rejects_timestamp_granularity_without_verbose_json() {
        let result = build_transcription_request(
            "whisper_small_q5".to_string(),
            AudioInput::WavPath(PathBuf::from("audio.wav")),
            None,
            None,
            Some("json"),
            vec!["segment".to_string()],
            Vec::new(),
            false,
            false,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn formats_srt_and_vtt() {
        let response = sample_response();
        assert!(format_srt(&response).contains("00:00:01,250 --> 00:00:02,500"));
        assert!(format_vtt(&response).starts_with("WEBVTT"));
    }
}
