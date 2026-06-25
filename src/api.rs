use std::{
    ffi::OsStr,
    future::Future,
    io,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
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
use tokio::io::AsyncWriteExt;
use tower_http::cors::CorsLayer;

use crate::{
    models::{InstallOptions, ModelDownloadProgress},
    provider::{SpeechProvider, TranscribeError},
    service::{AudioInput, SpeechService, TranscribeRequest},
    TimestampGranularity, Transcription,
};

#[derive(Clone)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub service: Arc<SpeechService>,
    pub api_key: Option<String>,
    pub event_sink: Option<ApiEventSink>,
    /// When true, responses include permissive CORS headers so browser-based
    /// clients on any origin can call the API.
    pub cors: bool,
    pub transcription_provider: Option<Arc<SpeechProvider>>,
    pub local_models: Vec<ApiModelInfo>,
    pub local_model_source: Option<ApiModelSource>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiModelInfo {
    pub id: String,
    object: &'static str,
    pub label: String,
    pub description: String,
    pub tags: Vec<String>,
    pub capabilities: Vec<String>,
}

impl ApiModelInfo {
    pub fn new(
        id: String,
        label: String,
        description: String,
        tags: Vec<String>,
        capabilities: Vec<String>,
    ) -> Self {
        Self {
            id,
            object: "model",
            label,
            description,
            tags,
            capabilities,
        }
    }
}

#[derive(Clone)]
struct ApiState {
    service: Arc<SpeechService>,
    provider: Arc<SpeechProvider>,
    api_key: Option<Arc<str>>,
    event_sink: Option<ApiEventSink>,
    local_models: Arc<Vec<ApiModelInfo>>,
    local_model_source: Option<ApiModelSource>,
}

pub type ApiEventSink = Arc<dyn Fn(ApiEvent) + Send + Sync + 'static>;
pub type ApiModelSource = Arc<dyn Fn() -> Vec<ApiModelInfo> + Send + Sync + 'static>;

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

#[derive(Debug, Serialize)]
struct RemoteModel {
    id: String,
    object: &'static str,
    label: String,
    description: String,
    tags: &'static [&'static str],
    capabilities: &'static [&'static str],
}

#[derive(Debug, Serialize)]
struct ListResponse<T> {
    object: &'static str,
    data: Vec<T>,
}

impl<T> ListResponse<T> {
    fn new(data: Vec<T>) -> Self {
        Self {
            object: "list",
            data,
        }
    }
}

struct ParsedTranscriptionRequest {
    request: TranscribeRequest,
    response_format: ResponseFormat,
    timestamp_granularities: Vec<TimestampGranularity>,

    uploaded_file: Option<TempUpload>,
}

struct TranscriptionRequestParts {
    model: String,
    audio: AudioInput,
    language: Option<String>,
    prompt: Option<String>,
    response_format: Option<String>,
    timestamp_granularities: Vec<String>,
    dictionary: Vec<String>,
    timestamps: bool,
    stream: bool,
}

static TEMP_UPLOAD_COUNTER: AtomicU64 = AtomicU64::new(0);

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

#[derive(Debug, Serialize)]
struct JsonTranscriptionResponse {
    text: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct VerboseTranscriptionResponse {
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
    let api_key = config.api_key.filter(|key| !key.trim().is_empty());
    if !addr.ip().is_loopback() && api_key.is_none() {
        return Err(anyhow!("an API key is required when listening on LAN"));
    }

    let cors_enabled = config.cors;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let service = config.service;
    let provider = config
        .transcription_provider
        .unwrap_or_else(|| Arc::new(SpeechProvider::Local(Arc::clone(&service))));
    let state = ApiState {
        service,
        provider,
        api_key: api_key.map(Arc::from),
        event_sink: config.event_sink,
        local_models: Arc::new(config.local_models),
        local_model_source: config.local_model_source,
    };
    state.log("info", format!("Local API listening on http://{addr}"));
    let model_management_enabled =
        !state.local_models.is_empty() || state.local_model_source.is_some();

    let mut app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/audio/transcriptions", post(transcribe))
        .layer(DefaultBodyLimit::max(1024 * 1024 * 1024));

    if model_management_enabled {
        app = app
            .route("/v1/models/{id}/install", post(install_model))
            .route("/v1/models/{id}", delete(delete_model));
    }

    let mut app = app.with_state(state);

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
) -> Result<Response, (StatusCode, Json<ErrorBody>)> {
    authorize(&state, &headers)?;
    state.log("info", "GET /v1/models".to_string());
    if let Some(remote_ids) = state.provider.remote_model_ids().await {
        let remote_ids = remote_ids.map_err(map_transcribe_error)?;
        let data = remote_ids.into_iter().map(remote_model).collect::<Vec<_>>();
        return Ok(Json(ListResponse::new(data)).into_response());
    }

    let models = state
        .local_model_source
        .as_ref()
        .map(|source| source())
        .unwrap_or_else(|| (*state.local_models).clone());
    Ok(Json(ListResponse::new(models)).into_response())
}

fn remote_model(id: String) -> RemoteModel {
    let label = id.clone();
    RemoteModel {
        id,
        object: "model",
        label,
        description: "Remote transcription model configured for this server.".to_string(),
        tags: &["Remote"],
        capabilities: &["dictionary_prompt", "timestamps"],
    }
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
        .install(
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
    state.service.delete(&id).map(Json).map_err(map_error)
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

    if !is_multipart_content_type(content_type) {
        return Err(map_error(anyhow!(
            "Use multipart/form-data with a `file` field for transcription"
        )));
    }

    let ParsedTranscriptionRequest {
        request,
        response_format,
        timestamp_granularities,
        uploaded_file,
    } = transcribe_request_from_multipart(request, &state).await?;

    let result = state
        .provider
        .transcribe(request)
        .await
        .map(|response| {
            state.log("info", format!("Transcribed with {}", response.model_id));
            state.log_model(
                "info",
                format!("Loaded model {}", response.model_id),
                &response.model_id,
            );
            format_transcription_response(response, response_format, &timestamp_granularities)
        })
        .map_err(map_transcribe_error);

    drop(uploaded_file);
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

fn is_multipart_content_type(value: &str) -> bool {
    value
        .to_ascii_lowercase()
        .starts_with("multipart/form-data")
}

fn build_transcription_request(
    parts: TranscriptionRequestParts,
) -> Result<ParsedTranscriptionRequest, (StatusCode, Json<ErrorBody>)> {
    let TranscriptionRequestParts {
        model,
        audio,
        language,
        prompt,
        response_format,
        timestamp_granularities,
        dictionary,
        timestamps,
        stream,
    } = parts;

    if stream {
        return Err(map_error(anyhow!(
            "Streaming transcription responses are not supported"
        )));
    }

    let response_format = parse_response_format(response_format.as_deref().unwrap_or("json"))?;
    let timestamp_granularities = parse_timestamp_granularities(timestamp_granularities)?;
    if !timestamp_granularities.is_empty() && response_format != ResponseFormat::VerboseJson {
        return Err(map_error(anyhow!(
            "`timestamp_granularities` requires response_format `verbose_json`"
        )));
    }
    let needs_timestamps = timestamps
        || !timestamp_granularities.is_empty()
        || matches!(
            response_format,
            ResponseFormat::VerboseJson | ResponseFormat::Srt | ResponseFormat::Vtt
        );

    Ok(ParsedTranscriptionRequest {
        request: TranscribeRequest {
            audio,
            model_id: model,
            language,
            prompt,
            dictionary,
            timestamps: needs_timestamps,
            timestamp_granularity: service_timestamp_granularity(&timestamp_granularities)
                .or_else(|| needs_timestamps.then_some(TimestampGranularity::Segment)),
        },
        response_format,
        timestamp_granularities,
        uploaded_file: None,
    })
}

fn service_timestamp_granularity(values: &[TimestampGranularity]) -> Option<TimestampGranularity> {
    if values.contains(&TimestampGranularity::Word) {
        Some(TimestampGranularity::Word)
    } else if values.contains(&TimestampGranularity::Segment) {
        Some(TimestampGranularity::Segment)
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
    response: Transcription,
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

pub(crate) fn verbose_response(
    response: Transcription,
    timestamp_granularities: &[TimestampGranularity],
) -> VerboseTranscriptionResponse {
    let segments = verbose_segments(&response);
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

fn verbose_words(response: &Transcription) -> Vec<VerboseWord> {
    if let Some(words) = &response.words {
        return words
            .iter()
            .map(|word| VerboseWord {
                word: word.text.clone(),
                start: word.start,
                end: word.end,
            })
            .collect();
    }

    caption_segments(response)
        .into_iter()
        .flat_map(|segment| words_for_segment(&segment))
        .collect()
}

fn words_for_segment(segment: &crate::TranscriptionSegment) -> Vec<VerboseWord> {
    let words = segment.text.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        return Vec::new();
    }

    let word_count = words.len();
    let duration = (segment.end - segment.start).max(0.0);
    let step = duration / word_count as f32;
    words
        .into_iter()
        .enumerate()
        .map(|(idx, word)| {
            let start = segment.start + step * idx as f32;
            let end = if idx + 1 == word_count {
                segment.end
            } else {
                segment.start + step * (idx + 1) as f32
            };
            VerboseWord {
                word: word.to_string(),
                start,
                end,
            }
        })
        .collect()
}

fn verbose_segments(response: &Transcription) -> Vec<VerboseSegment> {
    caption_segments(response)
        .into_iter()
        .enumerate()
        .map(|(id, segment)| VerboseSegment {
            id,
            seek: 0,
            start: segment.start,
            end: segment.end,
            text: segment.text,
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

pub(crate) fn format_srt(response: &Transcription) -> String {
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

pub(crate) fn format_vtt(response: &Transcription) -> String {
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

pub(crate) fn caption_segments(response: &Transcription) -> Vec<crate::TranscriptionSegment> {
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

    while let Some(mut field) = multipart
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
                uploaded_file = Some(write_temp_audio(extension.as_deref(), &mut field).await?);
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

    let mut parsed = build_transcription_request(TranscriptionRequestParts {
        model,
        audio: AudioInput::WavPath(audio_path),
        language,
        prompt,
        response_format,
        timestamp_granularities,
        dictionary,
        timestamps,
        stream,
    })?;
    parsed.uploaded_file = Some(upload);
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
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "1" | "yes" | "on"
    )
}

fn map_error(error: anyhow::Error) -> (StatusCode, Json<ErrorBody>) {
    (StatusCode::BAD_REQUEST, Json(error_body(error.to_string())))
}

fn map_server_error(error: anyhow::Error) -> (StatusCode, Json<ErrorBody>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(typed_error_body(error.to_string(), "server_error")),
    )
}

fn map_transcribe_error(error: TranscribeError) -> (StatusCode, Json<ErrorBody>) {
    match error {
        TranscribeError::Local(err) => map_local_transcribe_error(err),
        #[cfg(feature = "remote")]
        TranscribeError::Remote(err) => map_remote_error(err),
    }
}

fn map_local_transcribe_error(error: anyhow::Error) -> (StatusCode, Json<ErrorBody>) {
    if error.to_string().starts_with("Unknown model:") {
        map_error(error)
    } else {
        map_server_error(error)
    }
}

#[cfg(feature = "remote")]
fn map_remote_error(error: crate::remote::RemoteError) -> (StatusCode, Json<ErrorBody>) {
    let status = match error.kind {
        crate::remote::RemoteErrorKind::RateLimited => StatusCode::TOO_MANY_REQUESTS,
        crate::remote::RemoteErrorKind::QuotaExceeded => StatusCode::PAYMENT_REQUIRED,
        crate::remote::RemoteErrorKind::Unauthorized => StatusCode::UNAUTHORIZED,
        crate::remote::RemoteErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
        crate::remote::RemoteErrorKind::NotFound => StatusCode::NOT_FOUND,
        crate::remote::RemoteErrorKind::UpstreamUnavailable
        | crate::remote::RemoteErrorKind::Other => StatusCode::BAD_GATEWAY,
    };
    let error_type = match error.kind {
        crate::remote::RemoteErrorKind::RateLimited => "rate_limit_error",
        crate::remote::RemoteErrorKind::QuotaExceeded => "insufficient_quota",
        crate::remote::RemoteErrorKind::Unauthorized => "authentication_error",
        crate::remote::RemoteErrorKind::InvalidRequest => "invalid_request_error",
        crate::remote::RemoteErrorKind::NotFound => "not_found_error",
        crate::remote::RemoteErrorKind::UpstreamUnavailable
        | crate::remote::RemoteErrorKind::Other => "upstream_error",
    };
    (
        status,
        Json(ErrorBody {
            error: ErrorDetail {
                message: error.user_message(),
                error_type,
                param: error.param,
                code: error.code,
            },
        }),
    )
}

fn error_body(message: impl Into<String>) -> ErrorBody {
    typed_error_body(message, "invalid_request_error")
}

fn typed_error_body(message: impl Into<String>, error_type: &'static str) -> ErrorBody {
    ErrorBody {
        error: ErrorDetail {
            message: message.into(),
            error_type,
            param: None,
            code: None,
        },
    }
}

/// Streams the multipart field to a unique temp file without buffering the
/// whole upload in memory.
async fn write_temp_audio(
    extension: Option<&OsStr>,
    field: &mut axum::extract::multipart::Field<'_>,
) -> Result<TempUpload, (StatusCode, Json<ErrorBody>)> {
    for _ in 0..16 {
        let path = temp_audio_path(extension);
        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .await;

        let file = match file {
            Ok(file) => file,
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(map_server_error(
                    anyhow!(err).context(format!("create temp upload {}", path.display())),
                ));
            }
        };

        let upload = TempUpload::new(path);
        let mut writer = tokio::io::BufWriter::new(file);
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|err| map_error(anyhow!(err.to_string())))?
        {
            writer
                .write_all(&chunk)
                .await
                .with_context(|| format!("write uploaded audio to {}", upload.path().display()))
                .map_err(map_server_error)?;
        }
        writer
            .flush()
            .await
            .with_context(|| format!("write uploaded audio to {}", upload.path().display()))
            .map_err(map_server_error)?;
        return Ok(upload);
    }

    Err(map_server_error(anyhow!(
        "failed to create a unique temp upload path"
    )))
}

fn temp_audio_path(extension: Option<&OsStr>) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let sequence = TEMP_UPLOAD_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir().join(format!(
        "glimpse-speech-upload-{}-{timestamp}-{sequence}",
        std::process::id(),
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

    fn sample_response() -> Transcription {
        Transcription {
            text: "hello world".to_string(),
            segments: Some(vec![crate::TranscriptionSegment {
                start: 1.25,
                end: 2.5,
                text: "hello world".to_string(),
            }]),
            words: None,
            model_id: "whisper_small_q5".to_string(),
            language: Some("en".to_string()),
            duration_ms: 1250,
        }
    }

    #[test]
    fn parse_bool_accepts_case_and_common_truthy_values() {
        for truthy in ["true", "True", "TRUE", "1", "Yes", "ON", " true "] {
            assert!(parse_bool(truthy), "{truthy:?} should be truthy");
        }
        for falsy in ["false", "False", "0", "no", "", "off"] {
            assert!(!parse_bool(falsy), "{falsy:?} should be falsy");
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
        let result = build_transcription_request(TranscriptionRequestParts {
            model: "whisper_small_q5".to_string(),
            audio: AudioInput::WavPath(PathBuf::from("audio.wav")),
            language: None,
            prompt: None,
            response_format: Some("json".to_string()),
            timestamp_granularities: vec!["segment".to_string()],
            dictionary: Vec::new(),
            timestamps: false,
            stream: false,
        });
        assert!(result.is_err());
    }

    #[test]
    fn verbose_json_requests_segment_timestamps_by_default() {
        let parsed = build_transcription_request(TranscriptionRequestParts {
            model: "whisper_small_q5".to_string(),
            audio: AudioInput::WavPath(PathBuf::from("audio.wav")),
            language: None,
            prompt: None,
            response_format: Some("verbose_json".to_string()),
            timestamp_granularities: Vec::new(),
            dictionary: Vec::new(),
            timestamps: false,
            stream: false,
        })
        .unwrap();

        assert!(parsed.request.timestamps);
        assert_eq!(
            parsed.request.timestamp_granularity,
            Some(TimestampGranularity::Segment)
        );
    }

    #[test]
    fn formats_srt_and_vtt() {
        let response = sample_response();
        assert!(format_srt(&response).contains("00:00:01,250 --> 00:00:02,500"));
        assert!(format_vtt(&response).starts_with("WEBVTT"));
    }

    #[test]
    fn verbose_words_split_segment_text() {
        let words = verbose_words(&sample_response());
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].start, 1.25);
        assert!((words[0].end - 1.875).abs() < f32::EPSILON);
        assert_eq!(words[1].word, "world");
        assert!((words[1].start - 1.875).abs() < f32::EPSILON);
        assert_eq!(words[1].end, 2.5);
    }

    #[test]
    fn verbose_words_use_upstream_word_timestamps() {
        let mut response = sample_response();
        response.words = Some(vec![
            crate::TranscriptionSegment {
                start: 1.25,
                end: 1.5,
                text: "hello".to_string(),
            },
            crate::TranscriptionSegment {
                start: 2.0,
                end: 2.5,
                text: "world".to_string(),
            },
        ]);

        let words = verbose_words(&response);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].end, 1.5);
        assert_eq!(words[1].start, 2.0);
    }
}
