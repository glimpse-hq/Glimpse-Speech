use std::{
    ffi::OsStr,
    future::Future,
    io,
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use axum::{
    Json, Router,
    body::Body,
    extract::{
        DefaultBodyLimit, FromRequest, Multipart, Path, Request, State, multipart::MultipartError,
    },
    http::{
        HeaderMap, StatusCode,
        header::{CONTENT_TYPE, HOST, ORIGIN},
    },
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use serde::Serialize;
use tokio::{io::AsyncWriteExt, sync::Semaphore, time::Instant};
use tower_http::cors::CorsLayer;

use crate::{
    TimestampGranularity, Transcription, UnsupportedValue,
    models::{InstallOptions, ModelDownloadProgress, ModelError},
    provider::{SpeechProvider, TranscribeError},
    service::{AudioInput, SpeechService, TranscribeRequest},
};

type ApiError = (StatusCode, Json<ErrorBody>);

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

pub(crate) const DEFAULT_MAX_UPLOAD_MIB: u64 = 768;
pub(crate) const DEFAULT_MAX_QUEUED_TRANSCRIPTIONS: usize = 4;
pub(crate) const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 15 * 60;
const DEFAULT_MAX_TEXT_FIELD_BYTES: usize = 64 * 1024;
// All text fields of a request together, since `dictionary` may repeat.
const TEXT_FIELDS_PER_REQUEST: usize = 4;

/// Resource limits for [`serve_with_limits`]. [`serve`] and
/// [`serve_with_shutdown`] use [`ApiLimits::default`].
#[derive(Debug, Clone)]
pub struct ApiLimits {
    /// Largest audio upload in bytes. The default, 768 MiB, fits an hour of
    /// 48 kHz stereo 16-bit WAV (about 660 MiB); compressed audio fits many hours.
    pub max_upload_bytes: u64,
    /// Largest text field (`prompt`, `dictionary`, `model`, ...) in bytes, 64
    /// KiB by default. A request's text fields together get four times this.
    pub max_text_field_bytes: usize,
    /// Transcription requests accepted at once, uploading, waiting for the
    /// engine or running. Further requests get a 429. Default 4; 0 counts as 1.
    pub max_queued_transcriptions: usize,
    /// Time a transcription request may take, upload included: 408 while
    /// uploading, 504 after. Default 15 minutes; `None` for no limit.
    pub request_timeout: Option<Duration>,
}

impl Default for ApiLimits {
    fn default() -> Self {
        Self {
            max_upload_bytes: DEFAULT_MAX_UPLOAD_MIB * 1024 * 1024,
            max_text_field_bytes: DEFAULT_MAX_TEXT_FIELD_BYTES,
            max_queued_transcriptions: DEFAULT_MAX_QUEUED_TRANSCRIPTIONS,
            request_timeout: Some(Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)),
        }
    }
}

impl ApiLimits {
    /// What all of a request's text fields may take together.
    fn text_budget(&self) -> usize {
        self.max_text_field_bytes
            .saturating_mul(TEXT_FIELDS_PER_REQUEST)
    }

    fn transcription_permits(&self) -> usize {
        self.max_queued_transcriptions
            .clamp(1, Semaphore::MAX_PERMITS)
    }
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
    loopback: bool,
    cors: bool,
    event_sink: Option<ApiEventSink>,
    local_models: Arc<Vec<ApiModelInfo>>,
    local_model_source: Option<ApiModelSource>,
    limits: ApiLimits,
    transcriptions: Arc<Semaphore>,
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

struct TempUpload(PathBuf);

impl Drop for TempUpload {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponseFormat {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

impl ResponseFormat {
    /// Formats that carry timestamps and therefore need segment output.
    pub(crate) fn needs_timestamps(self) -> bool {
        matches!(self, Self::VerboseJson | Self::Srt | Self::Vtt)
    }
}

impl FromStr for ResponseFormat {
    type Err = UnsupportedValue;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            "verbose_json" => Ok(Self::VerboseJson),
            "srt" => Ok(Self::Srt),
            "vtt" => Ok(Self::Vtt),
            other => Err(UnsupportedValue(other.to_string())),
        }
    }
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
    serve_with_limits(config, ApiLimits::default(), shutdown).await
}

pub async fn serve_with_limits(
    config: ApiConfig,
    limits: ApiLimits,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let addr = SocketAddr::new(parse_host(&config.host)?, config.port);
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
        loopback: addr.ip().is_loopback(),
        cors: cors_enabled,
        event_sink: config.event_sink,
        local_models: Arc::new(config.local_models),
        local_model_source: config.local_model_source,
        transcriptions: Arc::new(Semaphore::new(limits.transcription_permits())),
        limits,
    };
    state.log("info", format!("Local API listening on http://{addr}"));

    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

fn router(state: ApiState) -> Router {
    let model_management_enabled =
        !state.local_models.is_empty() || state.local_model_source.is_some();
    let cors_enabled = state.cors;
    // A backstop for the whole body; the upload and text fields are checked
    // against their own limits as they stream in.
    let body_limit = state
        .limits
        .max_upload_bytes
        .saturating_add(state.limits.text_budget() as u64)
        .saturating_add(1024 * 1024);

    let mut app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/transcriptions", post(transcribe))
        .layer(DefaultBodyLimit::max(
            usize::try_from(body_limit).unwrap_or(usize::MAX),
        ));

    if model_management_enabled {
        app = app
            .route("/v1/models/{id}/install", post(install_model))
            .route("/v1/models/{id}", delete(delete_model));
    }

    let mut app = app.with_state(state);

    if cors_enabled {
        app = app.layer(CorsLayer::permissive());
    }
    app
}

fn parse_host(host: &str) -> Result<IpAddr> {
    let host = host.trim().trim_start_matches('[').trim_end_matches(']');
    if host.eq_ignore_ascii_case("localhost") {
        return Ok(IpAddr::from([127, 0, 0, 1]));
    }
    host.parse()
        .with_context(|| format!("`{host}` is not an IP address or `localhost`"))
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn list_models(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
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
        .map_or_else(|| (*state.local_models).clone(), |source| source());
    Ok(Json(ListResponse::new(models)).into_response())
}

fn remote_model(id: String) -> ApiModelInfo {
    ApiModelInfo::new(
        id.clone(),
        id,
        "Remote transcription model configured for this server.".to_string(),
        vec!["Remote".to_string()],
        vec!["dictionary_prompt".to_string(), "timestamps".to_string()],
    )
}

async fn install_model(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<InstallResponse>, ApiError> {
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
) -> Result<Json<crate::models::ModelStatus>, ApiError> {
    authorize(&state, &headers)?;
    state.log("info", format!("DELETE /v1/models/{id}"));
    state.service.delete(&id).map(Json).map_err(map_error)
}

async fn transcribe(
    State(state): State<ApiState>,
    headers: HeaderMap,
    request: Request<Body>,
) -> Result<Response, ApiError> {
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

    // Taken before the upload, so it also bounds concurrent uploads.
    let Ok(permit) = Arc::clone(&state.transcriptions).try_acquire_owned() else {
        state.log(
            "warn",
            "Rejected a transcription: too many in progress".to_string(),
        );
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(typed_error_body(
                "Too many transcriptions in progress, try again shortly",
                "rate_limit_error",
            )),
        ));
    };
    let deadline = state
        .limits
        .request_timeout
        .and_then(|timeout| Instant::now().checked_add(timeout));
    let timeout = describe_duration(state.limits.request_timeout.unwrap_or_default());

    let (
        ParsedTranscriptionRequest {
            request,
            response_format,
            timestamp_granularities,
        },
        upload,
    ) = before(deadline, transcribe_request_from_multipart(request, &state))
        .await
        .ok_or_else(|| {
            (
                StatusCode::REQUEST_TIMEOUT,
                Json(error_body(format!(
                    "The upload did not finish within {timeout}"
                ))),
            )
        })??;

    // Once the engine starts it can't be interrupted, so the task keeps its
    // queue slot and the upload until it finishes, even after the request
    // times out. A request still queued for the engine at the deadline gives
    // up instead of starting.
    let provider = Arc::clone(&state.provider);
    let engine_deadline = deadline.map(Instant::into_std);
    let task = tokio::spawn(async move {
        let result = provider.transcribe_before(request, engine_deadline).await;
        drop((permit, upload));
        result
    });
    let result = before(deadline, task)
        .await
        .ok_or_else(|| {
            (
                StatusCode::GATEWAY_TIMEOUT,
                Json(typed_error_body(
                    format!("The transcription did not finish within {timeout}"),
                    "server_error",
                )),
            )
        })?
        .map_err(|err| map_server_error(anyhow!("transcription task failed: {err}")))?;

    result
        .map(|response| {
            state.log_model(
                "info",
                format!("Transcribed with {}", response.model_id),
                &response.model_id,
            );
            format_transcription_response(response, response_format, &timestamp_granularities)
        })
        .map_err(map_transcribe_error)
}

/// "15 minutes", "90 seconds", "1.5 seconds" or "100 ms".
fn describe_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if duration.subsec_nanos() == 0 && secs > 0 {
        match secs {
            1 => "1 second".to_string(),
            60 => "1 minute".to_string(),
            secs if secs % 60 == 0 => format!("{} minutes", secs / 60),
            secs => format!("{secs} seconds"),
        }
    } else if secs == 0 {
        format!("{} ms", duration.as_millis().max(1))
    } else {
        format!(
            "{} seconds",
            (duration.as_secs_f64() * 1000.0).round() / 1000.0
        )
    }
}

/// Runs `future` until `deadline`, or to completion without one.
async fn before<F: Future>(deadline: Option<Instant>, future: F) -> Option<F::Output> {
    match deadline {
        Some(deadline) => tokio::time::timeout_at(deadline, future).await.ok(),
        None => Some(future.await),
    }
}

impl ApiState {
    fn log(&self, level: &'static str, message: String) {
        self.emit(level, message, None);
    }

    fn log_model(&self, level: &'static str, message: String, model_id: &str) {
        self.emit(level, message, Some(model_id.to_string()));
    }

    fn emit(&self, level: &'static str, message: String, model_id: Option<String>) {
        if let Some(sink) = &self.event_sink {
            sink(ApiEvent {
                level,
                message,
                model_id,
            });
        }
    }
}

fn authorize(state: &ApiState, headers: &HeaderMap) -> Result<(), ApiError> {
    // Browsers still send simple POSTs cross-origin without CORS, and DNS
    // rebinding reaches loopback under a foreign Host.
    if !state.cors && headers.contains_key(ORIGIN) {
        return Err(forbidden("Browser requests need CORS enabled"));
    }
    if state.loopback && !is_loopback_host(headers) {
        return Err(forbidden("Host must be localhost or a loopback address"));
    }

    let Some(expected) = &state.api_key else {
        return Ok(());
    };

    let bearer = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, token)| token.trim());
    let api_key = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(str::trim);

    if [bearer, api_key]
        .into_iter()
        .flatten()
        .any(|candidate| constant_time_eq(candidate.as_bytes(), expected.as_bytes()))
    {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            Json(error_body("Missing or invalid API key")),
        ))
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .fold(0u8, |diff, (a, b)| diff | (a ^ b))
            == 0
}

fn is_loopback_host(headers: &HeaderMap) -> bool {
    let Some(host) = headers.get(HOST).and_then(|value| value.to_str().ok()) else {
        return false;
    };
    let host = match host.rsplit_once(':') {
        Some((name, port)) if port.bytes().all(|b| b.is_ascii_digit()) => name,
        _ => host,
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

fn forbidden(message: &str) -> ApiError {
    (StatusCode::FORBIDDEN, Json(error_body(message)))
}

fn is_multipart_content_type(value: &str) -> bool {
    value
        .to_ascii_lowercase()
        .starts_with("multipart/form-data")
}

fn build_transcription_request(
    parts: TranscriptionRequestParts,
) -> Result<ParsedTranscriptionRequest, ApiError> {
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

    let response_format = response_format
        .as_deref()
        .unwrap_or("json")
        .parse::<ResponseFormat>()
        .map_err(|value| map_error(anyhow!("Unsupported response_format {value}")))?;
    let timestamp_granularities = parse_timestamp_granularities(&timestamp_granularities)?;
    if !timestamp_granularities.is_empty() && response_format != ResponseFormat::VerboseJson {
        return Err(map_error(anyhow!(
            "`timestamp_granularities` requires response_format `verbose_json`"
        )));
    }
    let needs_timestamps =
        timestamps || !timestamp_granularities.is_empty() || response_format.needs_timestamps();

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

fn parse_timestamp_granularities(values: &[String]) -> Result<Vec<TimestampGranularity>, ApiError> {
    let mut parsed = Vec::new();
    for entry in values.iter().flat_map(|value| split_field_values(value)) {
        let granularity = entry
            .parse::<TimestampGranularity>()
            .map_err(|value| map_error(anyhow!("Unsupported timestamp granularity {value}")))?;
        if !parsed.contains(&granularity) {
            parsed.push(granularity);
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
        ResponseFormat::Srt => {
            text_response(format_srt(&response), "application/x-subrip; charset=utf-8")
        }
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
    let duration = match response.duration_ms {
        0 => segments.last().map_or(0.0, |segment| segment.end),
        duration_ms => duration_ms as f32 / 1000.0,
    };

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
    let mut segments: Vec<_> = response
        .segments
        .iter()
        .flatten()
        .filter(|segment| !segment.text.trim().is_empty())
        .cloned()
        .collect();
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
) -> Result<(ParsedTranscriptionRequest, TempUpload), ApiError> {
    let mut multipart = Multipart::from_request(request, state)
        .await
        .map_err(|err| (err.status(), Json(error_body(err.body_text()))))?;
    let mut model = None;
    let mut language = None;
    let mut prompt = None;
    let mut response_format = None;
    let mut dictionary = Vec::new();
    let mut timestamp_granularities = Vec::new();
    let mut timestamps = false;
    let mut stream = false;
    let mut uploaded_file: Option<TempUpload> = None;
    let mut text = TextBudget {
        field_limit: state.limits.max_text_field_bytes,
        total_limit: state.limits.text_budget(),
        remaining: state.limits.text_budget(),
    };

    while let Some(mut field) = multipart.next_field().await.map_err(map_multipart_error)? {
        let Some(name) = field.name().map(str::to_string) else {
            continue;
        };

        match name.as_str() {
            "file" | "audio" => {
                if uploaded_file.is_some() {
                    return Err(map_error(anyhow!("Send one audio file per request")));
                }
                let extension = field.file_name().and_then(|name| {
                    PathBuf::from(name)
                        .extension()
                        .map(std::ffi::OsStr::to_owned)
                });
                uploaded_file = Some(
                    write_temp_audio(
                        extension.as_deref(),
                        &mut field,
                        state.limits.max_upload_bytes,
                    )
                    .await?,
                );
            }
            "model" => model = Some(text.read(field).await?),
            "language" => {
                let value = text.read(field).await?;
                if !value.trim().is_empty() {
                    language = Some(value);
                }
            }
            "prompt" => {
                let value = text.read(field).await?;
                if !value.trim().is_empty() {
                    prompt = Some(value);
                }
            }
            "response_format" => response_format = Some(text.read(field).await?),
            "dictionary" | "dictionary[]" | "glimpse_dictionary" => {
                dictionary.extend(split_field_values(&text.read(field).await?));
            }
            "timestamp_granularities" | "timestamp_granularities[]" => {
                timestamp_granularities.extend(split_field_values(&text.read(field).await?));
            }
            "timestamps" => timestamps = parse_bool(&text.read(field).await?),
            "stream" => stream = parse_bool(&text.read(field).await?),
            _ => {}
        }
    }

    let upload =
        uploaded_file.ok_or_else(|| map_error(anyhow!("Missing multipart file field `file`")))?;
    let audio_path = upload.0.clone();
    let model = model.ok_or_else(|| map_error(anyhow!("Missing multipart field `model`")))?;

    let parsed = build_transcription_request(TranscriptionRequestParts {
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
    Ok((parsed, upload))
}

/// Text fields are read into memory, so each has a size limit, and so do all
/// of a request's text fields together.
struct TextBudget {
    field_limit: usize,
    total_limit: usize,
    remaining: usize,
}

impl TextBudget {
    async fn read(
        &mut self,
        mut field: axum::extract::multipart::Field<'_>,
    ) -> Result<String, ApiError> {
        let name = field.name().unwrap_or_default().to_string();
        let limit = self.field_limit.min(self.remaining);
        let mut bytes = Vec::new();
        while let Some(chunk) = field.chunk().await.map_err(map_multipart_error)? {
            if bytes.len().saturating_add(chunk.len()) > limit {
                return Err(payload_too_large(if limit < self.field_limit {
                    format!(
                        "Text fields are larger than {} bytes in total",
                        self.total_limit
                    )
                } else {
                    format!("`{name}` is larger than {} bytes", self.field_limit)
                }));
            }
            bytes.extend_from_slice(&chunk);
        }
        self.remaining -= bytes.len();
        String::from_utf8(bytes).map_err(|_| map_error(anyhow!("`{name}` is not valid UTF-8")))
    }
}

fn payload_too_large(message: String) -> ApiError {
    (StatusCode::PAYLOAD_TOO_LARGE, Json(error_body(message)))
}

fn map_multipart_error(error: MultipartError) -> ApiError {
    (error.status(), Json(error_body(error.body_text())))
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

fn map_error(error: anyhow::Error) -> ApiError {
    (StatusCode::BAD_REQUEST, Json(error_body(error.to_string())))
}

fn map_server_error(error: anyhow::Error) -> ApiError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(typed_error_body(error.to_string(), "server_error")),
    )
}

fn map_transcribe_error(error: TranscribeError) -> ApiError {
    match error {
        TranscribeError::Local(err) => map_local_transcribe_error(err),
        #[cfg(feature = "remote")]
        TranscribeError::Remote(err) => map_remote_error(err),
    }
}

fn map_local_transcribe_error(error: anyhow::Error) -> ApiError {
    if error.chain().any(|cause| cause.is::<ModelError>()) {
        map_error(error)
    } else {
        map_server_error(error)
    }
}

#[cfg(feature = "remote")]
fn map_remote_error(error: crate::remote::RemoteError) -> ApiError {
    use crate::remote::RemoteErrorKind;

    let (status, error_type) = match error.kind {
        RemoteErrorKind::RateLimited => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
        RemoteErrorKind::QuotaExceeded => (StatusCode::PAYMENT_REQUIRED, "insufficient_quota"),
        RemoteErrorKind::Unauthorized => (StatusCode::UNAUTHORIZED, "authentication_error"),
        RemoteErrorKind::InvalidRequest => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        RemoteErrorKind::NotFound => (StatusCode::NOT_FOUND, "not_found_error"),
        RemoteErrorKind::UpstreamUnavailable | RemoteErrorKind::Other => {
            (StatusCode::BAD_GATEWAY, "upstream_error")
        }
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
    max_bytes: u64,
) -> Result<TempUpload, ApiError> {
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

        let upload = TempUpload(path);
        let mut writer = tokio::io::BufWriter::new(file);
        let mut written = 0u64;
        while let Some(chunk) = field.chunk().await.map_err(map_multipart_error)? {
            written += chunk.len() as u64;
            if written > max_bytes {
                return Err(payload_too_large(format!(
                    "The audio file is larger than {} MiB",
                    max_bytes.div_ceil(1024 * 1024)
                )));
            }
            writer
                .write_all(&chunk)
                .await
                .with_context(|| format!("write uploaded audio to {}", upload.0.display()))
                .map_err(map_server_error)?;
        }
        writer
            .flush()
            .await
            .with_context(|| format!("write uploaded audio to {}", upload.0.display()))
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
    path.set_extension(extension.unwrap_or(OsStr::new("wav")));
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
            parse_timestamp_granularities(&["segment".to_string(), "word,segment".to_string()])
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
    fn parses_hosts_and_localhost() {
        assert_eq!(
            parse_host("localhost").unwrap(),
            IpAddr::from([127, 0, 0, 1])
        );
        assert!(parse_host("::1").unwrap().is_loopback());
        assert!(parse_host("[::1]").unwrap().is_loopback());
        assert!(!parse_host("0.0.0.0").unwrap().is_loopback());
        assert!(parse_host("example.com").is_err());
    }

    #[test]
    fn accepts_either_key_header_with_any_bearer_case() {
        let state = ApiState {
            service: Arc::new(SpeechService::new_loose_with_engine(
                std::env::temp_dir(),
                crate::models::ModelEngine::Whisper,
            )),
            provider: Arc::new(SpeechProvider::Local(Arc::new(
                SpeechService::new_loose_with_engine(
                    std::env::temp_dir(),
                    crate::models::ModelEngine::Whisper,
                ),
            ))),
            api_key: Some(Arc::from("secret")),
            loopback: false,
            cors: false,
            event_sink: None,
            local_models: Arc::new(Vec::new()),
            local_model_source: None,
            limits: ApiLimits::default(),
            transcriptions: Arc::new(Semaphore::new(1)),
        };
        let headers = |name: &'static str, value: &'static str| {
            let mut headers = HeaderMap::new();
            headers.insert(name, value.parse().unwrap());
            headers
        };
        assert!(authorize(&state, &headers("authorization", "Bearer secret")).is_ok());
        assert!(authorize(&state, &headers("authorization", "bearer secret")).is_ok());
        assert!(authorize(&state, &headers("x-api-key", "secret")).is_ok());
        assert!(authorize(&state, &headers("authorization", "Bearer secre")).is_err());
        assert!(authorize(&state, &headers("authorization", "Basic secret")).is_err());
        assert!(authorize(&state, &HeaderMap::new()).is_err());
    }

    mod limits {
        use std::{
            pin::Pin,
            task::{Context, Poll},
        };

        use axum::body::{Bytes, HttpBody};
        use tower::ServiceExt;

        use super::*;

        const BOUNDARY: &str = "glimpse-test-boundary";

        pub(super) fn state(limits: ApiLimits) -> ApiState {
            // A catalog that knows no models: every id is unknown.
            let service = Arc::new(SpeechService::new(crate::service::SpeechConfig {
                model_cache_dir: std::env::temp_dir().join("glimpse-speech-api-limits"),
                resolver: Arc::new(|_| None),
                loose_paths: false,
            }));
            ApiState {
                provider: Arc::new(SpeechProvider::Local(Arc::clone(&service))),
                service,
                api_key: None,
                loopback: false,
                cors: false,
                event_sink: None,
                local_models: Arc::new(Vec::new()),
                local_model_source: None,
                transcriptions: Arc::new(Semaphore::new(limits.transcription_permits())),
                limits,
            }
        }

        fn multipart(fields: &[(&str, &[u8])]) -> Vec<u8> {
            let mut body = Vec::new();
            for (name, value) in fields {
                let filename = if *name == "file" {
                    "; filename=\"audio.wav\""
                } else {
                    ""
                };
                body.extend_from_slice(
                    format!(
                        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"{filename}\r\n\r\n"
                    )
                    .as_bytes(),
                );
                body.extend_from_slice(value);
                body.extend_from_slice(b"\r\n");
            }
            body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
            body
        }

        fn request(body: Body) -> Request<Body> {
            Request::post("/v1/audio/transcriptions")
                .header(
                    CONTENT_TYPE,
                    format!("multipart/form-data; boundary={BOUNDARY}"),
                )
                .body(body)
                .unwrap()
        }

        async fn send(state: &ApiState, body: Body) -> (StatusCode, serde_json::Value) {
            let response = router(state.clone()).oneshot(request(body)).await.unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            (status, serde_json::from_slice(&bytes).unwrap_or_default())
        }

        /// A body whose first bytes arrive and the rest never does.
        struct Stalled(Option<Bytes>);

        impl HttpBody for Stalled {
            type Data = Bytes;
            type Error = std::convert::Infallible;

            fn poll_frame(
                mut self: Pin<&mut Self>,
                _: &mut Context<'_>,
            ) -> Poll<Option<Result<http_body::Frame<Bytes>, Self::Error>>> {
                match self.0.take() {
                    Some(bytes) => Poll::Ready(Some(Ok(http_body::Frame::data(bytes)))),
                    None => Poll::Pending,
                }
            }
        }

        fn stalled_upload() -> Body {
            Body::new(Stalled(Some(Bytes::from(format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nRIFF"
            )))))
        }

        #[tokio::test]
        async fn unknown_model_is_a_400() {
            let state = state(ApiLimits::default());
            let body = multipart(&[("model", b"/etc/hosts"), ("file", b"RIFF")]);
            let (status, json) = send(&state, Body::from(body)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(json["error"]["message"], "Unknown model: /etc/hosts");
        }

        #[tokio::test]
        async fn a_model_that_is_not_installed_is_a_400() {
            let mut state = state(ApiLimits::default());
            let service = Arc::new(SpeechService::new(crate::service::SpeechConfig {
                model_cache_dir: std::env::temp_dir().join("glimpse-speech-api-not-installed"),
                resolver: Arc::new(|id| {
                    (id == "known").then(|| crate::models::InstallSpec {
                        id: id.to_string(),
                        engine: crate::models::ModelEngine::Whisper,
                        storage: crate::models::ModelStorage::Directory,
                        files: vec![crate::models::RemoteFile {
                            url: "https://example.test/ggml-known.bin".to_string(),
                            path: "ggml-known.bin".to_string(),
                            size_bytes: None,
                            sha256: None,
                            extract: false,
                        }],
                        variant: None,
                    })
                }),
                loose_paths: false,
            }));
            state.provider = Arc::new(SpeechProvider::Local(Arc::clone(&service)));
            state.service = service;
            let body = multipart(&[("model", b"known"), ("file", b"RIFF")]);
            let (status, json) = send(&state, Body::from(body)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
            assert_eq!(json["error"]["type"], "invalid_request_error");
            assert_eq!(
                json["error"]["message"],
                "known is not fully installed. Missing: ggml-known.bin"
            );
        }

        #[tokio::test]
        async fn oversized_text_fields_are_rejected() {
            let state = state(ApiLimits {
                max_text_field_bytes: 16,
                ..ApiLimits::default()
            });
            let body = multipart(&[("prompt", &[b'a'; 17]), ("file", b"RIFF")]);
            let (status, json) = send(&state, Body::from(body)).await;
            assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{json}");

            // Each field fits, but repeated fields add up past four times the limit.
            let entry = [b'a'; 16];
            let fields = vec![("dictionary", &entry[..]); TEXT_FIELDS_PER_REQUEST + 1];
            let (status, json) = send(&state, Body::from(multipart(&fields))).await;
            assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{json}");
        }

        #[tokio::test]
        async fn oversized_uploads_are_rejected() {
            let state = state(ApiLimits {
                max_upload_bytes: 16,
                ..ApiLimits::default()
            });
            let body = multipart(&[("model", b"m"), ("file", &[0u8; 17])]);
            let (status, json) = send(&state, Body::from(body)).await;
            assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{json}");
        }

        #[tokio::test]
        async fn a_second_audio_file_is_rejected() {
            let state = state(ApiLimits::default());
            let body = multipart(&[("file", b"RIFF"), ("audio", b"RIFF")]);
            let (status, _) = send(&state, Body::from(body)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
        }

        #[tokio::test]
        async fn a_full_queue_answers_429() {
            let state = state(ApiLimits {
                max_queued_transcriptions: 1,
                ..ApiLimits::default()
            });
            let stalled = tokio::spawn({
                let state = state.clone();
                async move { send(&state, stalled_upload()).await }
            });
            while state.transcriptions.available_permits() > 0 {
                tokio::task::yield_now().await;
            }

            let body = multipart(&[("model", b"m"), ("file", b"RIFF")]);
            let (status, json) = send(&state, Body::from(body)).await;
            assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(json["error"]["type"], "rate_limit_error");

            stalled.abort();
            let _ = stalled.await;
            assert_eq!(state.transcriptions.available_permits(), 1);
        }

        #[tokio::test]
        async fn a_timed_out_transcription_frees_its_permit_and_upload_once_it_ends() {
            let cache = std::env::temp_dir()
                .join(format!("glimpse-speech-api-timeout-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&cache);
            std::fs::create_dir_all(&cache).unwrap();
            std::fs::write(cache.join("ggml-held.bin"), b"never loaded").unwrap();
            let service = Arc::new(
                SpeechService::new_loose_with_engine(
                    cache.clone(),
                    crate::models::ModelEngine::Whisper,
                )
                .only_cached_models(None),
            );
            let mut state = state(ApiLimits {
                max_queued_transcriptions: 2,
                request_timeout: Some(Duration::from_millis(100)),
                ..ApiLimits::default()
            });
            state.provider = Arc::new(SpeechProvider::Local(Arc::clone(&service)));
            state.service = Arc::clone(&service);
            // A unique extension finds this test's upload among the temp files.
            let extension = format!("held{}", std::process::id());
            let uploads = || {
                std::fs::read_dir(std::env::temp_dir())
                    .unwrap()
                    .flatten()
                    .filter(|entry| entry.path().extension() == Some(OsStr::new(&extension)))
                    .count()
            };
            let mut body = format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nggml-held.bin\r\n\
                 --{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.{extension}\"\r\n\r\n"
            )
            .into_bytes();
            body.extend_from_slice(b"RIFF\r\n");
            body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());

            // The engine is busy, so the transcription queues behind it.
            let engine = service.hold_engine();
            let (status, json) = send(&state, Body::from(body)).await;
            assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{json}");
            assert_eq!(json["error"]["type"], "server_error");
            assert_eq!(
                json["error"]["message"],
                "The transcription did not finish within 100 ms"
            );
            assert_eq!(state.transcriptions.available_permits(), 1);
            assert_eq!(uploads(), 1);

            // Past its deadline, the queued transcription gives up as soon as
            // it gets the engine, and releases both.
            drop(engine);
            tokio::time::timeout(Duration::from_secs(10), async {
                while state.transcriptions.available_permits() < 2 {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .expect("the permit comes back");
            assert_eq!(uploads(), 0);
            assert!(!service.is_loaded());
            let _ = std::fs::remove_dir_all(&cache);
        }

        #[tokio::test]
        async fn delete_removes_only_cached_model_directories() {
            let cache = std::env::temp_dir()
                .join(format!("glimpse-speech-api-delete-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&cache);
            std::fs::create_dir_all(cache.join("whisper_base")).unwrap();
            std::fs::write(cache.join("whisper_base/ggml-base.bin"), b"ggml").unwrap();
            std::fs::write(cache.join("ggml-tiny.bin"), b"ggml").unwrap();
            let service = Arc::new(
                SpeechService::new_loose_with_engine(
                    cache.clone(),
                    crate::models::ModelEngine::Whisper,
                )
                .only_cached_models(None),
            );
            let mut state = state(ApiLimits::default());
            state.provider = Arc::new(SpeechProvider::Local(Arc::clone(&service)));
            state.service = service;
            state.local_model_source = Some(Arc::new(Vec::new));
            let delete = |id: &str| {
                let request = Request::delete(format!("/v1/models/{id}"))
                    .body(Body::empty())
                    .unwrap();
                let app = router(state.clone());
                async move {
                    let response = app.oneshot(request).await.unwrap();
                    let status = response.status();
                    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                        .await
                        .unwrap();
                    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                    (status, json)
                }
            };

            for id in ["ggml-tiny.bin", "missing", "%2E%2E"] {
                let (status, json) = delete(id).await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{id}: {json}");
            }
            assert!(cache.join("ggml-tiny.bin").is_file());

            let (status, json) = delete("whisper_base").await;
            assert_eq!(status, StatusCode::OK, "{json}");
            assert_eq!(json["id"], "whisper_base");
            assert_eq!(json["installed"], false);
            assert!(!cache.join("whisper_base").exists());
            assert!(cache.is_dir());
            let _ = std::fs::remove_dir_all(&cache);
        }

        #[tokio::test]
        async fn a_stalled_upload_times_out() {
            let state = state(ApiLimits {
                request_timeout: Some(Duration::from_millis(50)),
                ..ApiLimits::default()
            });
            let (status, _) = send(&state, stalled_upload()).await;
            assert_eq!(status, StatusCode::REQUEST_TIMEOUT);
            assert_eq!(state.transcriptions.available_permits(), 4);
        }
    }

    #[test]
    fn describes_timeouts() {
        let cases = [
            (Duration::from_secs(900), "15 minutes"),
            (Duration::from_secs(60), "1 minute"),
            (Duration::from_secs(90), "90 seconds"),
            (Duration::from_secs(1), "1 second"),
            (Duration::from_millis(1500), "1.5 seconds"),
            (Duration::from_millis(50), "50 ms"),
            (Duration::from_micros(10), "1 ms"),
        ];
        for (duration, text) in cases {
            assert_eq!(describe_duration(duration), text);
        }
    }

    #[test]
    fn huge_limits_saturate() {
        let huge = ApiLimits {
            max_upload_bytes: u64::MAX,
            max_text_field_bytes: usize::MAX,
            max_queued_transcriptions: usize::MAX,
            request_timeout: None,
        };
        assert_eq!(huge.text_budget(), usize::MAX);
        assert_eq!(huge.transcription_permits(), Semaphore::MAX_PERMITS);
        assert_eq!(
            ApiLimits {
                max_queued_transcriptions: 0,
                ..ApiLimits::default()
            }
            .transcription_permits(),
            1
        );
        // Neither the body limit nor the semaphore overflows.
        let _ = router(limits::state(huge));
    }

    #[test]
    fn only_model_errors_are_client_errors() {
        let wrapped = anyhow::Error::from(ModelError::InvalidModelId("a/b".to_string()))
            .context("load the model");
        assert_eq!(
            map_local_transcribe_error(wrapped).0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            map_local_transcribe_error(anyhow!("Unknown model: lookalike")).0,
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn verbose_duration_uses_audio_length() {
        let json = verbose_response(sample_response(), &[]);
        assert_eq!(json.duration, 1.25);
        let mut remote = sample_response();
        remote.duration_ms = 0;
        assert_eq!(verbose_response(remote, &[]).duration, 2.5);
    }

    #[test]
    fn captions_skip_empty_segments() {
        let mut response = sample_response();
        response
            .segments
            .as_mut()
            .unwrap()
            .push(crate::TranscriptionSegment {
                start: 2.5,
                end: 3.0,
                text: "  ".to_string(),
            });
        assert_eq!(caption_segments(&response).len(), 1);
    }

    #[test]
    fn parses_response_formats() {
        assert_eq!("srt".parse(), Ok(ResponseFormat::Srt));
        assert_eq!("verbose_json".parse(), Ok(ResponseFormat::VerboseJson));
        assert!("xml".parse::<ResponseFormat>().is_err());
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
