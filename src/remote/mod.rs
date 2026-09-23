mod engine;
mod provider;

use std::{
    error::Error as StdError,
    fmt,
    time::{Duration, SystemTime},
};

pub use engine::{
    DiarizedSegment, DiarizedTranscription, RemoteConfig, RemoteEngine, RemoteRequestParams,
};

/// Reports whether an endpoint and model return speaker-diarized transcriptions.
pub fn supports_diarization(endpoint: &str, model: &str) -> bool {
    provider::resolve_profile(endpoint).supports_diarization(model.trim())
}

use reqwest::StatusCode;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteErrorKind {
    RateLimited,
    QuotaExceeded,
    Unauthorized,
    InvalidRequest,
    NotFound,
    UpstreamUnavailable,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteError {
    pub kind: RemoteErrorKind,
    pub status: u16,
    pub message: String,
    pub error_type: Option<String>,
    pub code: Option<String>,
    pub param: Option<String>,
    pub retry_after: Option<Duration>,
}

impl RemoteError {
    pub fn should_fallback(&self) -> bool {
        match (self.kind, self.status) {
            (RemoteErrorKind::UpstreamUnavailable, 0) => true,
            (RemoteErrorKind::UpstreamUnavailable, status) if status >= 500 => true,
            (RemoteErrorKind::Other, status) if status >= 500 => true,
            (_, 408) => true,
            _ => false,
        }
    }

    pub fn user_message(&self) -> String {
        match self.kind {
            RemoteErrorKind::RateLimited => {
                if let Some(retry_after) = self.retry_after {
                    let seconds = retry_after.as_secs().max(1);
                    format!(
                        "Remote speech rate limit reached. Try again in about {seconds} second{}.",
                        if seconds == 1 { "" } else { "s" }
                    )
                } else {
                    "Remote speech rate limit reached. Try again in a moment.".to_string()
                }
            }
            RemoteErrorKind::QuotaExceeded => {
                "Remote speech quota exceeded. Check your provider billing or usage limits."
                    .to_string()
            }
            RemoteErrorKind::Unauthorized => {
                "Remote speech API key is invalid or expired.".to_string()
            }
            RemoteErrorKind::NotFound => {
                "Remote speech endpoint or model was not found.".to_string()
            }
            RemoteErrorKind::UpstreamUnavailable => {
                "Remote speech provider is temporarily unavailable.".to_string()
            }
            RemoteErrorKind::InvalidRequest | RemoteErrorKind::Other => self.message.clone(),
        }
    }
}

impl fmt::Display for RemoteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.user_message())
    }
}

impl StdError for RemoteError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseFormat {
    Json,
    VerboseJson,
}

impl ResponseFormat {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::VerboseJson => "verbose_json",
        }
    }
}

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: Option<UpstreamErrorBody>,
    #[serde(default)]
    message: Option<String>,
    /// ElevenLabs.
    #[serde(default)]
    detail: Option<ErrorDetail>,
    /// Deepgram.
    #[serde(default)]
    err_code: Option<String>,
    #[serde(default)]
    err_msg: Option<String>,
    #[serde(default)]
    category: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpstreamErrorBody {
    message: String,
    #[serde(default, rename = "type")]
    error_type: Option<String>,
    #[serde(default, deserialize_with = "string_or_number")]
    code: Option<String>,
    #[serde(default)]
    param: Option<String>,
}

fn string_or_number<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    Ok(
        match Option::<serde_json::Value>::deserialize(deserializer)? {
            Some(serde_json::Value::String(code)) => Some(code),
            Some(serde_json::Value::Number(code)) => Some(code.to_string()),
            _ => None,
        },
    )
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ErrorDetail {
    Body(DetailBody),
    Validation(Vec<ValidationIssue>),
    Text(String),
}

#[derive(Debug, Deserialize)]
struct DetailBody {
    #[serde(default)]
    message: Option<String>,
    #[serde(default, rename = "type")]
    error_type: Option<String>,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    param: Option<String>,
    /// Older responses put the code here.
    #[serde(default)]
    status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ValidationIssue {
    #[serde(default)]
    msg: String,
    #[serde(default)]
    loc: Vec<serde_json::Value>,
}

#[derive(Default)]
struct ErrorFields {
    message: Option<String>,
    error_type: Option<String>,
    code: Option<String>,
    param: Option<String>,
}

impl ErrorEnvelope {
    fn into_fields(self) -> ErrorFields {
        if let Some(error) = self.error {
            return ErrorFields {
                message: Some(error.message),
                error_type: error.error_type,
                code: error.code,
                param: error.param,
            };
        }
        match self.detail {
            Some(ErrorDetail::Body(detail)) => ErrorFields {
                message: detail.message,
                error_type: detail.error_type,
                code: detail.code.or(detail.status),
                param: detail.param,
            },
            Some(ErrorDetail::Validation(issues)) => ErrorFields {
                message: Some(
                    issues
                        .iter()
                        .map(|issue| issue.msg.as_str())
                        .collect::<Vec<_>>()
                        .join("; "),
                ),
                param: issues
                    .first()
                    .and_then(|issue| issue.loc.last())
                    .and_then(|loc| loc.as_str())
                    .map(str::to_string),
                ..ErrorFields::default()
            },
            Some(ErrorDetail::Text(message)) => ErrorFields {
                message: Some(message),
                ..ErrorFields::default()
            },
            None => ErrorFields {
                message: self.err_msg.or(self.message),
                code: self.err_code.or(self.category),
                ..ErrorFields::default()
            },
        }
    }
}

pub fn config_error(message: impl Into<String>) -> RemoteError {
    RemoteError {
        kind: RemoteErrorKind::InvalidRequest,
        status: 0,
        message: message.into(),
        error_type: None,
        code: None,
        param: None,
        retry_after: None,
    }
}

pub fn transport_error(message: impl Into<String>) -> RemoteError {
    RemoteError {
        kind: RemoteErrorKind::UpstreamUnavailable,
        status: 0,
        message: message.into(),
        error_type: None,
        code: None,
        param: None,
        retry_after: None,
    }
}

pub fn parse_upstream_error(
    status: StatusCode,
    retry_after: Option<Duration>,
    body: &str,
) -> RemoteError {
    let fields = serde_json::from_str::<ErrorEnvelope>(body)
        .map(ErrorEnvelope::into_fields)
        .unwrap_or_default();
    let message = fields
        .message
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| body.trim().to_string());
    let message = if message.is_empty() {
        format!("Remote speech request failed with status {status}")
    } else {
        message
    };
    let ErrorFields {
        error_type,
        code,
        param,
        ..
    } = fields;
    let kind = classify_upstream_error(status, error_type.as_deref(), code.as_deref());

    RemoteError {
        kind,
        status: status.as_u16(),
        message,
        error_type,
        code,
        param,
        retry_after,
    }
}

fn classify_upstream_error(
    status: StatusCode,
    error_type: Option<&str>,
    code: Option<&str>,
) -> RemoteErrorKind {
    if matches_code(
        code,
        &[
            "rate_limit_exceeded",
            "rate_limit",
            "concurrent_limit_exceeded",
            "system_busy",
        ],
    ) {
        return RemoteErrorKind::RateLimited;
    }
    if matches_code(
        code,
        &[
            "insufficient_quota",
            "billing_not_active",
            "insufficient_credits",
            "quota_exceeded",
        ],
    ) {
        return RemoteErrorKind::QuotaExceeded;
    }
    if matches_code(
        code,
        &[
            "invalid_api_key",
            "invalid_authentication",
            "missing_api_key",
            "invalid_auth",
        ],
    ) {
        return RemoteErrorKind::Unauthorized;
    }
    if matches_code(code, &["model_not_found"]) {
        return RemoteErrorKind::NotFound;
    }

    if let Some(error_type) = error_type {
        let normalized = error_type.to_ascii_lowercase();
        if normalized.contains("insufficient_quota") || normalized.contains("billing") {
            return RemoteErrorKind::QuotaExceeded;
        }
        if normalized.contains("rate_limit") || normalized.contains("tokens") {
            return RemoteErrorKind::RateLimited;
        }
        if normalized.contains("invalid_request") {
            return RemoteErrorKind::InvalidRequest;
        }
        if normalized.contains("authentication") || normalized.contains("permission") {
            return RemoteErrorKind::Unauthorized;
        }
    }

    match status.as_u16() {
        401 | 403 => RemoteErrorKind::Unauthorized,
        402 => RemoteErrorKind::QuotaExceeded,
        404 => RemoteErrorKind::NotFound,
        408 | 429 => RemoteErrorKind::RateLimited,
        400 | 413 | 415 | 422 => RemoteErrorKind::InvalidRequest,
        code if code >= 500 => RemoteErrorKind::UpstreamUnavailable,
        _ => RemoteErrorKind::Other,
    }
}

fn matches_code(code: Option<&str>, expected: &[&str]) -> bool {
    code.is_some_and(|value| {
        let normalized = value.to_ascii_lowercase();
        expected.iter().any(|candidate| normalized == *candidate)
    })
}

pub fn parse_retry_after(value: Option<&reqwest::header::HeaderValue>) -> Option<Duration> {
    let raw = value?.to_str().ok()?.trim();
    if let Ok(seconds) = raw.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    httpdate::parse_http_date(raw)
        .ok()
        .map(|when| when.duration_since(SystemTime::now()).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_rate_limit_error() {
        let body = r#"{"error":{"message":"Rate limit reached","type":"tokens","code":"rate_limit_exceeded"}}"#;
        let err = parse_upstream_error(
            StatusCode::TOO_MANY_REQUESTS,
            Some(Duration::from_secs(30)),
            body,
        );
        assert_eq!(err.kind, RemoteErrorKind::RateLimited);
        assert_eq!(err.code.as_deref(), Some("rate_limit_exceeded"));
        assert_eq!(err.retry_after, Some(Duration::from_secs(30)));
    }

    #[test]
    fn openai_missing_model_is_not_found() {
        let body = r#"{"error":{"message":"The model `x` does not exist","type":"invalid_request_error","code":"model_not_found"}}"#;
        let err = parse_upstream_error(StatusCode::NOT_FOUND, None, body);
        assert_eq!(err.kind, RemoteErrorKind::NotFound);
    }

    #[test]
    fn numeric_error_codes_keep_the_message() {
        let body = r#"{"error":{"code":401,"message":"No auth credentials found"}}"#;
        let err = parse_upstream_error(StatusCode::UNAUTHORIZED, None, body);
        assert_eq!(err.kind, RemoteErrorKind::Unauthorized);
        assert_eq!(err.code.as_deref(), Some("401"));
        assert_eq!(err.message, "No auth credentials found");
    }

    #[test]
    fn upstream_503_can_fallback() {
        let err = parse_upstream_error(StatusCode::SERVICE_UNAVAILABLE, None, "upstream down");
        assert!(err.should_fallback());
    }

    #[test]
    fn unauthorized_does_not_fallback() {
        let err = parse_upstream_error(
            StatusCode::UNAUTHORIZED,
            None,
            r#"{"error":{"message":"Invalid API key","code":"invalid_api_key"}}"#,
        );
        assert!(!err.should_fallback());
    }

    #[test]
    fn parse_retry_after_accepts_seconds() {
        let value = reqwest::header::HeaderValue::from_static("30");
        assert_eq!(
            parse_retry_after(Some(&value)),
            Some(Duration::from_secs(30))
        );
    }

    #[test]
    fn parse_retry_after_accepts_http_date() {
        let retry_at = SystemTime::now() + Duration::from_secs(120);
        let value = reqwest::header::HeaderValue::from_str(&httpdate::fmt_http_date(retry_at))
            .expect("valid header");
        let parsed = parse_retry_after(Some(&value)).expect("retry date parses");
        assert!(parsed > Duration::ZERO);
        assert!(parsed <= Duration::from_secs(120));
    }
}
