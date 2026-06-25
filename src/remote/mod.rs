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

/// Reports whether an endpoint accepts speaker-diarized transcription requests.
pub fn supports_diarization(endpoint: &str) -> bool {
    provider::resolve_profile(endpoint).supports_diarization
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
            RemoteErrorKind::InvalidRequest => self.message.clone(),
            RemoteErrorKind::NotFound => {
                "Remote speech endpoint or model was not found.".to_string()
            }
            RemoteErrorKind::UpstreamUnavailable => {
                "Remote speech provider is temporarily unavailable.".to_string()
            }
            RemoteErrorKind::Other => self.message.clone(),
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
}

#[derive(Debug, Deserialize)]
struct UpstreamErrorBody {
    message: String,
    #[serde(default, rename = "type")]
    error_type: Option<String>,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    param: Option<String>,
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
    let parsed = serde_json::from_str::<ErrorEnvelope>(body).ok();
    let upstream = parsed.as_ref().and_then(|envelope| envelope.error.as_ref());
    let message = upstream
        .map(|error| error.message.clone())
        .or_else(|| {
            parsed
                .as_ref()
                .and_then(|envelope| envelope.message.clone())
        })
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| body.trim().to_string());
    let message = if message.is_empty() {
        format!("Remote speech request failed with status {status}")
    } else {
        message
    };
    let error_type = upstream.and_then(|error| error.error_type.clone());
    let code = upstream.and_then(|error| error.code.clone());
    let param = upstream.and_then(|error| error.param.clone());
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
    if matches_code(code, &["rate_limit_exceeded", "rate_limit"]) {
        return RemoteErrorKind::RateLimited;
    }
    if matches_code(code, &["insufficient_quota", "billing_not_active"]) {
        return RemoteErrorKind::QuotaExceeded;
    }
    if matches_code(code, &["invalid_api_key", "invalid_authentication"]) {
        return RemoteErrorKind::Unauthorized;
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
