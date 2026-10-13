//! Text generation with Apple's on-device model (FoundationModels, macOS 26+)
//! through the Swift shim in swift/apple_cleanup.swift. On other platforms,
//! builds without `cleanup-apple`, or older macOS, everything reports
//! `Unsupported`.

use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleAvailability {
    Available,
    /// Supported device, but Apple Intelligence is turned off in System Settings.
    NotEnabled,
    /// Model assets still downloading; may become available.
    NotReady,
    /// Wrong OS, device, or build.
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleErrorKind {
    /// Instructions, prompt and response don't fit the context window.
    ContextTooLarge,
    /// Guardrail violation or model refusal.
    Refused,
    UnsupportedLanguage,
    /// Rate limited, or another request is running.
    Busy,
    /// Model not available (Apple Intelligence off, assets missing, old OS).
    Unavailable,
    TimedOut,
    Other,
}

#[derive(Debug, Clone)]
pub struct AppleError {
    pub kind: AppleErrorKind,
    pub message: String,
}

impl std::fmt::Display for AppleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AppleError {}

/// Whether the on-device model can run right now. Cheap; safe to call per use.
pub fn apple_availability() -> AppleAvailability {
    #[cfg(apple_cleanup)]
    {
        apple::availability()
    }
    #[cfg(not(apple_cleanup))]
    {
        AppleAvailability::Unsupported
    }
}

/// Load the model and prefill `instructions` so the next `apple_generate`
/// with the same instructions starts faster. Returns immediately.
#[allow(unused_variables)]
pub fn apple_prewarm(instructions: &str) -> Result<(), AppleError> {
    #[cfg(apple_cleanup)]
    {
        apple::prewarm(instructions)
    }
    #[cfg(not(apple_cleanup))]
    {
        Err(unsupported())
    }
}

/// One-shot generation. Temperature 0 uses greedy sampling. Blocking; call
/// from a blocking-capable thread. The request is cancelled after `timeout`.
#[allow(unused_variables)]
pub fn apple_generate(
    instructions: &str,
    prompt: &str,
    temperature: f32,
    max_response_tokens: Option<u32>,
    timeout: Duration,
) -> Result<String, AppleError> {
    #[cfg(apple_cleanup)]
    {
        apple::generate(
            instructions,
            prompt,
            temperature,
            max_response_tokens,
            timeout,
        )
    }
    #[cfg(not(apple_cleanup))]
    {
        Err(unsupported())
    }
}

#[cfg(not(apple_cleanup))]
fn unsupported() -> AppleError {
    AppleError {
        kind: AppleErrorKind::Unavailable,
        message: "on-device model is not supported in this build".to_string(),
    }
}

#[cfg(apple_cleanup)]
mod apple {
    use std::ffi::{CStr, CString, c_char};
    use std::time::Duration;

    use serde::Deserialize;

    use super::{AppleAvailability, AppleError, AppleErrorKind};

    unsafe extern "C" {
        fn gs_cleanup_availability() -> i32;
        fn gs_cleanup_prewarm(instructions: *const c_char) -> i32;
        fn gs_cleanup_generate(
            instructions: *const c_char,
            prompt: *const c_char,
            temperature: f64,
            max_tokens: i32,
            timeout_ms: i64,
        ) -> *mut c_char;
        fn gs_cleanup_string_free(ptr: *mut c_char);
    }

    pub fn availability() -> AppleAvailability {
        match unsafe { gs_cleanup_availability() } {
            0 => AppleAvailability::Available,
            1 => AppleAvailability::NotEnabled,
            2 => AppleAvailability::NotReady,
            _ => AppleAvailability::Unsupported,
        }
    }

    pub fn prewarm(instructions: &str) -> Result<(), AppleError> {
        let instructions = cstring(instructions)?;
        match unsafe { gs_cleanup_prewarm(instructions.as_ptr()) } {
            0 => Ok(()),
            _ => Err(error(
                AppleErrorKind::Unavailable,
                "on-device model unavailable",
            )),
        }
    }

    #[derive(Deserialize)]
    struct ShimResponse {
        #[serde(default)]
        text: String,
        #[serde(default)]
        error: Option<String>,
        #[serde(default)]
        kind: Option<String>,
    }

    pub fn generate(
        instructions: &str,
        prompt: &str,
        temperature: f32,
        max_response_tokens: Option<u32>,
        timeout: Duration,
    ) -> Result<String, AppleError> {
        let instructions = cstring(instructions)?;
        let prompt = cstring(prompt)?;
        if timeout.is_zero() {
            return Err(error(AppleErrorKind::TimedOut, "on-device model timed out"));
        }
        let max_tokens = max_response_tokens.map_or(0, |n| i32::try_from(n).unwrap_or(i32::MAX));
        let timeout_ms = i64::try_from(timeout.as_micros().div_ceil(1_000)).unwrap_or(i64::MAX);
        let ptr = unsafe {
            gs_cleanup_generate(
                instructions.as_ptr(),
                prompt.as_ptr(),
                f64::from(temperature),
                max_tokens,
                timeout_ms,
            )
        };
        let raw = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        unsafe { gs_cleanup_string_free(ptr) };
        let response: ShimResponse = serde_json::from_str(&raw)
            .map_err(|err| error(AppleErrorKind::Other, format!("bad shim response: {err}")))?;
        match response.error {
            Some(message) => Err(error(error_kind(response.kind.as_deref()), message)),
            None => Ok(response.text.trim().to_string()),
        }
    }

    fn error_kind(kind: Option<&str>) -> AppleErrorKind {
        match kind {
            Some("context") => AppleErrorKind::ContextTooLarge,
            Some("refused") => AppleErrorKind::Refused,
            Some("language") => AppleErrorKind::UnsupportedLanguage,
            Some("busy") => AppleErrorKind::Busy,
            Some("unavailable") => AppleErrorKind::Unavailable,
            Some("timeout") => AppleErrorKind::TimedOut,
            _ => AppleErrorKind::Other,
        }
    }

    fn cstring(value: &str) -> Result<CString, AppleError> {
        CString::new(value).map_err(|_| error(AppleErrorKind::Other, "text contains a NUL byte"))
    }

    fn error(kind: AppleErrorKind, message: impl Into<String>) -> AppleError {
        AppleError {
            kind,
            message: message.into(),
        }
    }
}
