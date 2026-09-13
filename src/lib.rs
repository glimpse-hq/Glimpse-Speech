#[cfg(feature = "api")]
pub mod api;
pub mod audio;
pub mod cleanup;
#[cfg(feature = "cli")]
pub mod cli;
pub mod dictionary;
pub mod engines;
pub mod models;
pub mod provider;
#[cfg(feature = "remote")]
pub mod remote;
pub mod service;
#[cfg(feature = "whisper")]
pub mod vad;

use std::path::Path;

#[cfg(feature = "whisper")]
pub(crate) fn silence_native_logs() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(native_log::install);
}

#[cfg(not(feature = "whisper"))]
pub(crate) fn silence_native_logs() {}

/// Route transcribe.cpp and its ggml diagnostics into the `log` facade once,
/// so they never reach the process's stderr.
#[cfg(transcribe_engine)]
pub(crate) fn silence_transcribe_cpp_logs() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(transcribe_cpp::init_logging);
}

#[cfg(feature = "whisper")]
mod native_log {
    use std::collections::VecDeque;
    use std::ffi::{CStr, c_char, c_void};
    use std::sync::Mutex;

    use whisper_rs::whisper_rs_sys;

    const MAX_LINES: usize = 64;

    static CORE_ML_LINES: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());

    unsafe extern "C" fn capture(
        _level: whisper_rs_sys::ggml_log_level,
        text: *const c_char,
        _user_data: *mut c_void,
    ) {
        if text.is_null() {
            return;
        }
        let message = unsafe { CStr::from_ptr(text) }.to_string_lossy();
        if message.contains("Core ML") {
            let mut lines = CORE_ML_LINES.lock().unwrap();
            if lines.len() >= MAX_LINES {
                lines.pop_front();
            }
            lines.push_back(message.trim().to_string());
        }
    }

    pub(crate) fn install() {
        unsafe {
            whisper_rs_sys::whisper_log_set(Some(capture), std::ptr::null_mut());
            whisper_rs_sys::ggml_log_set(Some(capture), std::ptr::null_mut());
        }
    }

    pub(crate) fn take_lines() -> Vec<String> {
        CORE_ML_LINES.lock().unwrap().drain(..).collect()
    }
}

/// Drains captured whisper.cpp Core ML log lines.
#[cfg(feature = "whisper")]
pub fn take_coreml_log() -> Vec<String> {
    native_log::take_lines()
}

/// Raw output of a transcription engine: text plus optional segments.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub segments: Option<Vec<TranscriptionSegment>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<TranscriptionSegment>>,
    /// Language detected by the engine, when supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionSegment {
    /// Segment start time in seconds.
    pub start: f32,
    /// Segment end time in seconds.
    pub end: f32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Transcription {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<TranscriptionSegment>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<TranscriptionSegment>>,
    pub model_id: String,
    pub language: Option<String>,
    pub duration_ms: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampGranularity {
    Segment,
    Word,
}

impl TimestampGranularity {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Segment => "segment",
            Self::Word => "word",
        }
    }
}

impl std::str::FromStr for TimestampGranularity {
    type Err = UnsupportedValue;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "segment" => Ok(Self::Segment),
            "word" => Ok(Self::Word),
            other => Err(UnsupportedValue(other.to_string())),
        }
    }
}

/// A string that does not name any variant of the enum being parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedValue(pub String);

impl std::fmt::Display for UnsupportedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}`", self.0)
    }
}

impl std::error::Error for UnsupportedValue {}

pub trait TranscriptionEngine {
    type InferenceParams;
    type ModelParams: Default;

    /// Load with default model params.
    fn load_model(&mut self, model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.load_model_with_params(model_path, Self::ModelParams::default())
    }

    /// Load with explicit model params.
    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>>;

    fn unload_model(&mut self);

    /// Transcribe already-decoded samples (16 kHz, mono, f32 in [-1, 1]).
    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>>;

    /// Transcribe a WAV file.
    fn transcribe_file(
        &mut self,
        wav_path: &Path,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let samples = audio::read_wav_samples(wav_path)?;
        self.transcribe_samples(samples, params)
    }
}
