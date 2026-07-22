//! Engine backed by the macOS 26 SpeechAnalyzer via the Swift shim in
//! swift/apple_speech.swift. Models are built into the OS; there is nothing
//! to download or load from disk.

use std::error::Error;
use std::ffi::{c_char, CStr, CString};
use std::path::Path;

use serde::Deserialize;

use crate::{TranscriptionEngine, TranscriptionResult, TranscriptionSegment};

unsafe extern "C" {
    fn gs_apple_availability() -> i32;
    fn gs_apple_locale_status(locale: *const c_char) -> i32;
    fn gs_apple_stream_start(
        locale: *const c_char,
        long_form: i32,
        vocabulary_json: *const c_char,
    ) -> i64;
    fn gs_apple_stream_feed(handle: i64, samples: *const f32, count: usize) -> i32;
    fn gs_apple_stream_text(handle: i64) -> *mut c_char;
    fn gs_apple_stream_finish(handle: i64) -> *mut c_char;
    fn gs_apple_stream_cancel(handle: i64);
    fn gs_apple_string_free(ptr: *mut c_char);
    fn gs_apple_supported_locales() -> *mut c_char;
}

fn take_string(ptr: *mut c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let value = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { gs_apple_string_free(ptr) };
    Some(value)
}

fn locale_cstring(language: Option<&str>) -> CString {
    CString::new(language.unwrap_or("")).unwrap_or_default()
}

/// Whether the OS provides the speech engine (macOS 26+, Apple Silicon).
pub fn available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| unsafe { gs_apple_availability() == 0 })
}

/// Supported locales as BCP-47 identifiers, empty when unavailable.
pub fn supported_locales() -> Vec<String> {
    let raw = take_string(unsafe { gs_apple_supported_locales() }).unwrap_or_default();
    serde_json::from_str(&raw).unwrap_or_default()
}

/// 0 installed, 1 downloadable, 2 unsupported locale, anything else an error.
pub fn locale_status(language: Option<&str>) -> i32 {
    let locale = locale_cstring(language);
    unsafe { gs_apple_locale_status(locale.as_ptr()) }
}

#[derive(Debug, Default, Clone)]
pub struct AppleInferenceParams {
    pub language: Option<String>,
    /// Use the long-form transcriber (finer segments, no dictation extras).
    pub long_form: bool,
    pub dictionary: Vec<String>,
}

#[derive(Deserialize)]
struct ShimResult {
    #[serde(default)]
    text: String,
    #[serde(default)]
    segments: Vec<ShimSegment>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct ShimSegment {
    start: f32,
    end: f32,
    text: String,
}

#[derive(Default)]
pub struct AppleEngine {
    stream: Option<i64>,
    stream_language: Option<String>,
    stream_dictionary: Vec<String>,
}

fn vocabulary_cstring(terms: &[String]) -> CString {
    if terms.is_empty() {
        return CString::default();
    }
    CString::new(serde_json::to_string(terms).unwrap_or_default()).unwrap_or_default()
}

fn start_session(language: Option<&str>, long_form: bool, dictionary: &[String]) -> Option<i64> {
    let locale = locale_cstring(language);
    let vocabulary = vocabulary_cstring(dictionary);
    let handle = unsafe {
        gs_apple_stream_start(locale.as_ptr(), i32::from(long_form), vocabulary.as_ptr())
    };
    (handle != 0).then_some(handle)
}

impl AppleEngine {
    pub fn new() -> Self {
        Self::default()
    }

    fn finish_stream(handle: i64) -> Result<ShimResult, Box<dyn Error>> {
        let raw = take_string(unsafe { gs_apple_stream_finish(handle) })
            .ok_or("apple speech returned no result")?;
        let parsed: ShimResult = serde_json::from_str(&raw)?;
        if let Some(error) = parsed.error {
            return Err(error.into());
        }
        Ok(parsed)
    }

    /// Sets language and vocabulary for the next streaming session.
    pub fn configure_stream(&mut self, language: Option<String>, dictionary: Vec<String>) {
        self.stream_language = language;
        self.stream_dictionary = dictionary;
    }

    pub fn transcribe_chunk(&mut self, chunk: &[f32]) -> Result<(), Box<dyn Error>> {
        let handle = match self.stream {
            Some(handle) => handle,
            None => {
                let handle = start_session(
                    self.stream_language.as_deref(),
                    false,
                    &self.stream_dictionary,
                )
                .ok_or("failed to start apple speech session")?;
                self.stream = Some(handle);
                handle
            }
        };
        let status = unsafe { gs_apple_stream_feed(handle, chunk.as_ptr(), chunk.len()) };
        if status != 0 {
            return Err("apple speech rejected audio chunk".into());
        }
        Ok(())
    }

    pub fn get_transcript(&self) -> String {
        match self.stream {
            Some(handle) => {
                take_string(unsafe { gs_apple_stream_text(handle) }).unwrap_or_default()
            }
            None => String::new(),
        }
    }

    /// Ends the stream and returns the fully finalized transcript.
    pub fn finalize(&mut self) -> Result<String, Box<dyn Error>> {
        let Some(handle) = self.stream.take() else {
            return Ok(String::new());
        };
        Ok(Self::finish_stream(handle)?.text)
    }

    pub fn reset(&mut self) {
        if let Some(handle) = self.stream.take() {
            unsafe { gs_apple_stream_cancel(handle) };
        }
    }
}

impl Drop for AppleEngine {
    fn drop(&mut self) {
        self.reset();
    }
}

impl TranscriptionEngine for AppleEngine {
    type InferenceParams = AppleInferenceParams;
    type ModelParams = ();

    fn load_model_with_params(
        &mut self,
        _model_path: &Path,
        _params: Self::ModelParams,
    ) -> Result<(), Box<dyn Error>> {
        if !available() {
            return Err("Apple speech requires macOS 26 or later on Apple Silicon".into());
        }
        Ok(())
    }

    fn unload_model(&mut self) {
        self.reset();
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn Error>> {
        let params = params.unwrap_or_default();
        let language = params.language.clone();
        let handle = start_session(language.as_deref(), params.long_form, &params.dictionary)
            .ok_or("failed to start apple speech session")?;
        // Feed in ~1s chunks so the analyzer can pipeline while we copy.
        for chunk in samples.chunks(16_000) {
            let status = unsafe { gs_apple_stream_feed(handle, chunk.as_ptr(), chunk.len()) };
            if status != 0 {
                unsafe { gs_apple_stream_cancel(handle) };
                return Err("apple speech rejected audio chunk".into());
            }
        }
        let parsed = Self::finish_stream(handle)?;
        let segments: Vec<TranscriptionSegment> = parsed
            .segments
            .into_iter()
            .map(|segment| TranscriptionSegment {
                start: segment.start,
                end: segment.end,
                text: segment.text,
            })
            .collect();
        Ok(TranscriptionResult {
            text: parsed.text,
            segments: if segments.is_empty() {
                None
            } else {
                Some(segments)
            },
            words: None,
            language,
        })
    }
}
