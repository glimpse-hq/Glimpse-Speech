use std::{borrow::Cow, path::Path};

use parakeet_rs::{Nemotron, NemotronMode};

use crate::{TranscriptionEngine, TranscriptionResult};

/// Chunk size in samples for streaming (560ms at 16kHz).
pub const STREAMING_CHUNK_SAMPLES: usize = 8960;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NemotronModelParams;

#[derive(Debug, Clone, Default)]
pub struct NemotronInferenceParams {
    pub language: Option<String>,
}

#[derive(Default)]
pub struct NemotronEngine {
    runtime: Option<Nemotron>,
}

impl NemotronEngine {
    pub fn new() -> Self {
        Self::default()
    }

    fn runtime_mut(&mut self) -> Result<&mut Nemotron, Box<dyn std::error::Error>> {
        self.runtime
            .as_mut()
            .ok_or_else(|| io_error("Model not loaded. Call load_model() first."))
    }

    /// Process a single streaming chunk (~560ms of audio at 16kHz).
    /// Returns the incremental text produced by this chunk (may be empty).
    pub fn transcribe_chunk(
        &mut self,
        samples: &[f32],
    ) -> Result<String, Box<dyn std::error::Error>> {
        let runtime = self.runtime_mut()?;
        runtime.transcribe_chunk(samples).map_err(nemotron_error)
    }

    /// Get the full accumulated transcript from all chunks processed so far.
    pub fn get_transcript(&self) -> String {
        self.runtime
            .as_ref()
            .map(|r| r.get_transcript())
            .unwrap_or_default()
    }

    /// Reset streaming state for a new transcription session.
    pub fn reset(&mut self) {
        if let Some(runtime) = self.runtime.as_mut() {
            runtime.reset();
        }
    }
}

impl Drop for NemotronEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for NemotronEngine {
    type InferenceParams = NemotronInferenceParams;
    type ModelParams = NemotronModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        _params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        validate_model_path(model_path)?;
        let exec_config = parakeet_rs::ExecutionConfig::default()
            .with_intra_threads(crate::engines::inference_threads());
        let runtime =
            Nemotron::from_pretrained(model_path, Some(exec_config)).map_err(nemotron_error)?;
        self.runtime = Some(runtime);
        Ok(())
    }

    fn unload_model(&mut self) {
        self.runtime = None;
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let runtime = self.runtime_mut()?;
        apply_language(runtime, params.as_ref().and_then(|p| p.language.as_deref()))?;
        runtime.reset();

        // Feed all audio through the streaming interface in chunks
        for chunk in samples.chunks(STREAMING_CHUNK_SAMPLES) {
            let _ = runtime.transcribe_chunk(chunk).map_err(nemotron_error)?;
        }

        let text = runtime.get_transcript().trim().to_string();
        runtime.reset();

        Ok(TranscriptionResult {
            text,
            segments: None,
            words: None,
            language: None,
        })
    }
}

fn validate_model_path(model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !model_path.exists() {
        return Err(io_error(format!(
            "Nemotron model directory not found: {}",
            model_path.display()
        )));
    }

    if !model_path.is_dir() {
        return Err(io_error(format!(
            "Nemotron model path must be a directory: {}",
            model_path.display()
        )));
    }

    Ok(())
}

fn nemotron_error(error: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    io_error(format!("parakeet-rs Nemotron error: {error}"))
}

fn apply_language(
    runtime: &mut Nemotron,
    language: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    if runtime.mode() != NemotronMode::Multilingual {
        return Ok(());
    }

    let Some(language) = language.and_then(nemotron_language) else {
        return Ok(());
    };

    runtime
        .set_target_lang(language.as_ref())
        .map_err(nemotron_error)
}

fn nemotron_language(language: &str) -> Option<Cow<'_, str>> {
    let language = language.trim();
    if language.is_empty() || language == "auto" {
        return Some(Cow::Borrowed("auto"));
    }

    match language {
        "he" => return Some(Cow::Borrowed("he-IL")),
        "ja" => return Some(Cow::Borrowed("ja-JP")),
        "mt" => return Some(Cow::Borrowed("mt-MT")),
        "th" => return Some(Cow::Borrowed("th-TH")),
        "vi" => return Some(Cow::Borrowed("vi-VN")),
        "zh" => return Some(Cow::Borrowed("zh-CN")),
        _ => {}
    }

    if NEMOTRON_LANGUAGE_CODES.contains(&language) {
        Some(Cow::Borrowed(language))
    } else {
        Some(Cow::Borrowed("auto"))
    }
}

const NEMOTRON_LANGUAGE_CODES: &[&str] = &[
    "ar", "ar-AR", "bg", "bg-BG", "cs", "cs-CZ", "da", "da-DK", "de", "de-DE", "el", "el-GR", "en",
    "en-GB", "en-US", "es", "es-ES", "es-US", "et", "et-EE", "fi", "fi-FI", "fr", "fr-CA", "fr-FR",
    "he-IL", "hi", "hi-IN", "hr", "hr-HR", "hu", "hu-HU", "it", "it-IT", "ja-JP", "ko", "ko-KR",
    "lt", "lt-LT", "lv", "lv-LV", "mt-MT", "nb", "nb-NO", "nl", "nl-NL", "nn", "nn-NO", "no", "pl",
    "pl-PL", "pt", "pt-BR", "pt-PT", "ro", "ro-RO", "ru", "ru-RU", "sk", "sk-SK", "sl", "sl-SI",
    "sv", "sv-SE", "th-TH", "tr", "tr-TR", "uk", "uk-UA", "vi-VN", "zh-CN",
];

fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}
