use std::path::Path;

use parakeet_rs::ParakeetUnified;

use crate::{TranscriptionEngine, TranscriptionResult};

/// Chunk size in samples for streaming (~560ms at 16kHz). The unified model
/// buffers internally and only emits once a full chunk plus its right-context
/// window has arrived (or on `flush`), so this feed size is not load-bearing;
/// it mirrors the Nemotron cadence for a comparable live-update rate.
pub const STREAMING_CHUNK_SAMPLES: usize = 8960;

const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnifiedModelParams;

#[derive(Debug, Clone, Default)]
pub struct UnifiedInferenceParams {
    /// The unified model is English-only and does not expose language forcing,
    /// so this is currently a no-op kept for signature parity with Nemotron.
    pub language: Option<String>,
}

#[derive(Default)]
pub struct UnifiedEngine {
    runtime: Option<ParakeetUnified>,
}

impl UnifiedEngine {
    pub fn new() -> Self {
        Self::default()
    }

    fn runtime_mut(&mut self) -> Result<&mut ParakeetUnified, Box<dyn std::error::Error>> {
        self.runtime
            .as_mut()
            .ok_or_else(|| io_error("Model not loaded. Call load_model() first."))
    }

    /// Feed a single streaming chunk (~560ms of audio at 16kHz). Returns the
    /// incremental text produced so far (may be empty: the model holds a
    /// right-context tail until enough audio has arrived, or until `flush`).
    pub fn transcribe_chunk(
        &mut self,
        samples: &[f32],
    ) -> Result<String, Box<dyn std::error::Error>> {
        let runtime = self.runtime_mut()?;
        runtime.transcribe_chunk(samples).map_err(unified_error)
    }

    /// Process whatever audio is still buffered. Call once when the stream
    /// ends so the trailing words held in the right-context window are emitted.
    pub fn flush(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let runtime = self.runtime_mut()?;
        runtime.flush().map_err(unified_error)
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

impl Drop for UnifiedEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for UnifiedEngine {
    type InferenceParams = UnifiedInferenceParams;
    type ModelParams = UnifiedModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        _params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        validate_model_path(model_path)?;
        let exec_config = parakeet_rs::ExecutionConfig::default()
            .with_intra_threads(crate::engines::inference_threads());
        let runtime = ParakeetUnified::from_pretrained(model_path, Some(exec_config))
            .map_err(unified_error)?;
        self.runtime = Some(runtime);
        Ok(())
    }

    fn unload_model(&mut self) {
        self.runtime = None;
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        _params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let runtime = self.runtime_mut()?;
        // `transcribe_audio` runs the offline path, which resets streaming
        // state internally before and is independent of any live session.
        let text = runtime
            .transcribe_audio(samples, SAMPLE_RATE, 1)
            .map_err(unified_error)?;

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            segments: None,
            words: None,
            language: None,
        })
    }
}

const REQUIRED_FILES: [&str; 4] = [
    "encoder.onnx",
    "encoder.onnx.data",
    "decoder_joint.onnx",
    "tokenizer.model",
];

fn validate_model_path(model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !model_path.exists() {
        return Err(io_error(format!(
            "Unified model directory not found: {}",
            model_path.display()
        )));
    }

    if !model_path.is_dir() {
        return Err(io_error(format!(
            "Unified model path must be a directory: {}",
            model_path.display()
        )));
    }

    let missing: Vec<String> = REQUIRED_FILES
        .iter()
        .filter_map(|name| {
            let path = model_path.join(name);
            if path.exists() {
                None
            } else {
                Some((*name).to_string())
            }
        })
        .collect();

    if !missing.is_empty() {
        return Err(io_error(format!(
            "Missing Unified model files in {}: {}",
            model_path.display(),
            missing.join(", ")
        )));
    }

    Ok(())
}

fn unified_error(error: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    io_error(format!("parakeet-rs Unified error: {error}"))
}

fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}
