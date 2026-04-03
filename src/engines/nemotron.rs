use std::path::{Path, PathBuf};

use parakeet_rs::Nemotron;

use crate::{TranscriptionEngine, TranscriptionResult};

/// Chunk size in samples for streaming (560ms at 16kHz).
pub const STREAMING_CHUNK_SAMPLES: usize = 8960;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NemotronModelParams;

#[derive(Debug, Clone, Default)]
pub struct NemotronInferenceParams {
    pub language: Option<String>,
}

pub struct NemotronEngine {
    loaded_model_path: Option<PathBuf>,
    runtime: Option<Nemotron>,
}

impl Default for NemotronEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl NemotronEngine {
    pub fn new() -> Self {
        Self {
            loaded_model_path: None,
            runtime: None,
        }
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

    /// Returns true if this engine supports streaming transcription.
    pub fn supports_streaming(&self) -> bool {
        true
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
        let runtime = Nemotron::from_pretrained(model_path, None).map_err(nemotron_error)?;
        self.loaded_model_path = Some(model_path.to_path_buf());
        self.runtime = Some(runtime);
        Ok(())
    }

    fn unload_model(&mut self) {
        self.loaded_model_path = None;
        self.runtime = None;
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        _params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let runtime = self.runtime_mut()?;
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
            "Missing Nemotron model files in {}: {}",
            model_path.display(),
            missing.join(", ")
        )));
    }

    Ok(())
}

fn nemotron_error(error: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    io_error(format!("parakeet-rs Nemotron error: {error}"))
}

fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}

#[cfg(test)]
mod tests {
    use super::NemotronModelParams;

    #[test]
    fn default_params() {
        let params = NemotronModelParams::default();
        assert_eq!(params, NemotronModelParams);
    }
}
