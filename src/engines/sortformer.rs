use std::path::{Path, PathBuf};

use parakeet_rs::sortformer::{
    DiarizationConfig, Sortformer, SpeakerSegment as SortformerSpeakerSegment,
};

use crate::{diarization::SpeakerDiarizationSegment, SpeakerDiarizationEngine};

const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortformerRuntimeInfo {
    pub chunk_len: usize,
    pub fifo_len: usize,
    pub spkcache_len: usize,
    pub right_context: usize,
}

impl SortformerRuntimeInfo {
    pub fn latency(&self) -> f32 {
        (self.chunk_len + self.right_context) as f32 * 0.08
    }
}

#[derive(Debug, Clone)]
pub struct SortformerModelParams {
    pub config: DiarizationConfig,
    pub chunk_len: Option<usize>,
    pub fifo_len: Option<usize>,
    pub spkcache_len: Option<usize>,
    pub right_context: Option<usize>,
}

impl Default for SortformerModelParams {
    fn default() -> Self {
        Self::callhome()
    }
}

impl SortformerModelParams {
    pub fn callhome() -> Self {
        Self::with_config(DiarizationConfig::callhome())
    }

    pub fn dihard3() -> Self {
        Self::with_config(DiarizationConfig::dihard3())
    }

    pub fn with_config(config: DiarizationConfig) -> Self {
        Self {
            config,
            chunk_len: None,
            fifo_len: None,
            spkcache_len: None,
            right_context: None,
        }
    }

    pub fn with_streaming_overrides(
        mut self,
        chunk_len: usize,
        fifo_len: usize,
        spkcache_len: usize,
        right_context: usize,
    ) -> Self {
        self.chunk_len = Some(chunk_len);
        self.fifo_len = Some(fifo_len);
        self.spkcache_len = Some(spkcache_len);
        self.right_context = Some(right_context);
        self
    }
}

pub struct SortformerEngine {
    loaded_model_path: Option<PathBuf>,
    runtime: Option<Sortformer>,
}

impl Default for SortformerEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SortformerEngine {
    pub fn new() -> Self {
        Self {
            loaded_model_path: None,
            runtime: None,
        }
    }

    fn runtime_mut(&mut self) -> Result<&mut Sortformer, Box<dyn std::error::Error>> {
        self.runtime
            .as_mut()
            .ok_or_else(|| io_error("Model not loaded. Call load_model() first."))
    }

    pub fn model_path(&self) -> Option<&Path> {
        self.loaded_model_path.as_deref()
    }

    pub fn runtime_info(&self) -> Option<SortformerRuntimeInfo> {
        self.runtime.as_ref().map(|runtime| SortformerRuntimeInfo {
            chunk_len: runtime.chunk_len,
            fifo_len: runtime.fifo_len,
            spkcache_len: runtime.spkcache_len,
            right_context: runtime.right_context,
        })
    }
}

impl Drop for SortformerEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl SpeakerDiarizationEngine for SortformerEngine {
    type ModelParams = SortformerModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        validate_model_path(model_path)?;

        let mut runtime =
            Sortformer::with_config(model_path, None, params.config).map_err(sortformer_error)?;

        if let Some(chunk_len) = params.chunk_len {
            runtime.chunk_len = chunk_len;
        }
        if let Some(fifo_len) = params.fifo_len {
            runtime.fifo_len = fifo_len;
        }
        if let Some(spkcache_len) = params.spkcache_len {
            runtime.spkcache_len = spkcache_len;
        }
        if let Some(right_context) = params.right_context {
            runtime.right_context = right_context;
        }

        runtime.reset_state();

        self.loaded_model_path = Some(model_path.to_path_buf());
        self.runtime = Some(runtime);
        Ok(())
    }

    fn unload_model(&mut self) {
        self.loaded_model_path = None;
        self.runtime = None;
    }

    fn diarize_samples(
        &mut self,
        samples: Vec<f32>,
    ) -> Result<Vec<SpeakerDiarizationSegment>, Box<dyn std::error::Error>> {
        let segments = self
            .runtime_mut()?
            .diarize(samples, SAMPLE_RATE, 1)
            .map_err(sortformer_error)?;
        Ok(map_segments(segments))
    }
}

fn map_segments(segments: Vec<SortformerSpeakerSegment>) -> Vec<SpeakerDiarizationSegment> {
    segments
        .into_iter()
        .map(|segment| SpeakerDiarizationSegment {
            start: segment.start as f32 / SAMPLE_RATE as f32,
            end: segment.end as f32 / SAMPLE_RATE as f32,
            speaker_id: segment.speaker_id,
        })
        .collect()
}

fn validate_model_path(model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !model_path.exists() {
        return Err(io_error(format!(
            "Sortformer diarization model not found: {}",
            model_path.display()
        )));
    }

    if !model_path.is_file() {
        return Err(io_error(format!(
            "Sortformer diarization model path must be a file: {}",
            model_path.display()
        )));
    }

    let is_onnx = model_path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("onnx"))
        .unwrap_or(false);

    if !is_onnx {
        return Err(io_error(format!(
            "Sortformer diarization model must be an ONNX file: {}",
            model_path.display()
        )));
    }

    Ok(())
}

fn sortformer_error(error: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    io_error(format!("parakeet-rs Sortformer error: {error}"))
}

fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}
