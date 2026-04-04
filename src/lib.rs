pub mod audio;
pub mod diarization;
pub mod dictionary;
pub mod engines;

use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionResult {
    pub text: String,
    pub segments: Option<Vec<TranscriptionSegment>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionSegment {
    /// Segment start time in seconds.
    pub start: f32,
    /// Segment end time in seconds.
    pub end: f32,
    pub text: String,
}

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

pub trait SpeakerDiarizationEngine {
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

    /// Diarize already-decoded samples (16 kHz, mono, f32 in [-1, 1]).
    fn diarize_samples(
        &mut self,
        samples: Vec<f32>,
    ) -> Result<Vec<diarization::SpeakerDiarizationSegment>, Box<dyn std::error::Error>>;

    /// Diarize a WAV file.
    fn diarize_file(
        &mut self,
        wav_path: &Path,
    ) -> Result<Vec<diarization::SpeakerDiarizationSegment>, Box<dyn std::error::Error>> {
        let samples = audio::read_wav_samples(wav_path)?;
        self.diarize_samples(samples)
    }
}
