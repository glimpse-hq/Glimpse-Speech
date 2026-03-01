use std::path::{Path, PathBuf};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::{
    dictionary::build_dictionary_prompt, TranscriptionEngine, TranscriptionResult,
    TranscriptionSegment,
};

#[derive(Debug, Clone)]
pub struct WhisperModelParams {
    pub use_gpu: bool,
}

impl Default for WhisperModelParams {
    fn default() -> Self {
        Self { use_gpu: true }
    }
}

#[derive(Debug, Clone)]
pub struct WhisperInferenceParams {
    pub language: Option<String>,
    pub translate: bool,
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
    pub suppress_blank: bool,
    pub suppress_non_speech_tokens: bool,
    pub no_speech_thold: f32,
    pub dictionary: Vec<String>,
    pub initial_prompt: Option<String>,
}

impl Default for WhisperInferenceParams {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            print_special: false,
            print_progress: false,
            print_realtime: false,
            print_timestamps: false,
            suppress_blank: true,
            suppress_non_speech_tokens: true,
            no_speech_thold: 0.2,
            dictionary: Vec::new(),
            initial_prompt: None,
        }
    }
}

pub struct WhisperEngine {
    loaded_model_path: Option<PathBuf>,
    state: Option<whisper_rs::WhisperState>,
    context: Option<whisper_rs::WhisperContext>,
}

impl Default for WhisperEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperEngine {
    pub fn new() -> Self {
        Self {
            loaded_model_path: None,
            state: None,
            context: None,
        }
    }
}

impl Drop for WhisperEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for WhisperEngine {
    type InferenceParams = WhisperInferenceParams;
    type ModelParams = WhisperModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| io_error("model path is not valid UTF-8"))?;

        let context_params = WhisperContextParameters {
            use_gpu: params.use_gpu,
            ..WhisperContextParameters::default()
        };
        let context = WhisperContext::new_with_params(model_path_str, context_params)?;
        let state = context.create_state()?;

        self.context = Some(context);
        self.state = Some(state);
        self.loaded_model_path = Some(model_path.to_path_buf());
        Ok(())
    }

    fn unload_model(&mut self) {
        self.loaded_model_path = None;
        self.state = None;
        self.context = None;
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| io_error("Model not loaded. Call load_model() first."))?;

        let whisper_params = params.unwrap_or_default();

        let mut full_params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 3,
            patience: -1.0,
        });
        full_params.set_language(whisper_params.language.as_deref());
        full_params.set_translate(whisper_params.translate);
        full_params.set_print_special(whisper_params.print_special);
        full_params.set_print_progress(whisper_params.print_progress);
        full_params.set_print_realtime(whisper_params.print_realtime);
        full_params.set_print_timestamps(whisper_params.print_timestamps);
        full_params.set_suppress_blank(whisper_params.suppress_blank);
        full_params.set_suppress_nst(whisper_params.suppress_non_speech_tokens);
        full_params.set_no_speech_thold(whisper_params.no_speech_thold);

        let initial_prompt = whisper_params
            .initial_prompt
            .or_else(|| build_dictionary_prompt(&whisper_params.dictionary));

        if let Some(prompt) = initial_prompt.as_deref() {
            full_params.set_initial_prompt(prompt);
        }

        state.full(full_params, &samples)?;

        let num_segments = state.full_n_segments();
        let mut segments = Vec::new();
        let mut full_text = String::new();

        for i in 0..num_segments {
            let Some(segment) = state.get_segment(i) else {
                continue;
            };
            let text = segment
                .to_str_lossy()
                .map_err(|error| {
                    io_error(format!("failed to decode whisper segment text: {error}"))
                })?
                .to_string();
            let start = segment.start_timestamp() as f32 / 100.0;
            let end = segment.end_timestamp() as f32 / 100.0;

            segments.push(TranscriptionSegment {
                start,
                end,
                text: text.clone(),
            });
            full_text.push_str(&text);
        }

        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            segments: Some(segments),
        })
    }
}

fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}
