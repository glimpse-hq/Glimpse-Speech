use std::path::Path;

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
    pub word_timestamps: bool,
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
            word_timestamps: false,
        }
    }
}

#[derive(Default)]
pub struct WhisperEngine {
    state: Option<whisper_rs::WhisperState>,
    context: Option<whisper_rs::WhisperContext>,
}

impl WhisperEngine {
    pub fn new() -> Self {
        Self::default()
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
            flash_attn: true,
            ..WhisperContextParameters::default()
        };
        let context = WhisperContext::new_with_params(model_path_str, context_params)?;
        let state = context.create_state()?;

        self.context = Some(context);
        self.state = Some(state);
        Ok(())
    }

    fn unload_model(&mut self) {
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
        let language = normalize_whisper_language(whisper_params.language.as_deref())?;

        let mut full_params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
        full_params.set_no_context(true);
        full_params.set_n_threads(crate::engines::inference_threads() as i32);

        // Shrink the encoder context to the actual clip length: whisper always
        // encodes a 30 s window (1500 positions, 320 samples each), so short
        // dictation clips waste most of that work. The CoreML/ANE encoder has a
        // fixed-shape output, so this is only safe on GGML/Vulkan builds.
        // Capped at the model's own n_audio_ctx; whisper.cpp rejects anything
        // larger, and at the cap it behaves exactly like the default.
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        if let Some(max_audio_ctx) = self.context.as_ref().map(|ctx| ctx.n_audio_ctx()) {
            if max_audio_ctx > 0 {
                let audio_ctx = (samples.len() / 320 + 64).min(max_audio_ctx as usize);
                full_params.set_audio_ctx(audio_ctx as i32);
            }
        }
        full_params.set_language(language.as_deref());
        full_params.set_translate(whisper_params.translate);
        full_params.set_print_special(whisper_params.print_special);
        full_params.set_print_progress(whisper_params.print_progress);
        full_params.set_print_realtime(whisper_params.print_realtime);
        full_params.set_print_timestamps(whisper_params.print_timestamps);
        full_params.set_suppress_blank(whisper_params.suppress_blank);
        full_params.set_suppress_nst(whisper_params.suppress_non_speech_tokens);
        full_params.set_no_speech_thold(whisper_params.no_speech_thold);
        full_params.set_token_timestamps(whisper_params.word_timestamps);

        let initial_prompt = whisper_params
            .initial_prompt
            .or_else(|| build_dictionary_prompt(&whisper_params.dictionary));
        let initial_prompt = normalize_whisper_prompt(initial_prompt.as_deref())?;

        if let Some(prompt) = initial_prompt.as_deref() {
            full_params.set_initial_prompt(prompt);
        }

        state.full(full_params, &samples)?;

        let eot_token = self.context.as_ref().map(|context| context.token_eot());
        let mut words = whisper_params.word_timestamps.then(Vec::new);
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

            if let (Some(words), Some(eot_token)) = (words.as_mut(), eot_token) {
                append_segment_words(&segment, eot_token, words);
            }
            full_text.push_str(&text);
            segments.push(TranscriptionSegment { start, end, text });
        }

        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            segments: Some(segments),
            words: words.filter(|words| !words.is_empty()),
            language: whisper_rs::get_lang_str(state.full_lang_id_from_state()).map(str::to_string),
        })
    }
}

fn append_segment_words(
    segment: &whisper_rs::WhisperSegment<'_>,
    eot_token: whisper_rs::WhisperTokenId,
    words: &mut Vec<TranscriptionSegment>,
) {
    for index in 0..segment.n_tokens() {
        let Some(token) = segment.get_token(index) else {
            continue;
        };
        if token.token_id() >= eot_token {
            continue;
        }
        let Ok(piece) = token.to_str_lossy() else {
            continue;
        };
        if piece.trim().is_empty() {
            continue;
        }
        let data = token.token_data();
        let end = data.t1 as f32 / 100.0;
        match words.last_mut() {
            Some(last) if !piece.starts_with([' ', '\n']) => {
                last.text.push_str(&piece);
                last.end = end;
            }
            _ => words.push(TranscriptionSegment {
                start: data.t0 as f32 / 100.0,
                end,
                text: piece.trim_start().to_string(),
            }),
        }
    }
}

fn normalize_whisper_language(
    language: Option<&str>,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    normalize_optional_whisper_text(language, "language", true)
}

fn normalize_whisper_prompt(
    prompt: Option<&str>,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    normalize_optional_whisper_text(prompt, "initial prompt", false)
}

fn normalize_optional_whisper_text(
    value: Option<&str>,
    field: &str,
    treat_auto_as_none: bool,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let Some(value) = value else {
        return Ok(None);
    };

    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    if treat_auto_as_none && trimmed.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }

    if trimmed.contains('\0') {
        return Err(io_error(format!("whisper {field} contains a null byte")));
    }

    Ok(Some(trimmed.to_string()))
}

fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}
