use std::{path::Path, time::Instant};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::{
    dictionary::build_dictionary_prompt, TranscriptionEngine, TranscriptionResult,
    TranscriptionSegment,
};

#[derive(Debug, Clone)]
pub struct WhisperModelParams {
    pub use_gpu: bool,
    pub gpu_device: i32,
    pub dtw_preset: Option<whisper_rs::DtwModelPreset>,
}

impl Default for WhisperModelParams {
    fn default() -> Self {
        Self {
            use_gpu: true,
            gpu_device: 0,
            dtw_preset: None,
        }
    }
}

/// Maps a model variant (whisper family, with or without a "whisper-" prefix)
/// to the DTW alignment-heads preset for that model generation.
pub fn dtw_preset_for_variant(variant: &str) -> Option<whisper_rs::DtwModelPreset> {
    use whisper_rs::DtwModelPreset as Preset;

    let family = variant.strip_prefix("whisper-").unwrap_or(variant);
    Some(match family {
        "tiny" => Preset::Tiny,
        "tiny.en" => Preset::TinyEn,
        "base" => Preset::Base,
        "base.en" => Preset::BaseEn,
        "small" => Preset::Small,
        "small.en" => Preset::SmallEn,
        "medium" => Preset::Medium,
        "medium.en" => Preset::MediumEn,
        "large-v1" => Preset::LargeV1,
        "large-v2" => Preset::LargeV2,
        "large-v3" => Preset::LargeV3,
        "large-v3-turbo" => Preset::LargeV3Turbo,
        _ => return None,
    })
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
            // Off: suppressing these makes whisper hallucinate phrases ("Thank
            // you.") on silence instead of emitting a [BLANK_AUDIO] tag.
            suppress_non_speech_tokens: false,
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
        let total_started = Instant::now();
        let model_bytes = std::fs::metadata(model_path)
            .ok()
            .map(|metadata| metadata.len());
        let use_gpu = params.use_gpu;
        let gpu_device = params.gpu_device;
        let has_dtw = params.dtw_preset.is_some();
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| io_error("model path is not valid UTF-8"))?;

        // whisper.cpp disables DTW when flash attention is enabled. Prefer DTW
        // because Glimpse uses it for accurate word timestamps.
        let flash_attn = !has_dtw;
        let mut context_params = WhisperContextParameters {
            use_gpu,
            flash_attn,
            gpu_device,
            ..WhisperContextParameters::default()
        };
        // DTW cross-attention alignment gives accurate word timestamps; the
        // energy heuristic used otherwise drifts.
        if let Some(model_preset) = params.dtw_preset {
            context_params.dtw_parameters = whisper_rs::DtwParameters {
                mode: whisper_rs::DtwMode::ModelPreset { model_preset },
                ..whisper_rs::DtwParameters::default()
            };
        }
        tracing::info!(
            "[WhisperEngine] load start path={} bytes={:?} use_gpu={} gpu_device={} flash_attn={} dtw={}",
            model_path.display(),
            model_bytes,
            use_gpu,
            gpu_device,
            flash_attn,
            has_dtw
        );
        log_coreml_lines("before_load");
        let context_started = Instant::now();
        let context = WhisperContext::new_with_params(model_path_str, context_params)?;
        let context_elapsed = context_started.elapsed();
        let state_started = Instant::now();
        let state = match context.create_state() {
            Ok(state) => state,
            Err(error) => {
                log_coreml_lines("create_state_error");
                return Err(error.into());
            }
        };
        let state_elapsed = state_started.elapsed();
        log_coreml_lines("after_create_state");

        self.context = Some(context);
        self.state = Some(state);
        tracing::info!(
            "[WhisperEngine] load complete total={:.2}s context={:.2}s create_state={:.2}s",
            total_started.elapsed().as_secs_f32(),
            context_elapsed.as_secs_f32(),
            state_elapsed.as_secs_f32()
        );
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

fn log_coreml_lines(phase: &str) {
    for line in crate::take_coreml_log() {
        tracing::info!("[WhisperEngine] coreml phase={} {}", phase, line);
    }
}

fn append_segment_words(
    segment: &whisper_rs::WhisperSegment<'_>,
    eot_token: whisper_rs::WhisperTokenId,
    words: &mut Vec<TranscriptionSegment>,
) {
    let mut tokens: Vec<(String, whisper_rs::WhisperTokenData)> = Vec::new();
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
        tokens.push((piece.into_owned(), token.token_data()));
    }

    // With DTW, t_dtw is the aligned onset of each token, so a token ends
    // where the next one begins. Without it, fall back to the t0/t1 heuristic.
    let use_dtw = tokens.iter().all(|(_, data)| data.t_dtw >= 0) && !tokens.is_empty();
    for index in 0..tokens.len() {
        let (piece, data) = &tokens[index];
        let (start_cs, end_cs) = if use_dtw {
            let end = tokens
                .get(index + 1)
                .map(|(_, next)| next.t_dtw)
                .unwrap_or(data.t1.max(data.t_dtw));
            (data.t_dtw, end.max(data.t_dtw))
        } else {
            (data.t0, data.t1.max(data.t0))
        };
        let start = start_cs as f32 / 100.0;
        let end = end_cs as f32 / 100.0;
        match words.last_mut() {
            Some(last) if !piece.starts_with([' ', '\n']) => {
                last.text.push_str(piece);
                last.end = end;
            }
            _ => words.push(TranscriptionSegment {
                start,
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
