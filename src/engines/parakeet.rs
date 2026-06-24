use std::path::Path;

use parakeet_rs::{ParakeetTDT, ParakeetUnified, TimestampMode, Transcriber};

use crate::{
    dictionary::sanitize_dictionary_entries,
    engines::{io_error, validate_model_dir},
    models::ModelLayout,
    TranscriptionEngine, TranscriptionResult, TranscriptionSegment,
};

const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Default, PartialEq)]
pub enum TimestampGranularity {
    #[default]
    Token,
    Word,
    Segment,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QuantizationType {
    #[default]
    FP32,
    Int8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParakeetModelParams {
    pub layout: ModelLayout,
    pub quantization: QuantizationType,
}

impl Default for ParakeetModelParams {
    fn default() -> Self {
        Self::fp32()
    }
}

impl ParakeetModelParams {
    pub fn fp32() -> Self {
        Self {
            layout: ModelLayout::ParakeetTdt,
            quantization: QuantizationType::FP32,
        }
    }

    pub fn int8() -> Self {
        Self {
            layout: ModelLayout::ParakeetTdt,
            quantization: QuantizationType::Int8,
        }
    }

    pub fn int8_with_layout(layout: ModelLayout) -> Self {
        Self {
            layout,
            quantization: QuantizationType::Int8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParakeetInferenceParams {
    pub timestamp_granularity: TimestampGranularity,
    pub language: Option<String>,
    pub dictionary: Vec<String>,
}

impl Default for ParakeetInferenceParams {
    fn default() -> Self {
        Self {
            timestamp_granularity: TimestampGranularity::Token,
            language: None,
            dictionary: Vec::new(),
        }
    }
}

#[derive(Default)]
pub struct ParakeetEngine {
    runtime: Option<ParakeetRuntime>,
}

enum ParakeetRuntime {
    Tdt(ParakeetTDT),
    Unified(ParakeetUnified),
}

impl ParakeetEngine {
    pub fn new() -> Self {
        Self::default()
    }

    fn runtime_mut(&mut self) -> Result<&mut ParakeetRuntime, Box<dyn std::error::Error>> {
        self.runtime
            .as_mut()
            .ok_or_else(|| io_error("Model not loaded. Call load_model() first."))
    }

    fn transcribe_inner(
        &mut self,
        samples: Vec<f32>,
        params: Option<ParakeetInferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let runtime = self.runtime_mut()?;
        let params = normalize_inference_params(params);
        let mode = map_timestamp_mode(params.timestamp_granularity.clone());

        // Current parakeet-rs TDT path does not expose explicit language forcing
        // or dictionary boosting, so these are currently no-ops.
        let _ = (&params.language, &params.dictionary);

        let raw_result = match runtime {
            ParakeetRuntime::Tdt(runtime) => runtime
                .transcribe_samples(samples, SAMPLE_RATE, 1, Some(mode))
                .map_err(parakeet_error)?,
            ParakeetRuntime::Unified(runtime) => runtime
                .transcribe_samples(samples, SAMPLE_RATE, 1, Some(mode))
                .map_err(parakeet_error)?,
        };

        Ok(map_result(raw_result, params.timestamp_granularity))
    }

    pub fn transcribe_chunk(
        &mut self,
        samples: &[f32],
    ) -> Result<String, Box<dyn std::error::Error>> {
        match self.runtime_mut()? {
            ParakeetRuntime::Unified(runtime) => {
                runtime.transcribe_chunk(samples).map_err(parakeet_error)
            }
            ParakeetRuntime::Tdt(_) => Err(io_error(
                "Streaming is only supported with unified Parakeet models",
            )),
        }
    }

    pub fn get_transcript(&self) -> String {
        match self.runtime.as_ref() {
            Some(ParakeetRuntime::Unified(runtime)) => runtime.get_transcript(),
            _ => String::new(),
        }
    }

    pub fn reset(&mut self) {
        if let Some(ParakeetRuntime::Unified(runtime)) = self.runtime.as_mut() {
            runtime.reset();
        }
    }
}

impl Drop for ParakeetEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for ParakeetEngine {
    type InferenceParams = ParakeetInferenceParams;
    type ModelParams = ParakeetModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        validate_model_dir(model_path, "Parakeet")?;
        let exec_config = parakeet_rs::ExecutionConfig::default()
            .with_intra_threads(crate::engines::inference_threads());
        let runtime = match params.layout {
            ModelLayout::ParakeetUnified => ParakeetRuntime::Unified(
                ParakeetUnified::from_pretrained(model_path, Some(exec_config))
                    .map_err(parakeet_error)?,
            ),
            _ => ParakeetRuntime::Tdt(
                ParakeetTDT::from_pretrained(model_path, Some(exec_config))
                    .map_err(parakeet_error)?,
            ),
        };
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
        self.transcribe_inner(samples, params)
    }
}

fn map_result(
    raw_result: parakeet_rs::TranscriptionResult,
    timestamp_granularity: TimestampGranularity,
) -> TranscriptionResult {
    let parakeet_rs::TranscriptionResult { text, tokens } = raw_result;

    let mapped: Vec<TranscriptionSegment> = tokens
        .into_iter()
        .filter_map(|token| {
            let text = token.text.trim().to_string();
            if text.is_empty() {
                None
            } else {
                Some(TranscriptionSegment {
                    start: token.start,
                    end: token.end,
                    text,
                })
            }
        })
        .collect();

    let (segments, words) = match timestamp_granularity {
        TimestampGranularity::Token => (None, None),
        TimestampGranularity::Segment => ((!mapped.is_empty()).then_some(mapped), None),
        TimestampGranularity::Word => {
            let words = attach_punctuation(mapped);
            let segments = group_words_into_sentences(&words);
            (
                (!segments.is_empty()).then_some(segments),
                (!words.is_empty()).then_some(words),
            )
        }
    };

    TranscriptionResult {
        text: text.trim().to_string(),
        segments,
        words,
        language: None,
    }
}

fn attach_punctuation(words: Vec<TranscriptionSegment>) -> Vec<TranscriptionSegment> {
    let mut out: Vec<TranscriptionSegment> = Vec::new();
    for word in words {
        let punctuation_only = word.text.chars().all(|ch| !ch.is_alphanumeric());
        match out.last_mut() {
            Some(last) if punctuation_only => {
                last.text.push_str(&word.text);
                last.end = word.end;
            }
            _ => out.push(word),
        }
    }
    out
}

fn group_words_into_sentences(words: &[TranscriptionSegment]) -> Vec<TranscriptionSegment> {
    let mut sentences: Vec<TranscriptionSegment> = Vec::new();
    let mut current: Option<TranscriptionSegment> = None;
    for word in words {
        match current.as_mut() {
            Some(sentence) => {
                sentence.text.push(' ');
                sentence.text.push_str(&word.text);
                sentence.end = word.end;
            }
            None => current = Some(word.clone()),
        }
        if word.text.ends_with(['.', '!', '?', '…']) {
            sentences.extend(current.take());
        }
    }
    sentences.extend(current);
    sentences
}

fn map_timestamp_mode(granularity: TimestampGranularity) -> TimestampMode {
    match granularity {
        TimestampGranularity::Token => TimestampMode::Tokens,
        TimestampGranularity::Word => TimestampMode::Words,
        TimestampGranularity::Segment => TimestampMode::Sentences,
    }
}

fn normalize_inference_params(params: Option<ParakeetInferenceParams>) -> ParakeetInferenceParams {
    let mut params = params.unwrap_or_default();
    params.dictionary = sanitize_dictionary_entries(&params.dictionary);
    params
}

fn parakeet_error(error: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    io_error(format!("parakeet-rs error: {error}"))
}
