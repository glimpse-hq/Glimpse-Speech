use std::path::{Path, PathBuf};

use parakeet_rs::{Parakeet, ParakeetTDT, TimestampMode, Transcriber};

use crate::{
    dictionary::sanitize_dictionary_entries, itn::apply_simple_english_itn, TranscriptionEngine,
    TranscriptionResult, TranscriptionSegment,
};

const REQUIRED_TDT_FP32_FILES: [&str; 4] = [
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "decoder_joint-model.onnx",
    "vocab.txt",
];

const REQUIRED_TDT_INT8_FILES: [&str; 3] = [
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "vocab.txt",
];

const REQUIRED_CTC_FILES: [&str; 3] = ["model.onnx", "model.onnx_data", "tokenizer.json"];

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParakeetArchitecture {
    #[default]
    Tdt,
    Ctc,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParakeetModelParams {
    pub architecture: ParakeetArchitecture,
    pub quantization: QuantizationType,
}

impl ParakeetModelParams {
    pub fn tdt_fp32() -> Self {
        Self {
            architecture: ParakeetArchitecture::Tdt,
            quantization: QuantizationType::FP32,
        }
    }

    pub fn tdt_int8() -> Self {
        Self {
            architecture: ParakeetArchitecture::Tdt,
            quantization: QuantizationType::Int8,
        }
    }

    pub fn ctc() -> Self {
        Self {
            architecture: ParakeetArchitecture::Ctc,
            quantization: QuantizationType::FP32,
        }
    }

    pub fn fp32() -> Self {
        Self::tdt_fp32()
    }

    pub fn int8() -> Self {
        Self::tdt_int8()
    }

    pub fn quantized(quantization: QuantizationType) -> Self {
        Self {
            architecture: ParakeetArchitecture::Tdt,
            quantization,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParakeetInferenceParams {
    pub timestamp_granularity: TimestampGranularity,
    pub language: Option<String>,
    pub dictionary: Vec<String>,
    pub enable_itn: bool,
}

impl Default for ParakeetInferenceParams {
    fn default() -> Self {
        Self {
            timestamp_granularity: TimestampGranularity::Token,
            language: None,
            dictionary: Vec::new(),
            enable_itn: false,
        }
    }
}

pub struct ParakeetEngine {
    loaded_model_path: Option<PathBuf>,
    runtime: Option<ParakeetRuntime>,
}

enum ParakeetRuntime {
    Tdt(ParakeetTDT),
    Ctc(Parakeet),
}

impl ParakeetRuntime {
    fn architecture(&self) -> ParakeetArchitecture {
        match self {
            Self::Tdt(_) => ParakeetArchitecture::Tdt,
            Self::Ctc(_) => ParakeetArchitecture::Ctc,
        }
    }
}

impl Default for ParakeetEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ParakeetEngine {
    pub fn new() -> Self {
        Self {
            loaded_model_path: None,
            runtime: None,
        }
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
        let mode = map_timestamp_mode(params.timestamp_granularity.clone(), runtime.architecture());

        // Current parakeet-rs TDT path does not expose explicit language forcing
        // or dictionary boosting, so these are currently no-ops.
        let _ = (&params.language, &params.dictionary);

        let raw_result = match runtime {
            ParakeetRuntime::Tdt(model) => model
                .transcribe_samples(samples, SAMPLE_RATE, 1, Some(mode))
                .map_err(parakeet_error)?,
            ParakeetRuntime::Ctc(model) => model
                .transcribe_samples(samples, SAMPLE_RATE, 1, Some(mode))
                .map_err(parakeet_error)?,
        };

        let mut result = map_result(raw_result, params.timestamp_granularity);
        result.text = apply_itn_if_enabled(&result.text, params.enable_itn);
        Ok(result)
    }

    fn transcribe_file_inner(
        &mut self,
        wav_path: &Path,
        params: Option<ParakeetInferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let runtime = self.runtime_mut()?;
        let params = normalize_inference_params(params);
        let mode = map_timestamp_mode(params.timestamp_granularity.clone(), runtime.architecture());

        // Current parakeet-rs TDT path does not expose explicit language forcing
        // or dictionary boosting, so these are currently no-ops.
        let _ = (&params.language, &params.dictionary);

        let raw_result = match runtime {
            ParakeetRuntime::Tdt(model) => model
                .transcribe_file(wav_path, Some(mode))
                .map_err(parakeet_error)?,
            ParakeetRuntime::Ctc(model) => model
                .transcribe_file(wav_path, Some(mode))
                .map_err(parakeet_error)?,
        };

        let mut result = map_result(raw_result, params.timestamp_granularity);
        result.text = apply_itn_if_enabled(&result.text, params.enable_itn);
        Ok(result)
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
        validate_model_path(model_path, params)?;
        let runtime = match params.architecture {
            ParakeetArchitecture::Tdt => ParakeetRuntime::Tdt(
                ParakeetTDT::from_pretrained(model_path, None).map_err(parakeet_error)?,
            ),
            ParakeetArchitecture::Ctc => ParakeetRuntime::Ctc(
                Parakeet::from_pretrained(model_path, None).map_err(parakeet_error)?,
            ),
        };
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
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        self.transcribe_inner(samples, params)
    }

    fn transcribe_file(
        &mut self,
        wav_path: &Path,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        self.transcribe_file_inner(wav_path, params)
    }
}

fn map_result(
    raw_result: parakeet_rs::TranscriptionResult,
    timestamp_granularity: TimestampGranularity,
) -> TranscriptionResult {
    let parakeet_rs::TranscriptionResult { text, tokens } = raw_result;

    let segments = match timestamp_granularity {
        TimestampGranularity::Token => None,
        TimestampGranularity::Word | TimestampGranularity::Segment => {
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

            if mapped.is_empty() {
                None
            } else {
                Some(mapped)
            }
        }
    };

    TranscriptionResult {
        text: text.trim().to_string(),
        segments,
    }
}

fn map_timestamp_mode(
    granularity: TimestampGranularity,
    architecture: ParakeetArchitecture,
) -> TimestampMode {
    match (architecture, granularity) {
        (_, TimestampGranularity::Token) => TimestampMode::Tokens,
        (_, TimestampGranularity::Word) => TimestampMode::Words,
        (ParakeetArchitecture::Tdt, TimestampGranularity::Segment) => TimestampMode::Sentences,
        (ParakeetArchitecture::Ctc, TimestampGranularity::Segment) => TimestampMode::Words,
    }
}

fn normalize_inference_params(params: Option<ParakeetInferenceParams>) -> ParakeetInferenceParams {
    let mut params = params.unwrap_or_default();
    params.dictionary = sanitize_dictionary_entries(&params.dictionary);
    params
}

fn apply_itn_if_enabled(text: &str, enabled: bool) -> String {
    if enabled {
        apply_simple_english_itn(text)
    } else {
        text.trim().to_string()
    }
}

fn validate_model_path(
    model_path: &Path,
    params: ParakeetModelParams,
) -> Result<(), Box<dyn std::error::Error>> {
    if !model_path.exists() {
        return Err(io_error(format!(
            "Parakeet model directory not found: {}",
            model_path.display()
        )));
    }

    if !model_path.is_dir() {
        return Err(io_error(format!(
            "Parakeet model path must be a directory: {}",
            model_path.display()
        )));
    }

    let required_files: &[&str] = match (params.architecture, params.quantization) {
        (ParakeetArchitecture::Tdt, QuantizationType::FP32) => &REQUIRED_TDT_FP32_FILES,
        (ParakeetArchitecture::Tdt, QuantizationType::Int8) => &REQUIRED_TDT_INT8_FILES,
        (ParakeetArchitecture::Ctc, QuantizationType::FP32) => &REQUIRED_CTC_FILES,
        (ParakeetArchitecture::Ctc, QuantizationType::Int8) => {
            return Err(io_error(
                "Parakeet CTC Int8 is not supported in this build. Use CTC FP32.",
            ));
        }
    };

    let missing: Vec<String> = required_files
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
            "Missing Parakeet model files in {}: {}",
            model_path.display(),
            missing.join(", ")
        )));
    }

    Ok(())
}

fn parakeet_error(error: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    io_error(format!("parakeet-rs error: {error}"))
}

fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}

#[cfg(test)]
mod tests {
    use super::{ParakeetArchitecture, ParakeetModelParams, QuantizationType};

    #[test]
    fn int8_constructor_sets_quantized_mode() {
        let params = ParakeetModelParams::int8();
        assert_eq!(params.architecture, ParakeetArchitecture::Tdt);
        assert_eq!(params.quantization, QuantizationType::Int8);
    }

    #[test]
    fn fp32_constructor_sets_full_precision_mode() {
        let params = ParakeetModelParams::fp32();
        assert_eq!(params.architecture, ParakeetArchitecture::Tdt);
        assert_eq!(params.quantization, QuantizationType::FP32);
    }

    #[test]
    fn ctc_constructor_sets_ctc_mode() {
        let params = ParakeetModelParams::ctc();
        assert_eq!(params.architecture, ParakeetArchitecture::Ctc);
        assert_eq!(params.quantization, QuantizationType::FP32);
    }
}
