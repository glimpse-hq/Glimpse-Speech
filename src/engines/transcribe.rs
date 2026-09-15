//! transcribe.cpp engine: GGUF models run through the `transcribe-cpp` crate.
//!
//! The compute backend is chosen by the crate's build features (Metal on
//! Apple platforms, Vulkan on Windows and Linux) and `Backend::Auto` picks
//! the best one present, with CPU as the fallback. On Apple Silicon a
//! compiled Core ML encoder companion (`<gguf stem>-encoder.mlmodelc`) next to
//! the GGUF moves the audio encoder to the Neural Engine.

use std::path::{Path, PathBuf};

use transcribe_cpp::{
    Backend, Error, Model, ModelOptions, Qwen3AsrRunOptions, RunExtension, RunOptions, Session,
    SessionOptions, TimestampKind, Transcript,
};

use crate::{
    TranscriptionEngine, TranscriptionResult, TranscriptionSegment,
    dictionary::build_dictionary_prompt, engines::io_error,
};

#[derive(Debug, Clone, Default)]
pub struct TranscribeModelParams {
    /// Compute backend; `Auto` takes the best one this build has.
    pub backend: Backend,
    /// Compiled Core ML encoder companion for the GGUF. `None` keeps the
    /// ggml encoder.
    pub coreml_encoder: Option<PathBuf>,
}

#[derive(Debug, Clone, Default)]
pub struct TranscribeInferenceParams {
    pub language: Option<String>,
    /// Vocabulary hints for Qwen3-ASR. Other GGUF families ignore these entries.
    pub dictionary: Vec<String>,
    pub timestamps: bool,
}

#[derive(Default)]
pub struct TranscribeEngine {
    session: Option<Session>,
    chunk_samples: Option<usize>,
    supports_context: bool,
}

impl TranscribeEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// The Core ML companion the catalog unpacks next to a GGUF, when it is
    /// present and complete. Only meaningful on Apple Silicon.
    pub fn companion_for(model_path: &Path) -> Option<PathBuf> {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return None;
        }
        let stem = model_path.file_stem()?.to_str()?;
        [Some(stem), stem.strip_suffix("-decoder")]
            .into_iter()
            .flatten()
            .map(|stem| model_path.with_file_name(format!("{stem}-encoder.mlmodelc")))
            .find(|dir| dir.join("coremldata.bin").is_file())
    }
}

impl TranscriptionEngine for TranscribeEngine {
    type InferenceParams = TranscribeInferenceParams;
    type ModelParams = TranscribeModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !model_path.is_file() {
            return Err(io_error(format!(
                "transcribe.cpp model file not found: {}",
                model_path.display()
            )));
        }
        crate::silence_transcribe_cpp_logs();
        let model = Model::load_with(
            model_path,
            &ModelOptions {
                backend: params.backend,
                device: None,
            },
        )
        .map_err(transcribe_error)?;
        let uses_coreml = params.coreml_encoder.is_some();
        let session = model
            .session_with(&SessionOptions {
                n_threads: crate::engines::inference_threads() as i32,
                coreml_encoder_path: params.coreml_encoder,
                ..Default::default()
            })
            .map_err(transcribe_error)?;
        tracing::info!(
            "[transcribe.cpp] loaded {} ({}) on {}",
            model.variant(),
            model.arch(),
            model.backend()
        );
        // These companions hold 15 seconds of audio. Qwen also needs this
        // limit without Core ML because of its native generation budget.
        self.chunk_samples = (model.arch() == "qwen3_asr"
            || (model.arch() == "parakeet" && uses_coreml))
            .then_some(15 * SAMPLE_RATE);
        self.session = Some(session);
        self.supports_context = model.arch() == "qwen3_asr";
        Ok(())
    }

    fn unload_model(&mut self) {
        self.session = None;
        self.chunk_samples = None;
        self.supports_context = false;
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| io_error("Model not loaded. Call load_model() first."))?;
        let params = params.unwrap_or_default();
        let context = self
            .supports_context
            .then(|| build_dictionary_prompt(&params.dictionary))
            .flatten();
        let options = RunOptions {
            family: context.map(|context| {
                RunExtension::Qwen3Asr(Qwen3AsrRunOptions {
                    context: Some(context),
                })
            }),
            language: params
                .language
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            timestamps: if params.timestamps {
                TimestampKind::Auto
            } else {
                TimestampKind::None
            },
            ..Default::default()
        };
        let chunks = decode_chunks(&samples, self.chunk_samples, |chunk| {
            session.run(chunk, &options)
        })
        .map_err(transcribe_error)?;
        let mut text = String::new();
        let mut language = params.language;
        let mut segments = Vec::new();
        let mut words = Vec::new();
        let mut chunks = chunks.into_iter().peekable();
        while let Some((offset, transcript)) = chunks.next() {
            let chunk_end = chunks.peek().map_or(samples.len(), |(next, _)| *next);
            let chunk_seconds = (chunk_end - offset) as f32 / SAMPLE_RATE as f32;
            let chunk_text = transcript.text.trim();
            if !chunk_text.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(chunk_text);
            }
            if language.is_none() {
                language = transcript.language;
            }
            let to_segment = |t0_ms: i64, t1_ms: i64, text: &str| TranscriptionSegment {
                // TDT duration predictions can extend beyond the final audio frame.
                start: offset as f32 / SAMPLE_RATE as f32
                    + (t0_ms as f32 / 1000.0).clamp(0.0, chunk_seconds),
                end: offset as f32 / SAMPLE_RATE as f32
                    + (t1_ms.max(t0_ms) as f32 / 1000.0).clamp(0.0, chunk_seconds),
                text: text.trim().to_string(),
            };
            segments.extend(
                transcript
                    .segments
                    .iter()
                    .map(|s| to_segment(s.t0_ms, s.t1_ms, &s.text))
                    .filter(|s| !s.text.is_empty()),
            );
            words.extend(
                transcript
                    .words
                    .iter()
                    .map(|w| to_segment(w.t0_ms, w.t1_ms, &w.text))
                    .filter(|w| !w.text.is_empty()),
            );
        }
        Ok(TranscriptionResult {
            text,
            segments: (params.timestamps && !segments.is_empty()).then_some(segments),
            words: (params.timestamps && !words.is_empty()).then_some(words),
            language,
        })
    }
}

const SAMPLE_RATE: usize = 16_000;

// Match Qwen's reference splitter: use a quiet boundary and preserve every
// sample exactly once. Search only before the limit so ANE capacity is respected.
// https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py
fn quiet_boundary(samples: &[f32], limit: usize) -> usize {
    let start = limit.saturating_sub(3 * SAMPLE_RATE).max(limit / 2);
    let window = (SAMPLE_RATE / 10).min(limit - start);
    let mut energy: f64 = samples[start..start + window]
        .iter()
        .map(|v| v.abs() as f64)
        .sum();
    let mut best = (energy, start);
    for end in start + window..limit {
        energy += samples[end].abs() as f64 - samples[end - window].abs() as f64;
        if energy <= best.0 {
            best = (energy, end + 1 - window);
        }
    }
    best.1 + window / 2
}

fn decode_chunks(
    samples: &[f32],
    max_samples: Option<usize>,
    mut decode: impl FnMut(&[f32]) -> Result<Transcript, Error>,
) -> Result<Vec<(usize, Transcript)>, Error> {
    let mut results = Vec::new();
    let mut pending = Vec::new();
    pending.push(0..samples.len());
    while let Some(range) = pending.pop() {
        let chunk = &samples[range.clone()];
        if let Some(limit) = max_samples.filter(|limit| chunk.len() > *limit) {
            let cut = range.start + quiet_boundary(chunk, limit);
            pending.push(cut..range.end);
            pending.push(range.start..cut);
            continue;
        }
        match decode(chunk) {
            Ok(transcript) => results.push((range.start, transcript)),
            Err(Error::InputTooLong(_) | Error::OutputTruncated { .. })
                if chunk.len() > SAMPLE_RATE =>
            {
                // Discard the partial output and re-decode both halves. Never
                // return a successful but incomplete transcript, or retry an
                // unrelated model/backend error. The one-second floor bounds retries.
                let cut = range.start + quiet_boundary(chunk, chunk.len() / 2);
                pending.push(cut..range.end);
                pending.push(range.start..cut);
            }
            Err(error) => return Err(error),
        }
    }
    Ok(results)
}

fn transcribe_error(error: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    io_error(format!("transcribe.cpp error: {error}"))
}

#[cfg(test)]
mod tests {
    use super::TranscribeEngine;

    #[test]
    fn splits_without_dropping_or_repeating_audio() {
        let samples: Vec<_> = (0..16_000 * 47).map(|i| i as f32).collect();
        let mut consumed = Vec::new();
        let chunks = super::decode_chunks(&samples, Some(16_000 * 15), |chunk| {
            assert!(chunk.len() <= 16_000 * 15);
            consumed.extend_from_slice(chunk);
            Ok(Default::default())
        })
        .unwrap();
        assert_eq!(consumed, samples);
        assert_eq!(chunks[0].0, 0);
        assert!(chunks.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }

    #[test]
    fn retries_length_errors_but_never_returns_partial_output() {
        let samples = vec![0.0; 16_000 * 8];
        let mut consumed = 0;
        let chunks = super::decode_chunks(&samples, None, |chunk| {
            if chunk.len() > 16_000 * 4 {
                return Err(transcribe_cpp::Error::InputTooLong("test".into()));
            }
            if chunk.len() > 16_000 * 2 {
                return Err(transcribe_cpp::Error::OutputTruncated {
                    message: "test".into(),
                    partial: Some(Box::new(transcribe_cpp::Transcript {
                        text: "partial must be discarded".into(),
                        ..Default::default()
                    })),
                });
            }
            consumed += chunk.len();
            Ok(transcribe_cpp::Transcript {
                text: "complete".into(),
                ..Default::default()
            })
        })
        .unwrap();
        assert_eq!(consumed, samples.len());
        assert!(chunks.iter().all(|(_, t)| t.text == "complete"));
    }

    #[test]
    fn errors_stop_without_unbounded_retries() {
        let mut attempts = 0;
        let result = super::decode_chunks(&vec![0.0; 16_000 * 4], None, |_| {
            attempts += 1;
            Err(transcribe_cpp::Error::InputTooLong("test".into()))
        });
        assert!(result.is_err());
        assert!(attempts <= 4);
        attempts = 0;
        let result = super::decode_chunks(&vec![0.0; 16_000 * 4], None, |_| {
            attempts += 1;
            Err(transcribe_cpp::Error::Backend("test".into()))
        });
        assert!(result.is_err());
        assert_eq!(attempts, 1);
    }

    #[test]
    fn companion_requires_complete_directory() {
        let dir =
            std::env::temp_dir().join(format!("glimpse-speech-companion-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = dir.join("Model-Q8_0.gguf");
        std::fs::write(&gguf, b"x").unwrap();
        assert_eq!(TranscribeEngine::companion_for(&gguf), None);

        let companion = dir.join("Model-Q8_0-encoder.mlmodelc");
        std::fs::create_dir_all(&companion).unwrap();
        assert_eq!(TranscribeEngine::companion_for(&gguf), None);

        std::fs::write(companion.join("coremldata.bin"), b"x").unwrap();
        let expected = cfg!(all(target_os = "macos", target_arch = "aarch64")).then_some(companion);
        assert_eq!(TranscribeEngine::companion_for(&gguf), expected);
        assert_eq!(
            TranscribeEngine::companion_for(&dir.join("Model-Q8_0-decoder.gguf")),
            expected
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
