//! transcribe.cpp engine: GGUF models and whisper.cpp GGML `.bin` files run
//! through the `transcribe-cpp` crate.
//!
//! The compute backend is chosen by the crate's build features (Metal on
//! Apple platforms, Vulkan on Windows and Linux) and `Backend::Auto` picks
//! the best one present, with CPU as the fallback. On Apple Silicon a
//! compiled Core ML encoder companion next to the model moves the audio
//! encoder to the Neural Engine: `<gguf stem>-encoder.mlmodelc` for a GGUF,
//! `whisper-<family>-encoder.mlmodelc` for a Whisper GGUF or
//! `ggml-<family>[-qX_Y].bin`.

use std::path::{Path, PathBuf};

use transcribe_cpp::{
    Backend, Error, ExtSlot, Model, ModelOptions, OwnedStream, ParakeetBufferedStreamOptions,
    ParakeetStreamOptions, Qwen3AsrRunOptions, RunExtension, RunOptions, Session, SessionOptions,
    StreamExtension, StreamOptions, TimestampKind, Transcript, WhisperRunOptions,
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
    /// Vocabulary hints for Qwen3-ASR and Whisper. Other families ignore these entries.
    pub dictionary: Vec<String>,
    /// Whisper decoder prompt, placed before the dictionary hints.
    pub prompt: Option<String>,
    pub timestamps: bool,
    /// Whisper computes word timings in an extra alignment pass, so only
    /// request them when they are used.
    pub word_timestamps: bool,
}

#[derive(Debug, Clone, Copy, Default)]
enum Family {
    #[default]
    Other,
    Qwen3Asr,
    Whisper {
        words: bool,
        multilingual: bool,
    },
}

#[derive(Default)]
pub struct TranscribeEngine {
    // Moves into `stream` while a stream is active.
    session: Option<Session>,
    stream: Option<OwnedStream>,
    stream_options: Option<StreamExtension>,
    stream_language: Option<String>,
    stream_text: String,
    // Language tags a Parakeet-family model accepts; empty for other families.
    languages: Vec<String>,
    chunk_samples: Option<usize>,
    family: Family,
}

impl TranscribeEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// The Core ML companion the catalog unpacks next to a model, when it is
    /// present and complete. Only meaningful on Apple Silicon.
    pub fn companion_for(model_path: &Path) -> Option<PathBuf> {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return None;
        }
        let stem = model_path.file_stem()?.to_str()?;
        // One Whisper encoder serves every quantization of a family.
        let candidates = if model_path.extension().is_some_and(|ext| ext == "bin") {
            let family = strip_quant_suffix(stem.strip_prefix("ggml-")?);
            vec![format!("whisper-{family}")]
        } else if stem.starts_with("whisper-") {
            vec![strip_quant_suffix(stem).to_string(), stem.to_string()]
        } else {
            [Some(stem), stem.strip_suffix("-decoder")]
                .into_iter()
                .flatten()
                .map(str::to_string)
                .collect()
        };
        candidates
            .into_iter()
            .map(|stem| model_path.with_file_name(format!("{stem}-encoder.mlmodelc")))
            .find(|dir| dir.join("coremldata.bin").is_file())
    }

    /// Feeds 16 kHz audio to the stream, starting one if needed, and returns
    /// the transcript so far. Only streaming models (Nemotron, Parakeet Unified).
    pub fn transcribe_chunk(
        &mut self,
        samples: &[f32],
    ) -> Result<String, Box<dyn std::error::Error>> {
        if self.stream.is_none() {
            let options = self
                .stream_options
                .clone()
                .ok_or_else(|| io_error("This model does not support streaming"))?;
            let run = RunOptions {
                language: self.language_hint(self.stream_language.as_deref()),
                timestamps: TimestampKind::None,
                ..Default::default()
            };
            let session = self
                .session
                .take()
                .ok_or_else(|| io_error("Model not loaded. Call load_model() first."))?;
            let options = StreamOptions {
                family: Some(options),
                ..Default::default()
            };
            match session.into_stream(&run, &options) {
                Ok(stream) => self.stream = Some(stream),
                Err((error, session)) => {
                    self.session = Some(session);
                    return Err(transcribe_error(error));
                }
            }
        }
        if let Some(stream) = self.stream.as_mut() {
            stream.feed(samples).map_err(transcribe_error)?;
            self.stream_text = stream.text().display().trim().to_string();
        }
        Ok(self.stream_text.clone())
    }

    /// Flushes the audio the stream still holds and returns the final transcript.
    pub fn finalize(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        if let Some(stream) = self.stream.as_mut() {
            stream.finalize().map_err(transcribe_error)?;
            self.stream_text = stream.text().full.trim().to_string();
        }
        Ok(self.stream_text.clone())
    }

    pub fn get_transcript(&self) -> String {
        self.stream_text.clone()
    }

    /// Language for the next stream.
    pub fn configure_stream(&mut self, language: Option<String>) {
        self.stream_language = language;
    }

    /// Ends any stream and clears its transcript.
    pub fn reset(&mut self) {
        if let Some(stream) = self.stream.take() {
            self.session = Some(stream.into_session());
        }
        self.stream_text.clear();
    }

    // Parakeet-family models reject tags outside their list, so a bare
    // language maps onto the first listed tag for it (pt-BR before pt-PT) and
    // anything else to auto.
    fn language_hint(&self, language: Option<&str>) -> Option<String> {
        let language = language
            .map(str::trim)
            .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("auto"))?;
        if self.languages.is_empty() {
            return Some(language.to_string());
        }
        let primary = |tag: &str| tag.split(['-', '_']).next().unwrap_or(tag).to_string();
        self.languages
            .iter()
            .find(|tag| tag.eq_ignore_ascii_case(language))
            .or_else(|| {
                self.languages
                    .iter()
                    .find(|tag| primary(tag).eq_ignore_ascii_case(&primary(language)))
            })
            .cloned()
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
        let arch = model.arch();
        let session_with = |coreml_encoder_path: Option<PathBuf>| {
            model.session_with(&SessionOptions {
                n_threads: crate::engines::inference_threads() as i32,
                coreml_encoder_path,
                ..Default::default()
            })
        };
        let mut uses_coreml = params.coreml_encoder.is_some();
        let mut session = session_with(params.coreml_encoder.clone());
        // Whisper keeps its ggml encoder, so a rejected companion only costs speed.
        if let (Err(error @ Error::InvalidArgument(_)), Some(encoder)) =
            (&session, &params.coreml_encoder)
            && arch == "whisper"
        {
            tracing::warn!(
                "[transcribe.cpp] Core ML encoder {} rejected, using the ggml encoder: {error}",
                encoder.display()
            );
            uses_coreml = false;
            session = session_with(None);
        }
        let session = session.map_err(transcribe_error)?;
        tracing::info!(
            "[transcribe.cpp] loaded {} ({}) on {} coreml={uses_coreml}",
            model.variant(),
            arch,
            model.backend()
        );
        // These companions hold 15 seconds of audio. Qwen also needs this
        // limit without Core ML because of its native generation budget.
        self.chunk_samples = (arch == "qwen3_asr" || (arch == "parakeet" && uses_coreml))
            .then_some(15 * SAMPLE_RATE);
        self.family = match arch.as_str() {
            "qwen3_asr" => Family::Qwen3Asr,
            "whisper" => {
                let capabilities = model.capabilities();
                Family::Whisper {
                    words: matches!(
                        capabilities.max_timestamp_kind,
                        TimestampKind::Word | TimestampKind::Token
                    ),
                    multilingual: capabilities.supports_language_detect,
                }
            }
            _ => Family::Other,
        };
        self.languages = if arch == "parakeet" {
            model.capabilities().languages
        } else {
            Vec::new()
        };
        // Streaming geometry matching the previous ONNX runtime: 560 ms chunks.
        let accepts = |kind| model.accepts_ext(ExtSlot::Stream, kind);
        self.stream_options =
            if accepts(transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM) {
                Some(StreamExtension::ParakeetBuffered(
                    ParakeetBufferedStreamOptions {
                        left_ms: Some(5600),
                        chunk_ms: Some(560),
                        right_ms: Some(560),
                    },
                ))
            } else if accepts(transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_PARAKEET_STREAM) {
                Some(StreamExtension::ParakeetStream(ParakeetStreamOptions {
                    att_context_right: Some(6),
                }))
            } else {
                None
            };
        self.stream = None;
        self.session = Some(session);
        Ok(())
    }

    fn unload_model(&mut self) {
        self.stream = None;
        self.session = None;
        self.stream_options = None;
        self.stream_text.clear();
        self.languages.clear();
        self.chunk_samples = None;
        self.family = Family::Other;
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        self.reset();
        let params = params.unwrap_or_default();
        let wants_timestamps = params.timestamps || params.word_timestamps;
        let mut language = self.language_hint(params.language.as_deref());
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| io_error("Model not loaded. Call load_model() first."))?;
        let (family, timestamps) = match self.family {
            Family::Qwen3Asr => (
                build_dictionary_prompt(&params.dictionary).map(|context| {
                    RunExtension::Qwen3Asr(Qwen3AsrRunOptions {
                        context: Some(context),
                    })
                }),
                auto_timestamps(wants_timestamps),
            ),
            Family::Whisper {
                words,
                multilingual,
            } => {
                // English-only Whisper rejects any other language hint.
                if !multilingual {
                    language = None;
                }
                let timestamps = if params.word_timestamps && words {
                    TimestampKind::Word
                } else {
                    auto_timestamps(wants_timestamps)
                };
                // whisper.cpp's decoding as Glimpse shipped it: non-speech tags
                // kept for silence, no no-speech gate (whisper.cpp's never fired).
                let options = WhisperRunOptions {
                    initial_prompt: whisper_prompt(params.prompt, &params.dictionary),
                    condition_on_prev_tokens: Some(true),
                    no_speech_thold: Some(f32::INFINITY),
                    suppress_non_speech: Some(false),
                    best_of: Some(5),
                    entropy_thold: Some(2.4),
                    greedy_prompt_tokens: Some(true),
                    ..Default::default()
                };
                (Some(RunExtension::Whisper(options)), timestamps)
            }
            Family::Other => (None, auto_timestamps(wants_timestamps)),
        };
        let options = RunOptions {
            family,
            language: language.clone(),
            timestamps,
            ..Default::default()
        };
        if let Family::Whisper {
            multilingual: false,
            ..
        } = self.family
        {
            language = Some("en".to_string());
        }
        let chunks = decode_chunks(&samples, self.chunk_samples, |chunk| {
            session.run(chunk, &options)
        })
        .map_err(transcribe_error)?;
        let mut text = String::new();
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
            segments: (wants_timestamps && !segments.is_empty()).then_some(segments),
            words: (wants_timestamps && !words.is_empty()).then_some(words),
            language,
        })
    }
}

const SAMPLE_RATE: usize = 16_000;

fn auto_timestamps(wanted: bool) -> TimestampKind {
    if wanted {
        TimestampKind::Auto
    } else {
        TimestampKind::None
    }
}

fn whisper_prompt(prompt: Option<String>, dictionary: &[String]) -> Option<String> {
    let prompt = prompt.filter(|prompt| !prompt.trim().is_empty());
    match (prompt, build_dictionary_prompt(dictionary)) {
        (Some(prompt), Some(dictionary)) => Some(format!("{prompt}\n\n{dictionary}")),
        (prompt, dictionary) => prompt.or(dictionary),
    }
}

// whisper.cpp names files `ggml-<family>[-qX_Y].bin`, GGUFs are
// `whisper-<family>-<Q8_0|Q5_K_M|F16|...>.gguf`.
fn strip_quant_suffix(stem: &str) -> &str {
    let Some((family, quant)) = stem.rsplit_once('-') else {
        return stem;
    };
    let quant = quant.as_bytes();
    if matches!(quant.first(), Some(b'q' | b'Q' | b'F'))
        && quant.get(1).is_some_and(u8::is_ascii_digit)
    {
        family
    } else {
        stem
    }
}

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
