//! transcribe.cpp engine: GGUF models and whisper.cpp GGML `.bin` files run
//! through the `transcribe-cpp` crate.
//!
//! The compute backend is chosen by the crate's build features (Metal on
//! Apple platforms, Vulkan on Windows and Linux) and `Backend::Auto` picks
//! the best one present, with CPU as the fallback. On Apple Silicon a
//! compiled Core ML encoder companion next to the model moves the audio
//! encoder to the Neural Engine: `<gguf stem>-encoder.mlmodelc` for a GGUF,
//! `whisper-<family>-encoder.mlmodelc` for a Whisper GGUF or
//! `ggml-<family>[-qX_Y].bin`, with Distil-Whisper using its teacher's.

use std::ops::Range;
use std::path::{Path, PathBuf};

use transcribe_cpp::{
    Backend, Error, ExtSlot, Model, ModelOptions, OwnedStream, ParakeetBufferedStreamOptions,
    ParakeetRunOptions, ParakeetStreamOptions, Qwen3AsrRunOptions, RunExtension, RunOptions,
    Session, SessionOptions, StreamExtension, StreamOptions, TimestampKind, Transcript,
    WhisperRunOptions,
};

use crate::{
    TranscriptionEngine, TranscriptionResult, TranscriptionSegment,
    dictionary::{build_dictionary_prompt, sanitize_dictionary_entries},
    engines::io_error,
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
    /// Vocabulary hints for Qwen3-ASR and Whisper, boosted phrases for Parakeet
    /// and Nemotron. Other families ignore these entries.
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
    /// Parakeet-family transducer that accepts phrase boosting.
    Parakeet,
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
    stream_dictionary: Vec<String>,
    stream_text: String,
    // Language tags a Parakeet-family model accepts; empty for other families.
    languages: Vec<String>,
    chunk_samples: Option<usize>,
    family: Family,
    // Parakeet TDT can return no words for whole spans of speech after long
    // pauses, so its decodes are checked against Silero.
    checks_dropped_speech: bool,
    // Parakeet's Core ML batch encodes the next chunk while one decodes.
    batches_chunks: bool,
}

impl TranscribeEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the app marked this encoder as still compiling for the Neural
    /// Engine, with a `.<encoder dir>.compiling` file next to it.
    pub fn is_compiling(encoder: &Path) -> bool {
        encoder.file_name().is_some_and(|name| {
            encoder
                .with_file_name(format!(".{}.compiling", name.to_string_lossy()))
                .is_file()
        })
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
            let family = distil_teacher(strip_quant_suffix(stem.strip_prefix("ggml-")?));
            vec![format!("whisper-{family}")]
        } else if stem.starts_with("whisper-") {
            vec![strip_quant_suffix(stem).to_string(), stem.to_string()]
        } else if stem.starts_with("distil-") {
            let family = distil_teacher(strip_quant_suffix(stem));
            vec![format!("whisper-{family}")]
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
            let mut run = RunOptions {
                family: match self.family {
                    Family::Parakeet => parakeet_boost(&self.stream_dictionary),
                    _ => None,
                },
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
            let mut begun = session.into_stream(&run, &options);
            // A rejected boost list must never fail dictation: stream unboosted.
            if let Err((Error::InvalidArgument(error), _)) = &begun
                && run.family.is_some()
            {
                tracing::warn!(
                    "[transcribe.cpp] phrase boosting rejected, streaming without it: {error}"
                );
                run.family = None;
                begun = match begun {
                    Err((_, session)) => session.into_stream(&run, &options),
                    ok => ok,
                };
            }
            match begun {
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

    /// Language and dictionary for the next stream.
    pub fn configure_stream(&mut self, language: Option<String>, dictionary: Vec<String>) {
        self.stream_language = language;
        self.stream_dictionary = dictionary;
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
        let on_cpu = model.backend().eq_ignore_ascii_case("cpu");
        let session_with = |coreml_encoder_path: Option<PathBuf>| {
            // Next to a GPU or Core ML encoder the CPU only runs the mel and
            // the decoder, which gained nothing past 8 threads on Windows
            // while the extra threads kept cores busy.
            let threads = if on_cpu && coreml_encoder_path.is_none() {
                crate::engines::inference_threads()
            } else {
                crate::engines::inference_threads().min(8)
            };
            model.session_with(&SessionOptions {
                n_threads: threads as i32,
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
        // Whisper gets the 28 second chunks Glimpse's Library uses: long-form
        // seeking can carry a garbage window's context into the next one.
        // A ggml Parakeet encoder attends over its whole input: 60 second
        // windows ran 2 to 3 times faster than whole recordings on Metal and
        // the CPU, with lower long-form WER, and Unified returned no text for
        // some recordings past 6 minutes.
        self.chunk_samples = match arch.as_str() {
            "whisper" => Some(28 * SAMPLE_RATE),
            "qwen3_asr" => Some(15 * SAMPLE_RATE),
            "parakeet" if uses_coreml => Some(15 * SAMPLE_RATE),
            "parakeet" => Some(60 * SAMPLE_RATE),
            _ => None,
        };
        self.family = match arch.as_str() {
            "parakeet"
                if model.accepts_ext(
                    ExtSlot::Run,
                    transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_PARAKEET_RUN,
                ) =>
            {
                Family::Parakeet
            }
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
        self.checks_dropped_speech = arch == "parakeet" && model.variant().starts_with("tdt-");
        self.batches_chunks = arch == "parakeet" && uses_coreml;
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
        self.checks_dropped_speech = false;
        self.batches_chunks = false;
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
            Family::Parakeet => (
                parakeet_boost(&params.dictionary),
                auto_timestamps(wants_timestamps),
            ),
            Family::Other => (None, auto_timestamps(wants_timestamps)),
        };
        let mut options = RunOptions {
            family,
            language: language.clone(),
            // The dropped-speech check needs words; Parakeet times come with its tokens.
            timestamps: if self.checks_dropped_speech {
                TimestampKind::Auto
            } else {
                timestamps
            },
            ..Default::default()
        };
        if let Family::Whisper {
            multilingual: false,
            ..
        } = self.family
        {
            language = Some("en".to_string());
        }
        let batch = self.batches_chunks;
        let (chunks, speech) = std::thread::scope(|scope| {
            // Silero runs while the model decodes.
            let vad = self
                .checks_dropped_speech
                .then(|| {
                    std::thread::Builder::new().spawn_scoped(scope, || speech_ranges(&samples))
                })
                .and_then(Result::ok);
            let mut chunks = decode_chunks(&samples, self.chunk_samples, |chunks| {
                run_chunks(session, chunks, &options, batch)
            });
            // A rejected boost list must never fail dictation: decode unboosted.
            if let (Err(Error::InvalidArgument(error)), Some(RunExtension::Parakeet(_))) =
                (&chunks, &options.family)
            {
                tracing::warn!(
                    "[transcribe.cpp] phrase boosting rejected, decoding without it: {error}"
                );
                options.family = None;
                chunks = decode_chunks(&samples, self.chunk_samples, |chunks| {
                    run_chunks(session, chunks, &options, batch)
                });
            }
            (chunks, vad.and_then(|vad| vad.join().ok().flatten()))
        });
        let mut result = assemble(chunks.map_err(transcribe_error)?, samples.len());
        let holes = speech.as_deref().map_or_else(Vec::new, |speech| {
            dropped_speech(result.words.as_deref().unwrap_or_default(), speech)
        });
        if let Some(speech) = speech
            && !holes.is_empty()
        {
            // Decoding the piece between long pauses on its own recovers it.
            let spans: Vec<_> = speech_between_long_pauses(&speech, samples.len())
                .into_iter()
                .filter(|span| {
                    *span != (0..samples.len()) && holes.iter().any(|hole| span.contains(hole))
                })
                .collect();
            let retries = decode_spans(&samples, &spans, self.chunk_samples, |chunks| {
                run_chunks(session, chunks, &options, batch)
            })
            .map(|decoded| {
                decoded
                    .into_iter()
                    .zip(spans)
                    .map(|(chunks, span)| (assemble(chunks, span.end), span))
                    .collect()
            });
            match retries {
                Ok(retries) => result = splice(result, retries),
                Err(error) => {
                    tracing::warn!("[transcribe.cpp] dropped-speech retry failed: {error}")
                }
            }
        }
        let non_empty = |items: Option<Vec<TranscriptionSegment>>| {
            items.filter(|items| wants_timestamps && !items.is_empty())
        };
        Ok(TranscriptionResult {
            text: result.text,
            segments: non_empty(result.segments),
            words: non_empty(result.words),
            language: language.or(result.language),
        })
    }
}

const SAMPLE_RATE: usize = 16_000;
// A stretch of detected speech at least this long without a word counts as dropped.
const DROPPED_SPEECH: usize = 3 * SAMPLE_RATE / 2;
// Gaps at least this long between padded speech regions (about a second of
// silence) split the retry.
const LONG_PAUSE: usize = SAMPLE_RATE / 2;

// Chunk transcripts keyed by input sample offset, in order, ending at `end`.
fn assemble(chunks: Vec<(usize, Transcript)>, end: usize) -> TranscriptionResult {
    let mut text = String::new();
    let mut segments = Vec::new();
    let mut words = Vec::new();
    let mut language = None;
    let mut chunks = chunks.into_iter().peekable();
    while let Some((offset, transcript)) = chunks.next() {
        let chunk_end = chunks.peek().map_or(end, |(next, _)| *next);
        let chunk_text = transcript.text.trim();
        if !chunk_text.is_empty() {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(chunk_text);
        }
        language = language.or(transcript.language);
        let chunk_seconds = (chunk_end - offset) as f32 / SAMPLE_RATE as f32;
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
    TranscriptionResult {
        text,
        segments: Some(segments),
        words: Some(words),
        language,
    }
}

// Silero's padded speech regions as merged sample ranges. The audio goes in by
// the minute so a long recording never needs a second full copy.
fn speech_ranges(samples: &[f32]) -> Option<Vec<Range<usize>>> {
    const PIECE: usize = 60 * SAMPLE_RATE;
    let mut ranges: Vec<Range<usize>> = Vec::new();
    for (index, piece) in samples.chunks(PIECE).enumerate() {
        let pcm: Vec<i16> = piece.iter().map(|&s| (s * 32_768.0) as i16).collect();
        let offset = index * PIECE;
        for (start, end) in crate::vad::speech_regions(&pcm, SAMPLE_RATE as u32)? {
            let sample = |seconds: f32| {
                (offset + (seconds * SAMPLE_RATE as f32) as usize).min(offset + piece.len())
            };
            let (start, end) = (sample(start), sample(end));
            match ranges.last_mut() {
                Some(last) if start <= last.end => last.end = last.end.max(end),
                _ if start < end => ranges.push(start..end),
                _ => {}
            }
        }
    }
    Some(ranges)
}

// Sample offsets inside speech regions where at least `DROPPED_SPEECH` of
// speech has no word.
fn dropped_speech(words: &[TranscriptionSegment], speech: &[Range<usize>]) -> Vec<usize> {
    let sample = |seconds: f32| (seconds * SAMPLE_RATE as f32) as usize;
    let mut holes = Vec::new();
    for range in speech {
        let mut covered = range.start;
        for word in words {
            let (start, end) = (sample(word.start), sample(word.end));
            if end <= range.start || start >= range.end {
                continue;
            }
            if start.saturating_sub(covered) >= DROPPED_SPEECH {
                holes.push(covered);
            }
            covered = covered.max(end);
        }
        if range.end.saturating_sub(covered) >= DROPPED_SPEECH {
            holes.push(covered);
        }
    }
    holes
}

// Speech split at long pauses; the first piece starts at 0 and the last ends
// at `len`.
fn speech_between_long_pauses(speech: &[Range<usize>], len: usize) -> Vec<Range<usize>> {
    let mut spans: Vec<Range<usize>> = Vec::new();
    for range in speech {
        match spans.last_mut() {
            Some(last) if range.start - last.end < LONG_PAUSE => last.end = range.end,
            _ => spans.push(range.clone()),
        }
    }
    if let Some(first) = spans.first_mut() {
        first.start = 0;
    }
    if let Some(last) = spans.last_mut() {
        last.end = len;
    }
    spans
}

// Each retry replaces the words of `first` in its span. Words outside the
// spans stay, split out of any segment that reaches into one.
fn splice(
    first: TranscriptionResult,
    retries: Vec<(TranscriptionResult, Range<usize>)>,
) -> TranscriptionResult {
    let spans: Vec<Range<f32>> = retries
        .iter()
        .map(|(_, span)| {
            span.start as f32 / SAMPLE_RATE as f32..span.end as f32 / SAMPLE_RATE as f32
        })
        .collect();
    let middle = |item: &TranscriptionSegment| (item.start + item.end) / 2.0;
    // The retry decoded every word that touches its span.
    let retried = |item: &TranscriptionSegment| {
        spans
            .iter()
            .any(|span| item.start < span.end && item.end > span.start)
    };
    // Words separated by a span land in different segments.
    let region = |item: &TranscriptionSegment| {
        spans
            .iter()
            .filter(|span| span.start <= middle(item))
            .count()
    };
    let first_words = first.words.unwrap_or_default();
    let mut segments = Vec::new();
    for segment in first.segments.unwrap_or_default() {
        let words: Vec<_> = first_words
            .iter()
            .filter(|word| word.start >= segment.start && word.end <= segment.end)
            .collect();
        let kept: Vec<_> = words
            .iter()
            .copied()
            .filter(|word| !retried(word))
            .collect();
        let runs: Vec<_> = kept.chunk_by(|a, b| region(a) == region(b)).collect();
        if kept.len() == words.len() && runs.len() <= 1 {
            segments.push(segment);
            continue;
        }
        for run in runs {
            if let (Some(head), Some(tail)) = (run.first(), run.last()) {
                segments.push(TranscriptionSegment {
                    start: head.start,
                    end: tail.end,
                    text: run
                        .iter()
                        .map(|word| word.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                });
            }
        }
    }
    let mut words: Vec<_> = first_words
        .iter()
        .filter(|word| !retried(word))
        .cloned()
        .collect();
    for (retry, _) in retries {
        segments.extend(retry.segments.unwrap_or_default());
        words.extend(retry.words.unwrap_or_default());
    }
    segments.sort_by(|a, b| a.start.total_cmp(&b.start));
    words.sort_by(|a, b| a.start.total_cmp(&b.start));
    TranscriptionResult {
        text: segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        segments: Some(segments),
        words: Some(words),
        language: first.language,
    }
}

fn auto_timestamps(wanted: bool) -> TimestampKind {
    if wanted {
        TimestampKind::Auto
    } else {
        TimestampKind::None
    }
}

// No extension for an empty dictionary, so the decode stays unboosted.
fn parakeet_boost(dictionary: &[String]) -> Option<RunExtension> {
    // Past the entry cap, the newest entries are boosted.
    let newest_first: Vec<String> = dictionary.iter().rev().cloned().collect();
    let mut boost_phrases = sanitize_dictionary_entries(&newest_first);
    // A NUL cannot cross the C API; drop such an entry rather than fail the run.
    boost_phrases.retain(|phrase| !phrase.contains('\0'));
    (!boost_phrases.is_empty()).then_some(RunExtension::Parakeet(ParakeetRunOptions {
        boost_phrases,
        boost_score: None,
    }))
}

fn whisper_prompt(prompt: Option<String>, dictionary: &[String]) -> Option<String> {
    let prompt = prompt.filter(|prompt| !prompt.trim().is_empty());
    match (prompt, build_dictionary_prompt(dictionary)) {
        (Some(prompt), Some(dictionary)) => Some(format!("{prompt}\n\n{dictionary}")),
        (prompt, dictionary) => prompt.or(dictionary),
    }
}

// Distil-Whisper keeps its teacher's encoder, so it shares that companion.
fn distil_teacher(family: &str) -> &str {
    match family {
        "distil-large-v3.5" => "large-v3",
        "distil-medium.en" => "medium.en",
        "distil-small.en" => "small.en",
        other => other,
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

// Results for the chunks in order. Unbatched runs stop at the first error;
// decode_spans decodes any chunk left without a result on its own.
fn run_chunks(
    session: &mut Session,
    chunks: &[&[f32]],
    options: &RunOptions,
    batch: bool,
) -> Result<Vec<Result<Transcript, Error>>, Error> {
    if batch && chunks.len() > 1 {
        return session.run_batch(chunks, options);
    }
    let mut results = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let result = session.run(chunk, options);
        let failed = result.is_err();
        results.push(result);
        if failed {
            break;
        }
    }
    Ok(results)
}

fn decode_chunks(
    samples: &[f32],
    max_samples: Option<usize>,
    decode: impl FnMut(&[&[f32]]) -> Result<Vec<Result<Transcript, Error>>, Error>,
) -> Result<Vec<(usize, Transcript)>, Error> {
    let span = 0..samples.len();
    Ok(decode_spans(samples, std::slice::from_ref(&span), max_samples, decode)?.remove(0))
}

// Decodes every span in one call, each split at quiet points to fit
// `max_samples`. Results are per span, keyed by absolute sample offset.
fn decode_spans(
    samples: &[f32],
    spans: &[Range<usize>],
    max_samples: Option<usize>,
    mut decode: impl FnMut(&[&[f32]]) -> Result<Vec<Result<Transcript, Error>>, Error>,
) -> Result<Vec<Vec<(usize, Transcript)>>, Error> {
    let mut chunks = Vec::new();
    for (index, span) in spans.iter().enumerate() {
        let mut start = span.start;
        while let Some(limit) = max_samples.filter(|limit| span.end - start > *limit) {
            let cut = start + quiet_boundary(&samples[start..span.end], limit);
            chunks.push((index, start..cut));
            start = cut;
        }
        chunks.push((index, start..span.end));
    }
    let pcm: Vec<&[f32]> = chunks
        .iter()
        .map(|(_, range)| &samples[range.clone()])
        .collect();
    let mut outcomes = decode(&pcm)?.into_iter();
    let mut results: Vec<Vec<(usize, Transcript)>> = spans.iter().map(|_| Vec::new()).collect();
    for (index, range) in chunks {
        let mut pending = vec![(range, outcomes.next())];
        while let Some((range, outcome)) = pending.pop() {
            let result = match outcome {
                Some(result) => result,
                None => decode(&[&samples[range.clone()]])?.pop().ok_or_else(|| {
                    Error::Backend("transcribe.cpp returned no result for a chunk".into())
                })?,
            };
            match result {
                Ok(transcript) => results[index].push((range.start, transcript)),
                Err(Error::InputTooLong(_) | Error::OutputTruncated { .. })
                    if range.len() > SAMPLE_RATE =>
                {
                    // Discard the partial output and re-decode both halves. Never
                    // return a successful but incomplete transcript, or retry an
                    // unrelated model/backend error. The one-second floor bounds retries.
                    let cut =
                        range.start + quiet_boundary(&samples[range.clone()], range.len() / 2);
                    pending.push((cut..range.end, None));
                    pending.push((range.start..cut, None));
                }
                Err(error) => return Err(error),
            }
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
        let chunks = super::decode_chunks(&samples, Some(16_000 * 15), |chunks| {
            Ok(chunks
                .iter()
                .map(|chunk| {
                    assert!(chunk.len() <= 16_000 * 15);
                    consumed.extend_from_slice(chunk);
                    Ok(Default::default())
                })
                .collect())
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
        let chunks = super::decode_chunks(&samples, None, |chunks| {
            Ok(chunks
                .iter()
                .map(|chunk| {
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
                .collect())
        })
        .unwrap();
        assert_eq!(consumed, samples.len());
        assert!(chunks.iter().all(|(_, t)| t.text == "complete"));
    }

    #[test]
    fn errors_stop_without_unbounded_retries() {
        let mut attempts = 0;
        let result = super::decode_chunks(&vec![0.0; 16_000 * 4], None, |chunks| {
            attempts += chunks.len();
            Ok(chunks
                .iter()
                .map(|_| Err(transcribe_cpp::Error::InputTooLong("test".into())))
                .collect())
        });
        assert!(result.is_err());
        assert!(attempts <= 4);
        attempts = 0;
        let result = super::decode_chunks(&vec![0.0; 16_000 * 4], None, |chunks| {
            attempts += chunks.len();
            Ok(chunks
                .iter()
                .map(|_| Err(transcribe_cpp::Error::Backend("test".into())))
                .collect())
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
