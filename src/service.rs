use std::{
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};

use anyhow::{Result, anyhow};

#[cfg(local_engines)]
use crate::TranscriptionEngine;

use crate::{
    TimestampGranularity, Transcription, TranscriptionResult,
    models::{
        InstallOptions, InstallSpec, ModelEngine, ModelError, ModelInstallManager, ModelStatus,
        ResolvedModel,
    },
};

pub type ModelResolver = Arc<dyn Fn(&str) -> Option<InstallSpec> + Send + Sync>;

#[derive(Clone)]
pub struct SpeechConfig {
    pub model_cache_dir: PathBuf,
    /// The model catalog.
    pub resolver: ModelResolver,
    /// Resolve ids the resolver doesn't know as a file path or a name in the
    /// cache. When false they are an `Unknown model` error. Only for ids the
    /// local user chose, never for ids from a request.
    pub loose_paths: bool,
}

impl SpeechConfig {
    /// No catalog: every model id is resolved as a file path or a name in the
    /// cache, like [`SpeechService::new_loose_with_engine`].
    pub fn loose(model_cache_dir: PathBuf) -> Self {
        Self {
            model_cache_dir,
            resolver: Arc::new(|_| None),
            loose_paths: true,
        }
    }
}

/// How a service resolves an id its resolver doesn't know.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LooseModels {
    /// Not at all: the id is an unknown model.
    None,
    /// As a plain name inside the cache, or the one pinned reference, which
    /// may be any path.
    Cached { pinned: Option<String> },
    /// As any file path, absolute or relative to the cache.
    Any,
}

#[derive(Debug, Clone)]
pub enum AudioInput {
    WavPath(PathBuf),
    Samples16Khz(Vec<f32>),
    PcmI16 { samples: Vec<i16>, sample_rate: u32 },
}

// Audio shorter than this stays on the loaded CPU engine: on short clips the
// GPU copy barely helps, and loading it would land in a dictation fallback.
const GPU_COPY_MIN_MS: u128 = 30_000;

#[derive(Debug, Clone)]
pub struct TranscribeRequest {
    pub audio: AudioInput,
    pub model_id: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub dictionary: Vec<String>,
    pub timestamps: bool,
    pub timestamp_granularity: Option<TimestampGranularity>,
}

pub struct SpeechService {
    model_manager: ModelInstallManager,
    resolver: ModelResolver,
    loose: LooseModels,
    loose_engine: ModelEngine,
    load_compiling_encoders: bool,
    loaded: Mutex<Option<LoadedEngine>>,
    // One request decodes ahead while another transcribes; the rest wait
    // without holding decoded audio.
    decode_slot: Mutex<()>,
}

struct TranscriptionWithDuration {
    result: TranscriptionResult,
    audio_duration_ms: u128,
}

/// 16 kHz mono samples, padded for the engines, and the input's length.
#[cfg_attr(not(local_engines), allow(dead_code))]
struct PreparedAudio {
    samples: Vec<f32>,
    duration_ms: u128,
}

/// What a request asks of the engine besides its audio. Apple Speech takes
/// no prompt or word timestamps.
#[cfg_attr(not(transcribe_engine), allow(dead_code))]
struct DecodeOptions {
    language: Option<String>,
    prompt: Option<String>,
    dictionary: Vec<String>,
    timestamps: bool,
    word_timestamps: bool,
}

struct LoadedEngine {
    model_id: String,
    path: PathBuf,
    warmed: bool,
    engine: EngineInstance,
    // Nemotron streams on the CPU and transcribes whole recordings with a
    // second copy on the GPU, loaded by the first one.
    gpu_copy: Option<ResolvedModel>,
    batch_engine: Option<EngineInstance>,
}

enum EngineInstance {
    #[cfg(apple_speech_engine)]
    Apple(crate::engines::apple::AppleEngine),
    #[cfg(transcribe_engine)]
    Transcribe(crate::engines::transcribe::TranscribeEngine),
}

#[cfg(streaming_engines)]
impl EngineInstance {
    fn streaming_transcribe_chunk(&mut self, chunk: &[f32]) -> Result<String> {
        match self {
            #[cfg(apple_speech_engine)]
            Self::Apple(engine) => {
                engine.transcribe_chunk(chunk).map_err(boxed_error)?;
                Ok(engine.get_transcript())
            }
            #[cfg(transcribe_engine)]
            Self::Transcribe(engine) => engine.transcribe_chunk(chunk).map_err(boxed_error),
        }
    }

    fn streaming_reset(&mut self) {
        match self {
            #[cfg(apple_speech_engine)]
            Self::Apple(engine) => engine.reset(),
            #[cfg(transcribe_engine)]
            Self::Transcribe(engine) => engine.reset(),
        }
    }

    fn streaming_configure(&mut self, language: Option<String>, dictionary: Vec<String>) {
        match self {
            #[cfg(apple_speech_engine)]
            Self::Apple(engine) => engine.configure_stream(language, dictionary),
            #[cfg(transcribe_engine)]
            Self::Transcribe(engine) => engine.configure_stream(language, dictionary),
        }
    }

    fn streaming_finalize(&mut self) -> Result<String> {
        match self {
            #[cfg(apple_speech_engine)]
            Self::Apple(engine) => engine.finalize().map_err(boxed_error),
            #[cfg(transcribe_engine)]
            Self::Transcribe(engine) => engine.finalize().map_err(boxed_error),
        }
    }

    fn streaming_get_transcript(&self) -> String {
        match self {
            #[cfg(apple_speech_engine)]
            Self::Apple(engine) => engine.get_transcript(),
            #[cfg(transcribe_engine)]
            Self::Transcribe(engine) => engine.get_transcript(),
        }
    }
}

impl SpeechService {
    /// A service over a model catalog: unless the config sets
    /// [`loose_paths`](SpeechConfig::loose_paths), ids the resolver doesn't
    /// know are rejected as `Unknown model`, never loaded as file paths.
    pub fn new(config: SpeechConfig) -> Self {
        Self::build(config, ModelEngine::Whisper)
    }

    /// A service with no catalog: a model id is a file path, a file name in
    /// the cache, or a cache directory holding one model file. Use
    /// [`only_cached_models`](Self::only_cached_models) before resolving ids
    /// that come from a request.
    pub fn new_loose_with_engine(model_cache_dir: PathBuf, engine: ModelEngine) -> Self {
        Self::build(SpeechConfig::loose(model_cache_dir), engine)
    }

    fn build(config: SpeechConfig, loose_engine: ModelEngine) -> Self {
        Self {
            model_manager: ModelInstallManager::new(config.model_cache_dir),
            resolver: config.resolver,
            loose: if config.loose_paths {
                LooseModels::Any
            } else {
                LooseModels::None
            },
            loose_engine,
            load_compiling_encoders: false,
            loaded: Mutex::new(None),
            decode_slot: Mutex::new(()),
        }
    }

    /// For services that resolve ids from requests, like `glimpse-speech
    /// serve`: an id outside the catalog must name a model file or directory
    /// directly inside the cache, so a request can't point the engine at any
    /// other file. `pinned`, the model the operator chose, may still be a path.
    /// Has no effect on a catalog service, which already rejects such ids.
    pub fn only_cached_models(mut self, pinned: Option<String>) -> Self {
        if self.loose != LooseModels::None {
            self.loose = LooseModels::Cached { pinned };
        }
        self
    }

    /// Loads Core ML encoders that are still marked as compiling. Only the
    /// service that runs the compile should use this; others skip the encoder
    /// so a load never waits minutes on a first Neural Engine compile.
    pub fn loading_compiling_encoders(mut self) -> Self {
        self.load_compiling_encoders = true;
        self
    }

    pub fn model_manager(&self) -> &ModelInstallManager {
        &self.model_manager
    }

    pub fn resolve(&self, model_id: &str) -> Result<ResolvedModel> {
        if let Some(spec) = (self.resolver)(model_id) {
            return self.model_manager.resolve(&spec);
        }
        match &self.loose {
            LooseModels::None => Err(unknown_model(model_id)),
            LooseModels::Cached { pinned } if pinned.as_deref() != Some(model_id) => self
                .model_manager
                .resolve_cached(model_id, self.loose_engine),
            LooseModels::Cached { .. } | LooseModels::Any => self
                .model_manager
                .resolve_loose(model_id, self.loose_engine),
        }
    }

    fn spec(&self, model_id: &str) -> Result<InstallSpec> {
        (self.resolver)(model_id).ok_or_else(|| unknown_model(model_id))
    }

    pub async fn install(
        &self,
        model_id: &str,
        options: InstallOptions<'_>,
    ) -> Result<ModelStatus> {
        let spec = self.spec(model_id)?;
        self.model_manager.install(&spec, options).await
    }

    pub fn model_status(&self, model_id: &str) -> Result<ModelStatus> {
        let spec = self.spec(model_id)?;
        self.model_manager.status(&spec)
    }

    /// Deletes a model's directory from the cache. A catalog service only
    /// deletes catalog models, and [`only_cached_models`](Self::only_cached_models)
    /// only model directories in the cache; anything else is `Unknown model`.
    pub fn delete(&self, model_id: &str) -> Result<ModelStatus> {
        let id = match (self.resolver)(model_id) {
            Some(spec) => spec.id,
            None => match &self.loose {
                LooseModels::Any => model_id.to_string(),
                LooseModels::Cached { .. } if self.model_manager.is_cached_model_dir(model_id) => {
                    model_id.to_string()
                }
                _ => return Err(unknown_model(model_id)),
            },
        };
        if self.loaded_model_id().as_deref() == Some(id.as_str()) {
            self.unload();
        }
        self.model_manager.delete(&id)
    }

    pub fn transcribe(&self, request: TranscribeRequest) -> Result<Transcription> {
        self.transcribe_before(request, None)
    }

    /// Like [`transcribe`](Self::transcribe), but gives up with an error when
    /// `deadline` has passed by the time the engine is free, for example
    /// after queueing behind other transcriptions. A started transcription
    /// runs to the end.
    pub fn transcribe_before(
        &self,
        request: TranscribeRequest,
        deadline: Option<Instant>,
    ) -> Result<Transcription> {
        let total_started = Instant::now();
        let TranscribeRequest {
            audio,
            model_id: requested_model,
            language: requested_language,
            prompt,
            dictionary,
            timestamps,
            timestamp_granularity,
        } = request;
        // Reject a bad model id before spending time decoding.
        self.resolve(&requested_model)?;
        let decode_slot = self
            .decode_slot
            .lock()
            .map_err(|_| anyhow!("speech service lock poisoned"))?;
        check_deadline(deadline, &requested_model)?;
        let audio = prepare_audio(audio)?;
        let mut guard = self.ensure_loaded_before(&requested_model, deadline)?;
        drop(decode_slot);
        let loaded = loaded_engine(&mut guard)?;
        let resolved_id = loaded.model_id.clone();
        let long = audio.duration_ms >= GPU_COPY_MIN_MS;
        if long && let Some(resolved) = loaded.gpu_copy.take() {
            match load_engine(&resolved, self.load_compiling_encoders, true) {
                Ok(engine) => loaded.batch_engine = Some(engine),
                Err(error) => tracing::warn!(
                    "[SpeechService] {} stays on the CPU, its GPU copy failed to load: {error}",
                    resolved.id
                ),
            }
        }
        let on_batch_engine = long && loaded.batch_engine.is_some();
        let engine = match loaded.batch_engine.as_mut() {
            Some(batch) if on_batch_engine => batch,
            _ => &mut loaded.engine,
        };
        let options = DecodeOptions {
            language: requested_language.clone(),
            prompt,
            dictionary,
            timestamps: timestamps || timestamp_granularity.is_some(),
            word_timestamps: timestamp_granularity == Some(TimestampGranularity::Word),
        };
        let transcribe_started = Instant::now();
        let transcription = transcribe_with_engine(engine, audio, options)?;
        let transcribe_elapsed = transcribe_started.elapsed();
        // Only the main engine serves warmups and streaming.
        if !on_batch_engine {
            loaded.warmed = true;
        }
        tracing::info!(
            "[SpeechService] transcribe model={} resolved={} total={:.2}s engine={:.2}s",
            requested_model,
            resolved_id,
            total_started.elapsed().as_secs_f32(),
            transcribe_elapsed.as_secs_f32()
        );

        let TranscriptionResult {
            text,
            segments,
            words,
            language,
        } = transcription.result;
        Ok(Transcription {
            text,
            segments,
            words,
            model_id: resolved_id,
            language: language.or(requested_language),
            duration_ms: transcription.audio_duration_ms,
        })
    }

    pub fn preload_and_warm(&self, model_id: &str) -> Result<()> {
        let total_started = Instant::now();
        let mut guard = self.ensure_loaded(model_id)?;
        let loaded = loaded_engine(&mut guard)?;
        if loaded.warmed {
            tracing::info!(
                "[SpeechService] warm model={} skipped already_warmed total={:.2}s",
                model_id,
                total_started.elapsed().as_secs_f32()
            );
            return Ok(());
        }

        let silence = prepare_audio(AudioInput::Samples16Khz(vec![0.0f32; 16_000 * 2]))?;
        let warm_started = Instant::now();
        transcribe_with_engine(
            &mut loaded.engine,
            silence,
            DecodeOptions {
                language: None,
                prompt: None,
                dictionary: Vec::new(),
                timestamps: false,
                word_timestamps: false,
            },
        )?;
        loaded.warmed = true;
        tracing::info!(
            "[SpeechService] warm model={} total={:.2}s silence_transcribe={:.2}s",
            model_id,
            total_started.elapsed().as_secs_f32(),
            warm_started.elapsed().as_secs_f32()
        );
        Ok(())
    }

    pub fn unload(&self) {
        if let Ok(mut guard) = self.loaded.lock() {
            *guard = None;
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.lock().is_ok_and(|guard| guard.is_some())
    }

    pub fn loaded_model_id(&self) -> Option<String> {
        self.loaded
            .lock()
            .ok()?
            .as_ref()
            .map(|loaded| loaded.model_id.clone())
    }

    #[cfg(streaming_engines)]
    pub fn streaming_transcribe_chunk(&self, model_id: &str, chunk: &[f32]) -> Result<String> {
        let mut guard = self.ensure_loaded(model_id)?;
        loaded_engine(&mut guard)?
            .engine
            .streaming_transcribe_chunk(chunk)
    }

    #[cfg(streaming_engines)]
    pub fn streaming_reset(&self) {
        if let Ok(mut guard) = self.loaded.lock()
            && let Some(loaded) = guard.as_mut()
        {
            loaded.engine.streaming_reset();
        }
    }

    /// Loads the model and sets language and vocabulary for its next
    /// streaming session. Fails when the model doesn't load.
    #[cfg(streaming_engines)]
    pub fn streaming_configure(
        &self,
        model_id: &str,
        language: Option<String>,
        dictionary: Vec<String>,
    ) -> Result<()> {
        let mut guard = self.ensure_loaded(model_id)?;
        loaded_engine(&mut guard)?
            .engine
            .streaming_configure(language, dictionary);
        Ok(())
    }

    /// Ends the stream and returns the final transcript.
    #[cfg(streaming_engines)]
    pub fn streaming_finalize(&self) -> String {
        let Ok(mut guard) = self.loaded.lock() else {
            return String::new();
        };
        let Some(loaded) = guard.as_mut() else {
            return String::new();
        };
        loaded.engine.streaming_finalize().unwrap_or_else(|err| {
            tracing::error!("[SpeechService] streaming finalize failed: {err}");
            loaded.engine.streaming_get_transcript()
        })
    }

    #[cfg(streaming_engines)]
    pub fn streaming_get_transcript(&self) -> String {
        self.loaded
            .lock()
            .ok()
            .and_then(|guard| {
                guard
                    .as_ref()
                    .map(|loaded| loaded.engine.streaming_get_transcript())
            })
            .unwrap_or_default()
    }

    fn ensure_loaded(&self, model_id: &str) -> Result<MutexGuard<'_, Option<LoadedEngine>>> {
        self.ensure_loaded_before(model_id, None)
    }

    /// Fails without loading anything when `deadline` passed while waiting
    /// for the engine.
    fn ensure_loaded_before(
        &self,
        model_id: &str,
        deadline: Option<Instant>,
    ) -> Result<MutexGuard<'_, Option<LoadedEngine>>> {
        let total_started = Instant::now();
        let resolve_started = Instant::now();
        let resolved = self.resolve(model_id)?;
        let resolve_elapsed = resolve_started.elapsed();
        let lock_started = Instant::now();
        let mut guard = self.lock_loaded()?;
        let lock_wait = lock_started.elapsed();
        check_deadline(deadline, model_id)?;
        let should_reload = guard
            .as_ref()
            .is_none_or(|loaded| loaded.model_id != resolved.id || loaded.path != resolved.path);

        if should_reload {
            let load_started = Instant::now();
            let bytes = std::fs::metadata(&resolved.path)
                .ok()
                .map(|metadata| metadata.len());
            tracing::info!(
                "[SpeechService] load start model={} engine={} path={} bytes={:?}",
                resolved.id,
                resolved.engine,
                resolved.path.display(),
                bytes
            );
            *guard = None;
            let engine = load_engine(&resolved, self.load_compiling_encoders, false)?;
            let load_elapsed = load_started.elapsed();
            *guard = Some(LoadedEngine {
                model_id: resolved.id.clone(),
                path: resolved.path.clone(),
                warmed: false,
                engine,
                gpu_copy: streams_on_cpu(&resolved).then(|| resolved.clone()),
                batch_engine: None,
            });
            tracing::info!(
                "[SpeechService] ensure_loaded model={} reloaded=true total={:.2}s resolve={:.2}s lock_wait={:.2}s load={:.2}s",
                resolved.id,
                total_started.elapsed().as_secs_f32(),
                resolve_elapsed.as_secs_f32(),
                lock_wait.as_secs_f32(),
                load_elapsed.as_secs_f32()
            );
        } else {
            tracing::debug!(
                "[SpeechService] ensure_loaded model={} reloaded=false total={:.2}s resolve={:.2}s lock_wait={:.2}s",
                resolved.id,
                total_started.elapsed().as_secs_f32(),
                resolve_elapsed.as_secs_f32(),
                lock_wait.as_secs_f32()
            );
        }

        Ok(guard)
    }

    fn lock_loaded(&self) -> Result<MutexGuard<'_, Option<LoadedEngine>>> {
        self.loaded
            .lock()
            .map_err(|_| anyhow!("speech service lock poisoned"))
    }

    /// Holds the engine as a running transcription would.
    #[cfg(all(test, feature = "api"))]
    pub(crate) fn hold_engine(&self) -> impl Sized + '_ {
        self.loaded.lock().unwrap()
    }
}

impl Clone for SpeechService {
    fn clone(&self) -> Self {
        Self {
            model_manager: self.model_manager.clone(),
            resolver: Arc::clone(&self.resolver),
            loose: self.loose.clone(),
            loose_engine: self.loose_engine,
            load_compiling_encoders: self.load_compiling_encoders,
            loaded: Mutex::new(None),
            decode_slot: Mutex::new(()),
        }
    }
}

pub type SharedSpeechService = Arc<SpeechService>;

fn unknown_model(model_id: &str) -> anyhow::Error {
    ModelError::UnknownModel(model_id.to_string()).into()
}

fn loaded_engine<'a>(
    guard: &'a mut MutexGuard<'_, Option<LoadedEngine>>,
) -> Result<&'a mut LoadedEngine> {
    guard.as_mut().ok_or_else(|| anyhow!("model did not load"))
}

#[cfg(local_engines)]
fn boxed_error(err: Box<dyn std::error::Error>) -> anyhow::Error {
    anyhow!(err.to_string())
}

// Nemotron streams on the CPU: its 560 ms chunks finished later on Vulkan
// (RTX 4000 Ada) and less steadily behind other GPU work on an M2 Pro. Whole
// recordings ran several times faster on the GPU, so they get a second copy,
// except Nemotron 3.5 on Metal, which was less accurate in Hungarian there.
fn streams_on_cpu(resolved: &ResolvedModel) -> bool {
    let variant = resolved.variant.as_deref().unwrap_or_default();
    #[cfg(transcribe_engine)]
    let gpu = transcribe_cpp::backend_available(transcribe_cpp::Backend::Vulkan)
        || (transcribe_cpp::backend_available(transcribe_cpp::Backend::Metal)
            && !variant.starts_with("nemotron-35-"));
    #[cfg(not(transcribe_engine))]
    let gpu = false;
    gpu && variant.starts_with("nemotron-")
}

#[cfg_attr(not(transcribe_engine), allow(unused_variables))]
fn load_engine(
    resolved: &ResolvedModel,
    load_compiling_encoders: bool,
    gpu_copy: bool,
) -> Result<EngineInstance> {
    match resolved.engine {
        ModelEngine::Whisper | ModelEngine::Transcribe => {
            #[cfg(transcribe_engine)]
            {
                use crate::engines::transcribe::{TranscribeEngine, TranscribeModelParams};

                let mut engine = TranscribeEngine::new();
                // A full model runs without its Core ML encoder; a decoder-only file needs it.
                let optional_encoder = !resolved
                    .path
                    .file_stem()
                    .is_some_and(|stem| stem.to_string_lossy().ends_with("-decoder"));
                let coreml_encoder =
                    TranscribeEngine::companion_for(&resolved.path).filter(|dir| {
                        load_compiling_encoders
                            || !optional_encoder
                            || !TranscribeEngine::is_compiling(dir)
                    });
                let variant = resolved.variant.as_deref().unwrap_or_default();
                // A full Parakeet model streams through its ggml encoder, which is
                // three times slower on the CPU, so only decoder-only files go there.
                let backend = if (coreml_encoder.is_some()
                    && !optional_encoder
                    && variant.starts_with("parakeet-"))
                    || (variant.starts_with("nemotron-") && !gpu_copy)
                {
                    transcribe_cpp::Backend::Cpu
                } else {
                    transcribe_cpp::Backend::Auto
                };
                let params = TranscribeModelParams {
                    coreml_encoder,
                    backend,
                };
                engine
                    .load_model_with_params(&resolved.path, params)
                    .map_err(boxed_error)?;
                Ok(EngineInstance::Transcribe(engine))
            }
            #[cfg(not(transcribe_engine))]
            {
                Err(anyhow!(
                    "transcribe.cpp support is not enabled on this build"
                ))
            }
        }
        ModelEngine::Apple => {
            #[cfg(apple_speech_engine)]
            {
                let mut engine = crate::engines::apple::AppleEngine::new();
                engine
                    .load_model_with_params(&resolved.path, ())
                    .map_err(boxed_error)?;
                Ok(EngineInstance::Apple(engine))
            }
            #[cfg(not(apple_speech_engine))]
            {
                Err(anyhow!("Apple speech support is not enabled on this build"))
            }
        }
    }
}

/// A queued transcription whose deadline passed before it started.
#[derive(Debug, thiserror::Error)]
#[error("The transcription timed out waiting for the engine (model {0})")]
pub(crate) struct DeadlinePassed(String);

fn check_deadline(deadline: Option<Instant>, model_id: &str) -> Result<()> {
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Err(DeadlinePassed(model_id.to_string()).into());
    }
    Ok(())
}

#[cfg_attr(not(local_engines), allow(unused_variables))]
fn transcribe_with_engine(
    engine: &mut EngineInstance,
    audio: PreparedAudio,
    options: DecodeOptions,
) -> Result<TranscriptionWithDuration> {
    match engine {
        #[cfg(apple_speech_engine)]
        EngineInstance::Apple(engine) => {
            let params = crate::engines::apple::AppleInferenceParams {
                language: options.language,
                long_form: options.timestamps,
                dictionary: options.dictionary,
            };
            transcribe_audio(engine, audio, Some(params))
        }
        #[cfg(transcribe_engine)]
        EngineInstance::Transcribe(engine) => {
            let params = crate::engines::transcribe::TranscribeInferenceParams {
                language: options.language,
                dictionary: options.dictionary,
                prompt: options.prompt,
                timestamps: options.timestamps,
                word_timestamps: options.word_timestamps,
            };
            transcribe_audio(engine, audio, Some(params))
        }
        #[allow(unreachable_patterns)]
        _ => Err(anyhow!("No speech engine support is enabled")),
    }
}

#[cfg(local_engines)]
fn transcribe_audio<E: TranscriptionEngine>(
    engine: &mut E,
    prepared: PreparedAudio,
    params: Option<E::InferenceParams>,
) -> Result<TranscriptionWithDuration> {
    let mut result = engine
        .transcribe_samples(prepared.samples, params)
        .map_err(boxed_error)?;
    // Decoder timings may include the silence added by prepare_audio.
    let duration = prepared.duration_ms as f32 / 1000.0;
    for span in result
        .segments
        .iter_mut()
        .chain(result.words.iter_mut())
        .flatten()
    {
        span.start = span.start.clamp(0.0, duration);
        span.end = span.end.clamp(span.start, duration);
    }
    Ok(TranscriptionWithDuration {
        result,
        audio_duration_ms: prepared.duration_ms,
    })
}

fn prepare_audio(audio: AudioInput) -> Result<PreparedAudio> {
    const MIN_SAMPLES: usize = 16_000;
    const EXTRA_PADDING: usize = 4_000;

    let (mut samples, source_sample_rate, source_sample_count) = match audio {
        AudioInput::WavPath(path) => {
            let samples =
                crate::audio::read_audio_samples(&path).map_err(|err| anyhow!(err.to_string()))?;
            let sample_count = samples.len();
            (samples, 16_000, sample_count)
        }
        AudioInput::Samples16Khz(samples) => {
            let sample_count = samples.len();
            (samples, 16_000, sample_count)
        }
        AudioInput::PcmI16 {
            samples,
            sample_rate,
        } => {
            if sample_rate > crate::audio::MAX_SAMPLE_RATE {
                return Err(anyhow!("Unsupported sample rate {sample_rate} Hz"));
            }
            let sample_count = samples.len();
            // Normalizes and resamples in one pass; at 16 kHz it only scales.
            let normalized = crate::audio::resample_i16_to_f32(&samples, sample_rate, 16_000);
            (normalized, sample_rate, sample_count)
        }
    };

    let padding_needed = MIN_SAMPLES.saturating_sub(samples.len()) + EXTRA_PADDING;
    samples.extend(std::iter::repeat_n(0.0f32, padding_needed));
    Ok(PreparedAudio {
        samples,
        duration_ms: audio_duration_ms(source_sample_count, source_sample_rate),
    })
}

fn audio_duration_ms(sample_count: usize, sample_rate: u32) -> u128 {
    if sample_rate == 0 {
        return 0;
    }
    (sample_count as u128 * 1000) / u128::from(sample_rate)
}

#[cfg(test)]
mod prepare_tests {
    use super::{AudioInput, prepare_audio};

    #[test]
    fn pcm_with_zero_sample_rate_does_not_blow_up() {
        let prepared = prepare_audio(AudioInput::PcmI16 {
            samples: vec![1i16, 2, 3, 4],
            sample_rate: 0,
        })
        .expect("prepare_audio");

        assert_eq!(prepared.samples.len(), 20_000);
        assert_eq!(prepared.duration_ms, 0);
    }

    #[test]
    fn pcm_above_the_supported_rate_is_rejected() {
        let result = prepare_audio(AudioInput::PcmI16 {
            samples: vec![0i16; 1_024],
            sample_rate: crate::audio::MAX_SAMPLE_RATE + 1,
        });
        assert!(result.is_err());
    }

    #[test]
    fn duration_uses_the_input_rate() {
        let prepared = prepare_audio(AudioInput::PcmI16 {
            samples: vec![0i16; 8_000 * 40],
            sample_rate: 8_000,
        })
        .expect("prepare_audio");

        assert_eq!(prepared.duration_ms, 40_000);
        assert!(prepared.duration_ms >= super::GPU_COPY_MIN_MS);
        assert_eq!(prepared.samples.len(), 16_000 * 40 + 4_000);
    }

    #[test]
    fn pcm_at_16khz_preserves_sample_count_and_duration() {
        let prepared = prepare_audio(AudioInput::PcmI16 {
            samples: vec![0i16; 32_000],
            sample_rate: 16_000,
        })
        .expect("prepare_audio");

        assert_eq!(prepared.duration_ms, 2_000);
        assert_eq!(prepared.samples.len(), 32_000 + 4_000);
    }
}

#[cfg(test)]
mod resolve_tests {
    use std::{fs, path::Path};

    use super::*;
    use crate::models::{ModelStorage, RemoteFile};

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "glimpse-speech-service-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("models/stray")).unwrap();
        fs::write(dir.join("models/stray/ggml-stray.bin"), b"ggml").unwrap();
        fs::write(dir.join("planted.bin"), b"not a model").unwrap();
        dir
    }

    fn catalog_spec(id: &str) -> InstallSpec {
        InstallSpec {
            id: id.to_string(),
            engine: ModelEngine::Whisper,
            storage: ModelStorage::File {
                artifact: "ggml-known.bin".to_string(),
            },
            files: vec![RemoteFile {
                url: "https://example.test/ggml-known.bin".to_string(),
                path: "ggml-known.bin".to_string(),
                size_bytes: None,
                sha256: None,
                extract: false,
            }],
            variant: None,
        }
    }

    fn catalog_service(models: &Path) -> SpeechService {
        SpeechService::new(SpeechConfig {
            model_cache_dir: models.to_path_buf(),
            resolver: Arc::new(|id| (id == "known").then(|| catalog_spec(id))),
            loose_paths: false,
        })
    }

    fn assert_unknown(service: &SpeechService, id: &str) {
        let err = service.resolve(id).unwrap_err().to_string();
        assert!(err.starts_with("Unknown model:"), "{id}: {err}");
    }

    #[test]
    fn catalog_service_never_resolves_paths() {
        let dir = scratch("catalog");
        let models = dir.join("models");
        fs::create_dir_all(models.join("known")).unwrap();
        fs::write(models.join("known/ggml-known.bin"), b"ggml").unwrap();
        let service = catalog_service(&models);

        assert_eq!(
            service.resolve("known").unwrap().path,
            models.join("known/ggml-known.bin")
        );
        let planted = dir.join("planted.bin");
        for id in [
            planted.to_str().unwrap(),
            "../planted.bin",
            "stray",
            "stray/ggml-stray.bin",
        ] {
            assert_unknown(&service, id);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn loose_config_still_resolves_paths() {
        let dir = scratch("loose");
        let planted = dir.join("planted.bin");
        let service = SpeechService::new(SpeechConfig::loose(dir.join("models")));
        assert_eq!(
            service.resolve(planted.to_str().unwrap()).unwrap().path,
            planted
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_catalog_with_loose_paths_resolves_both() {
        let dir = scratch("catalog-loose");
        let models = dir.join("models");
        fs::create_dir_all(models.join("known")).unwrap();
        fs::write(models.join("known/ggml-known.bin"), b"ggml").unwrap();
        let planted = dir.join("planted.bin");
        let service = SpeechService::new(SpeechConfig {
            model_cache_dir: models.clone(),
            resolver: Arc::new(|id| (id == "known").then(|| catalog_spec(id))),
            loose_paths: true,
        });
        assert_eq!(
            service.resolve("known").unwrap().path,
            models.join("known/ggml-known.bin")
        );
        assert_eq!(
            service.resolve(planted.to_str().unwrap()).unwrap().path,
            planted
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cached_models_take_names_and_the_pinned_path_only() {
        let dir = scratch("cached");
        let models = dir.join("models");
        let pinned = dir.join("pinned.bin");
        fs::write(&pinned, b"ggml").unwrap();
        let pinned_id = pinned.to_str().unwrap().to_string();
        let service = SpeechService::new_loose_with_engine(models.clone(), ModelEngine::Whisper)
            .only_cached_models(Some(pinned_id.clone()));

        assert_eq!(service.resolve(&pinned_id).unwrap().path, pinned);
        assert_eq!(
            service.resolve("stray").unwrap().path,
            models.join("stray/ggml-stray.bin")
        );
        let planted = dir.join("planted.bin");
        for id in [planted.to_str().unwrap(), "../planted.bin", "../pinned.bin"] {
            assert_unknown(&service, id);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_transcription_past_its_deadline_never_loads() {
        let dir = scratch("deadline");
        let service =
            SpeechService::new_loose_with_engine(dir.join("models"), ModelEngine::Whisper);
        let err = service
            .transcribe_before(
                TranscribeRequest {
                    audio: AudioInput::Samples16Khz(vec![0.0; 16_000]),
                    model_id: "stray".to_string(),
                    language: None,
                    prompt: None,
                    dictionary: Vec::new(),
                    timestamps: false,
                    timestamp_granularity: None,
                },
                Some(Instant::now()),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("timed out waiting for the engine"), "{err}");
        assert!(!service.is_loaded());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_cached_models_keeps_a_catalog_service_strict() {
        let dir = scratch("strict");
        let service = catalog_service(&dir.join("models")).only_cached_models(None);
        assert_unknown(&service, "stray");
        let _ = fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod delete_tests {
    use std::fs;

    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "glimpse-speech-delete-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        for model in ["known", "stray"] {
            fs::create_dir_all(dir.join(model)).unwrap();
            fs::write(dir.join(model).join("ggml.bin"), b"ggml").unwrap();
        }
        dir
    }

    fn known_spec() -> InstallSpec {
        InstallSpec {
            id: "known".to_string(),
            engine: ModelEngine::Whisper,
            storage: crate::models::ModelStorage::Directory,
            files: Vec::new(),
            variant: None,
        }
    }

    #[test]
    fn catalog_service_deletes_only_catalog_models() {
        let dir = scratch("catalog");
        let service = SpeechService::new(SpeechConfig {
            model_cache_dir: dir.clone(),
            resolver: Arc::new(|id| (id == "known").then(known_spec)),
            loose_paths: false,
        });

        let err = service.delete("stray").unwrap_err().to_string();
        assert!(err.starts_with("Unknown model:"), "{err}");
        assert!(dir.join("stray").is_dir());

        service.delete("known").unwrap();
        assert!(!dir.join("known").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cached_service_deletes_only_cache_directories() {
        let dir = scratch("cached");
        fs::write(dir.join("ggml-file.bin"), b"ggml").unwrap();
        let service = SpeechService::new_loose_with_engine(dir.clone(), ModelEngine::Whisper)
            .only_cached_models(None);

        for id in ["missing", "ggml-file.bin", "..", "../stray"] {
            let err = service.delete(id).unwrap_err().to_string();
            assert!(err.starts_with("Unknown model:"), "{id}: {err}");
        }
        assert!(dir.join("ggml-file.bin").is_file());

        service.delete("stray").unwrap();
        assert!(!dir.join("stray").exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
