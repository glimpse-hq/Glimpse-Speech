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
        InstallOptions, InstallSpec, ModelEngine, ModelInstallManager, ModelStatus, ResolvedModel,
    },
};

pub type ModelResolver = Arc<dyn Fn(&str) -> Option<InstallSpec> + Send + Sync>;

#[derive(Clone)]
pub struct SpeechConfig {
    pub model_cache_dir: PathBuf,
    pub resolver: ModelResolver,
}

impl SpeechConfig {
    pub fn loose(model_cache_dir: PathBuf) -> Self {
        Self {
            model_cache_dir,
            resolver: Arc::new(|_| None),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AudioInput {
    WavPath(PathBuf),
    Samples16Khz(Vec<f32>),
    PcmI16 { samples: Vec<i16>, sample_rate: u32 },
}

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
    loose_engine: ModelEngine,
    loaded: Mutex<Option<LoadedEngine>>,
}

struct TranscriptionWithDuration {
    result: TranscriptionResult,
    audio_duration_ms: u128,
}

#[cfg(local_engines)]
struct PreparedAudio {
    samples: Vec<f32>,
    duration_ms: u128,
}

struct LoadedEngine {
    model_id: String,
    path: PathBuf,
    warmed: bool,
    engine: EngineInstance,
}

enum EngineInstance {
    #[cfg(feature = "whisper")]
    Whisper(crate::engines::whisper::WhisperEngine),
    #[cfg(nvidia_engines)]
    Parakeet(crate::engines::parakeet::ParakeetEngine),
    #[cfg(nvidia_engines)]
    Nemotron(crate::engines::nemotron::NemotronEngine),
    #[cfg(apple_speech_engine)]
    Apple(crate::engines::apple::AppleEngine),
    #[cfg(transcribe_engine)]
    Transcribe(crate::engines::transcribe::TranscribeEngine),
}

#[cfg(streaming_engines)]
impl EngineInstance {
    fn streaming_transcribe_chunk(&mut self, chunk: &[f32]) -> Result<String> {
        match self {
            #[cfg(nvidia_engines)]
            Self::Parakeet(engine) => {
                engine.transcribe_chunk(chunk).map_err(boxed_error)?;
                Ok(engine.get_transcript())
            }
            #[cfg(nvidia_engines)]
            Self::Nemotron(engine) => {
                engine.transcribe_chunk(chunk).map_err(boxed_error)?;
                Ok(engine.get_transcript())
            }
            #[cfg(apple_speech_engine)]
            Self::Apple(engine) => {
                engine.transcribe_chunk(chunk).map_err(boxed_error)?;
                Ok(engine.get_transcript())
            }
            #[cfg(feature = "whisper")]
            Self::Whisper(_) => Err(anyhow!(
                "Streaming is only supported with Apple, Nemotron, or unified Parakeet models"
            )),
            #[cfg(transcribe_engine)]
            Self::Transcribe(_) => Err(anyhow!(
                "Streaming is only supported with Apple, Nemotron, or unified Parakeet models"
            )),
        }
    }

    fn streaming_reset(&mut self) {
        match self {
            #[cfg(nvidia_engines)]
            Self::Parakeet(engine) => engine.reset(),
            #[cfg(nvidia_engines)]
            Self::Nemotron(engine) => engine.reset(),
            #[cfg(apple_speech_engine)]
            Self::Apple(engine) => engine.reset(),
            #[cfg(feature = "whisper")]
            Self::Whisper(_) => {}
            #[cfg(transcribe_engine)]
            Self::Transcribe(_) => {}
        }
    }

    #[cfg_attr(not(apple_speech_engine), allow(unused_variables))]
    fn streaming_configure(&mut self, language: Option<String>, dictionary: Vec<String>) {
        match self {
            #[cfg(apple_speech_engine)]
            Self::Apple(engine) => engine.configure_stream(language, dictionary),
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    fn streaming_finalize(&mut self) -> Result<String> {
        match self {
            #[cfg(apple_speech_engine)]
            Self::Apple(engine) => engine.finalize().map_err(boxed_error),
            #[allow(unreachable_patterns)]
            _ => Ok(self.streaming_get_transcript().unwrap_or_default()),
        }
    }

    fn streaming_get_transcript(&self) -> Option<String> {
        match self {
            #[cfg(nvidia_engines)]
            Self::Parakeet(engine) => Some(engine.get_transcript()),
            #[cfg(nvidia_engines)]
            Self::Nemotron(engine) => Some(engine.get_transcript()),
            #[cfg(apple_speech_engine)]
            Self::Apple(engine) => Some(engine.get_transcript()),
            #[cfg(feature = "whisper")]
            Self::Whisper(_) => None,
            #[cfg(transcribe_engine)]
            Self::Transcribe(_) => None,
        }
    }
}

impl SpeechService {
    pub fn new(config: SpeechConfig) -> Self {
        Self::build(
            config.model_cache_dir,
            config.resolver,
            ModelEngine::Whisper,
        )
    }

    pub fn new_loose_with_engine(model_cache_dir: PathBuf, engine: ModelEngine) -> Self {
        Self::build(model_cache_dir, Arc::new(|_| None), engine)
    }

    fn build(model_cache_dir: PathBuf, resolver: ModelResolver, loose_engine: ModelEngine) -> Self {
        crate::silence_native_logs();
        Self {
            model_manager: ModelInstallManager::new(model_cache_dir),
            resolver,
            loose_engine,
            loaded: Mutex::new(None),
        }
    }

    pub fn model_manager(&self) -> &ModelInstallManager {
        &self.model_manager
    }

    pub fn resolve(&self, model_id: &str) -> Result<ResolvedModel> {
        match (self.resolver)(model_id) {
            Some(spec) => self.model_manager.resolve(&spec),
            None => self
                .model_manager
                .resolve_loose(model_id, self.loose_engine),
        }
    }

    fn spec(&self, model_id: &str) -> Result<InstallSpec> {
        (self.resolver)(model_id).ok_or_else(|| anyhow!("Unknown model: {model_id}"))
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

    pub fn delete(&self, model_id: &str) -> Result<ModelStatus> {
        self.model_manager.delete(model_id)
    }

    pub fn transcribe(&self, request: TranscribeRequest) -> Result<Transcription> {
        let total_started = Instant::now();
        let requested_language = request.language.clone();
        let requested_model = request.model_id.clone();
        let resolved_id = self.ensure_loaded(&requested_model)?;
        let lock_started = Instant::now();
        let mut guard = self.lock_loaded()?;
        let lock_wait = lock_started.elapsed();
        let loaded = loaded_engine(&mut guard)?;
        let transcribe_started = Instant::now();
        let transcription = transcribe_with_engine(&mut loaded.engine, request)?;
        let transcribe_elapsed = transcribe_started.elapsed();
        loaded.warmed = true;
        tracing::info!(
            "[SpeechService] transcribe model={} resolved={} total={:.2}s lock_wait={:.2}s engine={:.2}s",
            requested_model,
            resolved_id,
            total_started.elapsed().as_secs_f32(),
            lock_wait.as_secs_f32(),
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
        self.ensure_loaded(model_id)?;
        let lock_started = Instant::now();
        let mut guard = self.lock_loaded()?;
        let lock_wait = lock_started.elapsed();
        let loaded = loaded_engine(&mut guard)?;
        if loaded.warmed {
            tracing::info!(
                "[SpeechService] warm model={} skipped already_warmed total={:.2}s lock_wait={:.2}s",
                model_id,
                total_started.elapsed().as_secs_f32(),
                lock_wait.as_secs_f32()
            );
            return Ok(());
        }

        let silence = vec![0.0f32; 16_000 * 2];
        let warm_started = Instant::now();
        transcribe_with_engine(
            &mut loaded.engine,
            TranscribeRequest {
                audio: AudioInput::Samples16Khz(silence),
                model_id: loaded.model_id.clone(),
                language: None,
                prompt: None,
                dictionary: Vec::new(),
                timestamps: false,
                timestamp_granularity: None,
            },
        )?;
        loaded.warmed = true;
        tracing::info!(
            "[SpeechService] warm model={} total={:.2}s lock_wait={:.2}s silence_transcribe={:.2}s",
            model_id,
            total_started.elapsed().as_secs_f32(),
            lock_wait.as_secs_f32(),
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
        self.ensure_loaded(model_id)?;
        let mut guard = self.lock_loaded()?;
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

    /// Sets language and vocabulary for the next streaming session, for
    /// engines that take per-session configuration.
    #[cfg(streaming_engines)]
    pub fn streaming_configure(
        &self,
        model_id: &str,
        language: Option<String>,
        dictionary: Vec<String>,
    ) {
        if self.ensure_loaded(model_id).is_err() {
            return;
        }
        if let Ok(mut guard) = self.loaded.lock()
            && let Some(loaded) = guard.as_mut()
        {
            loaded.engine.streaming_configure(language, dictionary);
        }
    }

    /// Ends the stream and returns the final transcript. Engines that
    /// finalize per chunk just return the current transcript.
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
            loaded.engine.streaming_get_transcript().unwrap_or_default()
        })
    }

    #[cfg(streaming_engines)]
    pub fn streaming_get_transcript(&self) -> String {
        self.loaded
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref()?.engine.streaming_get_transcript())
            .unwrap_or_default()
    }

    fn ensure_loaded(&self, model_id: &str) -> Result<String> {
        let total_started = Instant::now();
        let resolve_started = Instant::now();
        let resolved = self.resolve(model_id)?;
        let resolve_elapsed = resolve_started.elapsed();
        let lock_started = Instant::now();
        let mut guard = self.lock_loaded()?;
        let lock_wait = lock_started.elapsed();
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
            let engine = load_engine(&resolved)?;
            let load_elapsed = load_started.elapsed();
            *guard = Some(LoadedEngine {
                model_id: resolved.id.clone(),
                path: resolved.path.clone(),
                warmed: false,
                engine,
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

        Ok(resolved.id)
    }

    fn lock_loaded(&self) -> Result<MutexGuard<'_, Option<LoadedEngine>>> {
        self.loaded
            .lock()
            .map_err(|_| anyhow!("speech service lock poisoned"))
    }
}

impl Clone for SpeechService {
    fn clone(&self) -> Self {
        Self {
            model_manager: self.model_manager.clone(),
            resolver: Arc::clone(&self.resolver),
            loose_engine: self.loose_engine,
            loaded: Mutex::new(None),
        }
    }
}

pub type SharedSpeechService = Arc<SpeechService>;

fn loaded_engine<'a>(
    guard: &'a mut MutexGuard<'_, Option<LoadedEngine>>,
) -> Result<&'a mut LoadedEngine> {
    guard.as_mut().ok_or_else(|| anyhow!("model did not load"))
}

#[cfg(local_engines)]
fn boxed_error(err: Box<dyn std::error::Error>) -> anyhow::Error {
    anyhow!(err.to_string())
}

fn load_engine(resolved: &ResolvedModel) -> Result<EngineInstance> {
    match resolved.engine {
        ModelEngine::Whisper => {
            #[cfg(feature = "whisper")]
            {
                use crate::engines::whisper::{
                    WhisperEngine, WhisperModelParams, dtw_preset_for_variant,
                };

                let mut engine = WhisperEngine::new();
                let params = WhisperModelParams {
                    dtw_preset: resolved.variant.as_deref().and_then(dtw_preset_for_variant),
                    ..Default::default()
                };
                engine
                    .load_model_with_params(&resolved.path, params)
                    .map_err(boxed_error)?;
                Ok(EngineInstance::Whisper(engine))
            }
            #[cfg(not(feature = "whisper"))]
            {
                Err(anyhow!("Whisper support is not enabled"))
            }
        }
        ModelEngine::Parakeet => {
            #[cfg(nvidia_engines)]
            {
                use crate::engines::parakeet::{ParakeetEngine, ParakeetModelParams};

                let mut engine = ParakeetEngine::new();
                engine
                    .load_model_with_params(
                        &resolved.path,
                        ParakeetModelParams::int8_with_layout(resolved.layout),
                    )
                    .map_err(boxed_error)?;
                Ok(EngineInstance::Parakeet(engine))
            }
            #[cfg(not(nvidia_engines))]
            {
                Err(anyhow!(
                    "NVIDIA speech support is not enabled on this build"
                ))
            }
        }
        ModelEngine::Nemotron => {
            #[cfg(nvidia_engines)]
            {
                let mut engine = crate::engines::nemotron::NemotronEngine::new();
                engine.load_model(&resolved.path).map_err(boxed_error)?;
                Ok(EngineInstance::Nemotron(engine))
            }
            #[cfg(not(nvidia_engines))]
            {
                Err(anyhow!(
                    "NVIDIA speech support is not enabled on this build"
                ))
            }
        }
        ModelEngine::Transcribe => {
            #[cfg(transcribe_engine)]
            {
                use crate::engines::transcribe::{TranscribeEngine, TranscribeModelParams};

                let mut engine = TranscribeEngine::new();
                let coreml_encoder = TranscribeEngine::companion_for(&resolved.path);
                let backend = if coreml_encoder.is_some()
                    && resolved
                        .variant
                        .as_deref()
                        .is_some_and(|v| v.starts_with("parakeet-"))
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

#[cfg_attr(not(local_engines), allow(unused_variables))]
fn transcribe_with_engine(
    engine: &mut EngineInstance,
    request: TranscribeRequest,
) -> Result<TranscriptionWithDuration> {
    match engine {
        #[cfg(feature = "whisper")]
        EngineInstance::Whisper(engine) => {
            let wants_timestamps = request.timestamps || request.timestamp_granularity.is_some();
            let params = crate::engines::whisper::WhisperInferenceParams {
                dictionary: if request.prompt.is_some() {
                    Vec::new()
                } else {
                    request.dictionary.clone()
                },
                language: request.language,
                initial_prompt: combined_prompt(request.prompt, &request.dictionary),
                print_timestamps: wants_timestamps,
                word_timestamps: request.timestamp_granularity == Some(TimestampGranularity::Word),
                ..Default::default()
            };
            transcribe_audio(engine, request.audio, Some(params))
        }
        #[cfg(nvidia_engines)]
        EngineInstance::Parakeet(engine) => {
            use crate::engines::parakeet::TimestampGranularity as Granularity;

            let timestamp_granularity = match request.timestamp_granularity {
                Some(TimestampGranularity::Word) => Granularity::Word,
                Some(TimestampGranularity::Segment) => Granularity::Segment,
                None if request.timestamps => Granularity::Segment,
                None => Granularity::Token,
            };
            let params = crate::engines::parakeet::ParakeetInferenceParams {
                timestamp_granularity,
                language: request.language,
                dictionary: request.dictionary,
            };
            transcribe_audio(engine, request.audio, Some(params))
        }
        #[cfg(nvidia_engines)]
        EngineInstance::Nemotron(engine) => {
            let params = crate::engines::nemotron::NemotronInferenceParams {
                language: request.language,
            };
            transcribe_audio(engine, request.audio, Some(params))
        }
        #[cfg(apple_speech_engine)]
        EngineInstance::Apple(engine) => {
            let params = crate::engines::apple::AppleInferenceParams {
                language: request.language,
                long_form: request.timestamps || request.timestamp_granularity.is_some(),
                dictionary: request.dictionary,
            };
            transcribe_audio(engine, request.audio, Some(params))
        }
        #[cfg(transcribe_engine)]
        EngineInstance::Transcribe(engine) => {
            let params = crate::engines::transcribe::TranscribeInferenceParams {
                language: request.language,
                dictionary: request.dictionary,
                timestamps: request.timestamps || request.timestamp_granularity.is_some(),
            };
            transcribe_audio(engine, request.audio, Some(params))
        }
        #[allow(unreachable_patterns)]
        _ => Err(anyhow!("No speech engine support is enabled")),
    }
}

#[cfg(feature = "whisper")]
fn combined_prompt(prompt: Option<String>, dictionary: &[String]) -> Option<String> {
    match (
        prompt,
        crate::dictionary::build_dictionary_prompt(dictionary),
    ) {
        (Some(prompt), Some(dictionary_prompt)) => Some(format!("{prompt}\n\n{dictionary_prompt}")),
        (Some(prompt), None) => Some(prompt),
        (None, Some(dictionary_prompt)) => Some(dictionary_prompt),
        (None, None) => None,
    }
}

#[cfg(local_engines)]
fn transcribe_audio<E: TranscriptionEngine>(
    engine: &mut E,
    audio: AudioInput,
    params: Option<E::InferenceParams>,
) -> Result<TranscriptionWithDuration> {
    let prepared = prepare_audio(audio)?;
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

#[cfg(local_engines)]
fn prepare_audio(audio: AudioInput) -> Result<PreparedAudio> {
    const MIN_SAMPLES: usize = 16_000;
    const EXTRA_PADDING: usize = 4_000;

    let (mut samples, source_sample_rate, source_sample_count) = match audio {
        AudioInput::WavPath(path) => {
            let samples = crate::audio::read_audio_samples(&path).map_err(boxed_error)?;
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

#[cfg(local_engines)]
fn audio_duration_ms(sample_count: usize, sample_rate: u32) -> u128 {
    if sample_rate == 0 {
        return 0;
    }
    (sample_count as u128 * 1000) / u128::from(sample_rate)
}

#[cfg(all(test, local_engines))]
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
