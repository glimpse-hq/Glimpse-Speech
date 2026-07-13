use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{anyhow, Result};

#[cfg(any(
    feature = "whisper",
    all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    )
))]
use crate::TranscriptionEngine;

use crate::{
    models::{
        InstallOptions, InstallSpec, ModelEngine, ModelInstallManager, ModelStatus, ResolvedModel,
    },
    TimestampGranularity, Transcription, TranscriptionResult,
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

#[cfg(any(
    feature = "whisper",
    all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    )
))]
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
    Whisper {
        engine: crate::engines::whisper::WhisperEngine,
    },
    #[cfg(all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    ))]
    Parakeet {
        engine: crate::engines::parakeet::ParakeetEngine,
    },
    #[cfg(all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    ))]
    Nemotron {
        engine: crate::engines::nemotron::NemotronEngine,
    },
}

impl EngineInstance {
    #[cfg(all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    ))]
    fn streaming_transcribe_chunk(&mut self, chunk: &[f32]) -> Result<String> {
        match self {
            EngineInstance::Parakeet { engine } => {
                engine
                    .transcribe_chunk(chunk)
                    .map_err(|err| anyhow!(err.to_string()))?;
                Ok(engine.get_transcript())
            }
            EngineInstance::Nemotron { engine } => {
                engine
                    .transcribe_chunk(chunk)
                    .map_err(|err| anyhow!(err.to_string()))?;
                Ok(engine.get_transcript())
            }
            #[cfg(feature = "whisper")]
            EngineInstance::Whisper { .. } => Err(anyhow!(
                "Streaming is only supported with Nemotron or unified Parakeet models"
            )),
        }
    }

    #[cfg(all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    ))]
    fn streaming_reset(&mut self) {
        match self {
            EngineInstance::Parakeet { engine } => engine.reset(),
            EngineInstance::Nemotron { engine } => engine.reset(),
            #[cfg(feature = "whisper")]
            EngineInstance::Whisper { .. } => {}
        }
    }

    #[cfg(all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    ))]
    fn streaming_get_transcript(&self) -> Option<String> {
        match self {
            EngineInstance::Parakeet { engine } => Some(engine.get_transcript()),
            EngineInstance::Nemotron { engine } => Some(engine.get_transcript()),
            #[cfg(feature = "whisper")]
            EngineInstance::Whisper { .. } => None,
        }
    }
}

impl SpeechService {
    pub fn new(config: SpeechConfig) -> Self {
        crate::silence_native_logs();
        Self {
            model_manager: ModelInstallManager::new(config.model_cache_dir),
            resolver: config.resolver,
            loose_engine: ModelEngine::Whisper,
            loaded: Mutex::new(None),
        }
    }

    pub fn new_loose_with_engine(model_cache_dir: PathBuf, engine: ModelEngine) -> Self {
        crate::silence_native_logs();
        Self {
            model_manager: ModelInstallManager::new(model_cache_dir),
            resolver: Arc::new(|_| None),
            loose_engine: engine,
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
        let loaded = guard
            .as_mut()
            .ok_or_else(|| anyhow!("model did not load"))?;
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

        Ok(Transcription {
            text: transcription.result.text,
            segments: transcription.result.segments,
            words: transcription.result.words,
            model_id: resolved_id,
            language: transcription.result.language.or(requested_language),
            duration_ms: transcription.audio_duration_ms,
        })
    }

    pub fn preload_and_warm(&self, model_id: &str) -> Result<()> {
        let total_started = Instant::now();
        self.ensure_loaded(model_id)?;
        let lock_started = Instant::now();
        let mut guard = self.lock_loaded()?;
        let lock_wait = lock_started.elapsed();
        let loaded = guard
            .as_mut()
            .ok_or_else(|| anyhow!("model did not load"))?;
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
        let _ = transcribe_with_engine(
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
        self.loaded
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    pub fn loaded_model_id(&self) -> Option<String> {
        self.loaded
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|loaded| loaded.model_id.clone()))
    }

    #[cfg(all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    ))]
    pub fn streaming_transcribe_chunk(&self, model_id: &str, chunk: &[f32]) -> Result<String> {
        self.ensure_loaded(model_id)?;
        let mut guard = self.lock_loaded()?;
        let loaded = guard
            .as_mut()
            .ok_or_else(|| anyhow!("model did not load"))?;
        loaded.engine.streaming_transcribe_chunk(chunk)
    }

    #[cfg(all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    ))]
    pub fn streaming_reset(&self) {
        if let Ok(mut guard) = self.loaded.lock() {
            if let Some(loaded) = guard.as_mut() {
                loaded.engine.streaming_reset();
            }
        }
    }

    #[cfg(all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    ))]
    pub fn streaming_get_transcript(&self) -> String {
        self.loaded
            .lock()
            .ok()
            .and_then(|guard| {
                guard
                    .as_ref()
                    .and_then(|loaded| loaded.engine.streaming_get_transcript())
            })
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
            .map(|loaded| loaded.model_id != resolved.id || loaded.path != resolved.path)
            .unwrap_or(true);

        if should_reload {
            let load_started = Instant::now();
            let path = resolved.path.display().to_string();
            let bytes = std::fs::metadata(&resolved.path)
                .ok()
                .map(|metadata| metadata.len());
            tracing::info!(
                "[SpeechService] load start model={} engine={} path={} bytes={:?}",
                resolved.id,
                resolved.engine,
                path,
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

    fn lock_loaded(&self) -> Result<std::sync::MutexGuard<'_, Option<LoadedEngine>>> {
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

fn load_engine(resolved: &crate::models::ResolvedModel) -> Result<EngineInstance> {
    match resolved.engine {
        ModelEngine::Whisper => {
            #[cfg(feature = "whisper")]
            {
                use crate::engines::whisper::{dtw_preset_for_variant, WhisperModelParams};

                let mut engine = crate::engines::whisper::WhisperEngine::new();
                let params = WhisperModelParams {
                    dtw_preset: resolved.variant.as_deref().and_then(dtw_preset_for_variant),
                    ..Default::default()
                };
                engine
                    .load_model_with_params(&resolved.path, params)
                    .map_err(|err| anyhow!(err.to_string()))?;
                Ok(EngineInstance::Whisper { engine })
            }

            #[cfg(not(feature = "whisper"))]
            {
                Err(anyhow!("Whisper support is not enabled"))
            }
        }
        ModelEngine::Parakeet => {
            #[cfg(all(
                feature = "nvidia",
                not(all(target_os = "macos", target_arch = "x86_64"))
            ))]
            {
                let mut engine = crate::engines::parakeet::ParakeetEngine::new();
                engine
                    .load_model_with_params(
                        &resolved.path,
                        crate::engines::parakeet::ParakeetModelParams::int8_with_layout(
                            resolved.layout,
                        ),
                    )
                    .map_err(|err| anyhow!(err.to_string()))?;
                Ok(EngineInstance::Parakeet { engine })
            }

            #[cfg(not(all(
                feature = "nvidia",
                not(all(target_os = "macos", target_arch = "x86_64"))
            )))]
            {
                Err(anyhow!(
                    "NVIDIA speech support is not enabled on this build"
                ))
            }
        }
        ModelEngine::Nemotron => {
            #[cfg(all(
                feature = "nvidia",
                not(all(target_os = "macos", target_arch = "x86_64"))
            ))]
            {
                let mut engine = crate::engines::nemotron::NemotronEngine::new();
                engine
                    .load_model(&resolved.path)
                    .map_err(|err| anyhow!(err.to_string()))?;
                Ok(EngineInstance::Nemotron { engine })
            }

            #[cfg(not(all(
                feature = "nvidia",
                not(all(target_os = "macos", target_arch = "x86_64"))
            )))]
            {
                Err(anyhow!(
                    "NVIDIA speech support is not enabled on this build"
                ))
            }
        }
    }
}

fn transcribe_with_engine(
    engine: &mut EngineInstance,
    _request: TranscribeRequest,
) -> Result<TranscriptionWithDuration> {
    match engine {
        #[cfg(feature = "whisper")]
        EngineInstance::Whisper { engine } => {
            let params = Some(crate::engines::whisper::WhisperInferenceParams {
                dictionary: if _request.prompt.is_some() {
                    Vec::new()
                } else {
                    _request.dictionary.clone()
                },
                language: _request.language,
                initial_prompt: combined_prompt(_request.prompt, &_request.dictionary),
                print_timestamps: _request.timestamps || _request.timestamp_granularity.is_some(),
                word_timestamps: matches!(
                    _request.timestamp_granularity,
                    Some(TimestampGranularity::Word)
                ),
                ..Default::default()
            });
            transcribe_audio(engine, _request.audio, params)
        }
        #[cfg(all(
            feature = "nvidia",
            not(all(target_os = "macos", target_arch = "x86_64"))
        ))]
        EngineInstance::Parakeet { engine } => {
            let timestamp_granularity = match _request.timestamp_granularity {
                Some(TimestampGranularity::Word) => {
                    crate::engines::parakeet::TimestampGranularity::Word
                }
                Some(TimestampGranularity::Segment) => {
                    crate::engines::parakeet::TimestampGranularity::Segment
                }
                None if _request.timestamps => {
                    crate::engines::parakeet::TimestampGranularity::Segment
                }
                None => crate::engines::parakeet::TimestampGranularity::Token,
            };
            let params = Some(crate::engines::parakeet::ParakeetInferenceParams {
                timestamp_granularity,
                language: _request.language,
                dictionary: _request.dictionary,
            });
            transcribe_audio(engine, _request.audio, params)
        }
        #[cfg(all(
            feature = "nvidia",
            not(all(target_os = "macos", target_arch = "x86_64"))
        ))]
        EngineInstance::Nemotron { engine } => transcribe_audio(
            engine,
            _request.audio,
            Some(crate::engines::nemotron::NemotronInferenceParams {
                language: _request.language,
            }),
        ),
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

#[cfg(any(
    feature = "whisper",
    all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    )
))]
fn transcribe_audio<E: TranscriptionEngine>(
    engine: &mut E,
    audio: AudioInput,
    params: Option<E::InferenceParams>,
) -> Result<TranscriptionWithDuration> {
    let prepared = prepare_audio(audio)?;
    let result = engine
        .transcribe_samples(prepared.samples, params)
        .map_err(|err| anyhow!(err.to_string()))?;
    Ok(TranscriptionWithDuration {
        result,
        audio_duration_ms: prepared.duration_ms,
    })
}

#[cfg(any(
    feature = "whisper",
    all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    )
))]
fn prepare_audio(audio: AudioInput) -> Result<PreparedAudio> {
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
            let sample_count = samples.len();
            if sample_rate == 16_000 {
                let normalized = samples
                    .into_iter()
                    .map(|sample| sample as f32 / 32_768.0)
                    .collect();
                (normalized, sample_rate, sample_count)
            } else {
                // Normalize and resample in a single pass over one output
                // buffer instead of materializing an intermediate f32 copy.
                (
                    crate::audio::resample_i16_to_f32(&samples, sample_rate, 16_000),
                    sample_rate,
                    sample_count,
                )
            }
        }
    };

    const MIN_SAMPLES: usize = 16_000;
    const EXTRA_PADDING: usize = 4_000;

    let padding_needed = MIN_SAMPLES.saturating_sub(samples.len()) + EXTRA_PADDING;
    samples.extend(std::iter::repeat_n(0.0f32, padding_needed));
    Ok(PreparedAudio {
        samples,
        duration_ms: audio_duration_ms(source_sample_count, source_sample_rate),
    })
}

#[cfg(any(
    feature = "whisper",
    all(
        feature = "nvidia",
        not(all(target_os = "macos", target_arch = "x86_64"))
    )
))]
fn audio_duration_ms(sample_count: usize, sample_rate: u32) -> u128 {
    if sample_rate == 0 {
        return 0;
    }
    ((sample_count as u128) * 1000) / sample_rate as u128
}

#[cfg(all(
    test,
    any(
        feature = "whisper",
        all(
            feature = "nvidia",
            not(all(target_os = "macos", target_arch = "x86_64"))
        )
    )
))]
mod prepare_tests {
    use super::{prepare_audio, AudioInput};

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
