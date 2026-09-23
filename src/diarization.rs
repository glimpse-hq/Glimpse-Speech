//! Speaker diarization through transcribe.cpp with NVIDIA Streaming Sortformer:
//! Nemotron-3 Diarization (up to 8 speakers) or Sortformer v2.1 (up to 4).
//! Produces who-spoke-when turns, no text.

use std::path::Path;

use transcribe_cpp::{
    Backend, Model, ModelOptions, RunExtension, RunOptions, SessionOptions, SortformerPreset,
    SortformerStreamOptions, TimestampKind,
};

type Error = Box<dyn std::error::Error + Send + Sync>;

const SAMPLE_RATE: u32 = 16_000;

/// Longest input accepted. transcribe.cpp holds the whole clip and its
/// spectrogram in memory, measured at about 1.5 GB peak per hour of audio.
const MAX_AUDIO_SECONDS: u64 = 3 * 60 * 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeakerTurn {
    pub start_ms: u64,
    pub end_ms: u64,
    /// 1-based, in order of first appearance, at most 8 (4 for Sortformer v2.1).
    pub speaker: u32,
}

/// Runs the diarizer over a whole clip of mono samples at any rate, resampled to 16 kHz internally.
pub fn diarize(
    model_path: &Path,
    samples: &[i16],
    sample_rate: u32,
    use_gpu: bool,
) -> Result<Vec<SpeakerTurn>, Error> {
    if !model_path.is_file() {
        return Err(error(format!(
            "diarization model file not found: {}",
            model_path.display()
        )));
    }
    if sample_rate == 0 {
        return Err(error("sample rate must be greater than zero"));
    }
    if sample_rate > crate::audio::MAX_SAMPLE_RATE {
        return Err(error(format!("unsupported sample rate {sample_rate} Hz")));
    }
    if samples.len() as u64 / u64::from(sample_rate) > MAX_AUDIO_SECONDS {
        return Err(error(format!(
            "audio is too long to diarize: the limit is {} hours",
            MAX_AUDIO_SECONDS / 3600
        )));
    }
    let audio = crate::audio::resample_i16_to_f32(samples, sample_rate, SAMPLE_RATE);
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    crate::silence_transcribe_cpp_logs();
    let model = Model::load_with(
        model_path,
        &ModelOptions {
            backend: if use_gpu { Backend::Auto } else { Backend::Cpu },
            device: None,
        },
    )
    .map_err(transcribe_error)?;
    if !matches!(model.arch().as_str(), "sortformer" | "nemotron3_diar") {
        return Err(error(format!(
            "{} is not a Sortformer diarization model",
            model_path.display()
        )));
    }
    let mut session = model
        .session_with(&SessionOptions {
            n_threads: crate::engines::inference_threads() as i32,
            ..Default::default()
        })
        .map_err(transcribe_error)?;
    let transcript = session
        .run(
            &audio,
            &RunOptions {
                timestamps: TimestampKind::None,
                // The published offline-file operating point.
                family: Some(RunExtension::Sortformer(SortformerStreamOptions {
                    preset: Some(SortformerPreset::VeryHighLatency),
                })),
                ..Default::default()
            },
        )
        .map_err(transcribe_error)?;

    let duration_ms = audio.len() as u64 * 1000 / u64::from(SAMPLE_RATE);
    let mut turns: Vec<SpeakerTurn> = transcript
        .speaker_segments
        .iter()
        .filter(|segment| segment.speaker_id >= 1)
        .map(|segment| SpeakerTurn {
            start_ms: (segment.t0_ms.max(0) as u64).min(duration_ms),
            end_ms: (segment.t1_ms.max(0) as u64).min(duration_ms),
            speaker: segment.speaker_id as u32,
        })
        .filter(|turn| turn.end_ms > turn.start_ms)
        .collect();
    turns.sort_by_key(|turn| (turn.start_ms, turn.end_ms, turn.speaker));
    Ok(turns)
}

fn error(message: impl Into<String>) -> Error {
    std::io::Error::other(message.into()).into()
}

fn transcribe_error(err: transcribe_cpp::Error) -> Error {
    error(format!("transcribe.cpp error: {err}"))
}
