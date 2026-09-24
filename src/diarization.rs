//! Speaker diarization through transcribe.cpp with NVIDIA Streaming Sortformer:
//! Nemotron-3 Diarization (up to 8 speakers) or Sortformer v2.1 (up to 4).
//! Produces who-spoke-when turns, no text.

use std::{mem::ManuallyDrop, path::Path, ptr::NonNull};

use transcribe_cpp::{
    Backend, Model, ModelOptions, RunExtension, RunOptions, Session, SessionOptions,
    SortformerLiveOptions, SortformerPreset, SortformerStreamOptions, SpeakerSegment,
    StreamExtension, StreamOptions, TimestampKind,
};

type Error = Box<dyn std::error::Error + Send + Sync>;

const SAMPLE_RATE: u32 = 16_000;

/// Longest input accepted. transcribe.cpp holds the whole clip and its
/// spectrogram in memory, measured at about 1.5 GB peak per hour of audio.
/// A live stream keeps only per-frame speaker activity but shares the cap.
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
    if sample_rate == 0 {
        return Err(error("sample rate must be greater than zero"));
    }
    if sample_rate > crate::audio::MAX_SAMPLE_RATE {
        return Err(error(format!("unsupported sample rate {sample_rate} Hz")));
    }
    if samples.len() as u64 / u64::from(sample_rate) > MAX_AUDIO_SECONDS {
        return Err(too_long());
    }
    let audio = crate::audio::resample_i16_to_f32(samples, sample_rate, SAMPLE_RATE);
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let mut session = open_session(model_path, use_gpu)?;
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
    Ok(to_turns(&transcript.speaker_segments, duration_ms))
}

/// Turns so far from [`LiveDiarizer::feed`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LiveTurns {
    pub turns: Vec<SpeakerTurn>,
    /// Turns ending after this can still grow or close; the rest are final.
    pub settled_ms: u64,
}

/// Diarizes 16 kHz mono audio pushed while it is recorded. Nemotron-3
/// Diarization only; Sortformer v2.1 has no live path.
pub struct LiveDiarizer {
    stream: ManuallyDrop<transcribe_cpp::Stream<'static>>,
    // Borrowed by `stream`. A raw pointer, since moving a `Box` would assert
    // unique access while the stream holds it.
    session: NonNull<Session>,
    received: u64,
    settled_ms: u64,
}

// SAFETY: the stream is the only user of the boxed session and both move
// together; `Stream` and `Session` are each `Send`.
unsafe impl Send for LiveDiarizer {}

impl LiveDiarizer {
    pub fn new(model_path: &Path, use_gpu: bool) -> Result<Self, Error> {
        let session = open_session(model_path, use_gpu)?;
        if session.model().arch() != "nemotron3_diar" {
            return Err(error(
                "live diarization needs Nemotron-3 Diarization, not Sortformer v2.1",
            ));
        }
        let session = NonNull::from(Box::leak(Box::new(session)));
        // SAFETY: `session` is a heap allocation nothing else touches until
        // `Drop` frees it, after dropping the stream that borrows it. The
        // 'static borrow never leaves this struct.
        let stream = unsafe { &mut *session.as_ptr() }.stream(
            &RunOptions {
                timestamps: TimestampKind::None,
                ..Default::default()
            },
            &StreamOptions {
                family: Some(StreamExtension::SortformerLive(SortformerLiveOptions {
                    preset: Some(SortformerPreset::LowLatency),
                })),
                ..Default::default()
            },
        );
        match stream {
            Ok(stream) => Ok(Self {
                stream: ManuallyDrop::new(stream),
                session,
                received: 0,
                settled_ms: 0,
            }),
            Err(err) => {
                // SAFETY: the failed `stream` call left no borrow behind.
                drop(unsafe { Box::from_raw(session.as_ptr()) });
                Err(transcribe_error(err))
            }
        }
    }

    /// Takes 16 kHz mono samples of any length and returns all turns so far.
    pub fn feed(&mut self, samples: &[f32]) -> Result<LiveTurns, Error> {
        if !samples.is_empty() {
            let received = self.received + samples.len() as u64;
            if received / u64::from(SAMPLE_RATE) > MAX_AUDIO_SECONDS {
                return Err(too_long());
            }
            self.received = received;
            let update = self.stream.feed(samples).map_err(transcribe_error)?;
            // Open rows end exactly at the committed frontier; settling 1 ms
            // before it keeps them provisional.
            self.settled_ms = (update.audio_committed_ms.max(0) as u64).saturating_sub(1);
        }
        Ok(LiveTurns {
            turns: to_turns(&self.stream.snapshot().speaker_segments, u64::MAX),
            settled_ms: self.settled_ms,
        })
    }

    /// Flushes buffered audio and returns the final turns.
    pub fn finish(mut self) -> Result<Vec<SpeakerTurn>, Error> {
        let update = self.stream.finalize().map_err(transcribe_error)?;
        Ok(to_turns(
            &self.stream.snapshot().speaker_segments,
            update.input_received_ms.max(0) as u64,
        ))
    }
}

impl Drop for LiveDiarizer {
    fn drop(&mut self) {
        // SAFETY: the stream goes first, ending its borrow of the session,
        // and neither is used again.
        unsafe {
            ManuallyDrop::drop(&mut self.stream);
            drop(Box::from_raw(self.session.as_ptr()));
        }
    }
}

fn open_session(model_path: &Path, use_gpu: bool) -> Result<Session, Error> {
    if !model_path.is_file() {
        return Err(error(format!(
            "diarization model file not found: {}",
            model_path.display()
        )));
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
    model
        .session_with(&SessionOptions {
            n_threads: crate::engines::inference_threads() as i32,
            ..Default::default()
        })
        .map_err(transcribe_error)
}

fn to_turns(segments: &[SpeakerSegment], duration_ms: u64) -> Vec<SpeakerTurn> {
    let mut turns: Vec<SpeakerTurn> = segments
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
    turns
}

fn too_long() -> Error {
    error(format!(
        "audio is too long to diarize: the limit is {} hours",
        MAX_AUDIO_SECONDS / 3600
    ))
}

fn error(message: impl Into<String>) -> Error {
    std::io::Error::other(message.into()).into()
}

fn transcribe_error(err: transcribe_cpp::Error) -> Error {
    error(format!("transcribe.cpp error: {err}"))
}
