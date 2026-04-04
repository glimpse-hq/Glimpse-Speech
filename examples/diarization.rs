#[cfg(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
use std::path::PathBuf;

#[cfg(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
use glimpse_speech::{
    diarization::attribute_speakers,
    engines::{
        parakeet::{
            ParakeetEngine, ParakeetInferenceParams, ParakeetModelParams, TimestampGranularity,
        },
        sortformer::{SortformerEngine, SortformerModelParams},
    },
    SpeakerDiarizationEngine, TranscriptionEngine,
};

#[cfg(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = PathBuf::from(
        args.get(1)
            .ok_or_else(|| {
                io_error(
                    "Usage: cargo run --example diarization --features nvidia -- <parakeet-model-dir> <sortformer.onnx> <audio.wav>",
                )
            })?
            .as_str(),
    );
    let sortformer_model = PathBuf::from(
        args.get(2)
            .ok_or_else(|| {
                io_error(
                    "Usage: cargo run --example diarization --features nvidia -- <parakeet-model-dir> <sortformer.onnx> <audio.wav>",
                )
            })?
            .as_str(),
    );
    let wav_path = PathBuf::from(
        args.get(3)
            .ok_or_else(|| {
                io_error(
                    "Usage: cargo run --example diarization --features nvidia -- <parakeet-model-dir> <sortformer.onnx> <audio.wav>",
                )
            })?
            .as_str(),
    );

    let mut transcription_engine = ParakeetEngine::new();
    transcription_engine.load_model_with_params(&model_dir, ParakeetModelParams::int8())?;

    let mut diarization_engine = SortformerEngine::new();
    diarization_engine
        .load_model_with_params(&sortformer_model, SortformerModelParams::callhome())?;

    if let Some(info) = diarization_engine.runtime_info() {
        eprintln!(
            "Sortformer config: chunk_len={}, fifo_len={}, spkcache_len={}, right_context={}, latency={:.2}s",
            info.chunk_len,
            info.fifo_len,
            info.spkcache_len,
            info.right_context,
            info.latency()
        );
    }

    let transcription = transcription_engine.transcribe_file(
        &wav_path,
        Some(ParakeetInferenceParams {
            timestamp_granularity: TimestampGranularity::Segment,
            ..Default::default()
        }),
    )?;
    let speaker_segments = diarization_engine.diarize_file(&wav_path)?;
    let result = attribute_speakers(transcription, speaker_segments)?;

    for segment in result.segments {
        let speaker = segment
            .speaker_id
            .map(|speaker_id| format!("Speaker {speaker_id}"))
            .unwrap_or_else(|| "UNKNOWN".to_string());

        println!(
            "[{:.2}s - {:.2}s] {}: {}",
            segment.start, segment.end, speaker, segment.text
        );
    }

    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}

#[cfg(not(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
)))]
fn main() {
    eprintln!("The diarization example is unavailable on Intel macOS builds.");
}
