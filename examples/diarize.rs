//! Diarize a mono PCM16 WAV with the Nemotron-3 Diarization GGUF
//! (https://huggingface.co/Glimpse-Dictation/Nemotron-3-Diarization-gguf).
//!
//!     cargo run --example diarize --features transcribe -- \
//!         models/nemotron-3-diarization-Q8_0.gguf samples/meeting.wav [cpu]

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::{path::PathBuf, time::Instant};

    use glimpse_speech::diarization::diarize;

    let args: Vec<String> = std::env::args().collect();
    let model = PathBuf::from(
        args.get(1)
            .map_or("models/nemotron-3-diarization-Q8_0.gguf", String::as_str),
    );
    let wav = PathBuf::from(args.get(2).map_or("samples/meeting.wav", String::as_str));
    let use_gpu = args.get(3).is_none_or(|arg| arg != "cpu");

    let mut reader = hound::WavReader::open(&wav)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.bits_per_sample != 16 {
        return Err("expected a mono PCM16 WAV".into());
    }
    let samples = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
    let audio_seconds = samples.len() as f32 / spec.sample_rate as f32;

    let started = Instant::now();
    let turns = diarize(&model, &samples, spec.sample_rate, use_gpu)?;
    let elapsed = started.elapsed().as_secs_f32();
    for turn in &turns {
        println!(
            "[{:.2}s - {:.2}s] speaker {}",
            turn.start_ms as f32 / 1000.0,
            turn.end_ms as f32 / 1000.0,
            turn.speaker
        );
    }
    println!(
        "{} turns, {:.1}s audio in {elapsed:.2}s ({:.0}x realtime)",
        turns.len(),
        audio_seconds,
        audio_seconds / elapsed
    );
    Ok(())
}
