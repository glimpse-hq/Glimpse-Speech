//! Diarize a mono PCM16 WAV with the Nemotron-3 Diarization GGUF
//! (https://huggingface.co/Glimpse-Dictation/Nemotron-3-Diarization-gguf).
//! `--live` feeds a 16 kHz WAV in 100 ms pieces through `LiveDiarizer`.
//!
//!     cargo run --example diarize --features transcribe -- \
//!         models/nemotron-3-diarization-Q8_0.gguf samples/meeting.wav [cpu] [--live]

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::{path::PathBuf, time::Instant};

    use glimpse_speech::diarization::{LiveDiarizer, diarize};

    let live = std::env::args().any(|arg| arg == "--live");
    let args: Vec<String> = std::env::args().filter(|arg| arg != "--live").collect();
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
    let turns = if live {
        if spec.sample_rate != 16_000 {
            return Err("--live expects a 16 kHz WAV".into());
        }
        let audio: Vec<f32> = samples.iter().map(|&s| f32::from(s) / 32768.0).collect();
        let mut diarizer = LiveDiarizer::new(&model, use_gpu)?;
        let mut slowest = 0.0f32;
        for (i, piece) in audio.chunks(1600).enumerate() {
            let fed = Instant::now();
            let so_far = diarizer.feed(piece)?;
            slowest = slowest.max(fed.elapsed().as_secs_f32());
            if i % 50 == 49 {
                let open = so_far
                    .turns
                    .iter()
                    .filter(|turn| turn.end_ms > so_far.settled_ms)
                    .count();
                println!(
                    "{:.1}s fed: {} turns ({open} open), settled to {:.2}s",
                    (i + 1) as f32 / 10.0,
                    so_far.turns.len(),
                    so_far.settled_ms as f32 / 1000.0
                );
            }
        }
        println!("slowest feed {:.0} ms", slowest * 1000.0);
        diarizer.finish()?
    } else {
        diarize(&model, &samples, spec.sample_rate, use_gpu)?
    };
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
