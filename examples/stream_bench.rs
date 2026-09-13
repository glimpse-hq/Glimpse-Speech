//! Feed a WAV through the Unified streaming path in fixed chunks and time each call.
#[cfg(nvidia_engines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use glimpse_speech::TranscriptionEngine;
    use glimpse_speech::engines::parakeet::{ParakeetEngine, ParakeetModelParams};
    use glimpse_speech::models::ModelLayout;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let model_dir = &args[1];
    let chunk_ms: usize = args[2].parse()?;
    let mut engine = ParakeetEngine::new();
    let t0 = Instant::now();
    engine.load_model_with_params(
        std::path::Path::new(model_dir),
        ParakeetModelParams::int8_with_layout(ModelLayout::ParakeetUnified),
    )?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    for wav in &args[3..] {
        let mut reader = hound::WavReader::open(wav)?;
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect(),
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        };
        let mut samples = samples;
        if let Ok(pad) = std::env::var("STREAM_BENCH_PAD_MS") {
            samples.extend(std::iter::repeat(0.0f32).take(pad.parse::<usize>().unwrap() * 16));
        }
        engine.reset();
        let chunk = chunk_ms * 16;
        let mut per_chunk = Vec::new();
        let t = Instant::now();
        for c in samples.chunks(chunk) {
            let s = Instant::now();
            engine.transcribe_chunk(c)?;
            per_chunk.push(s.elapsed().as_secs_f64() * 1000.0);
        }
        let total = t.elapsed().as_secs_f64() * 1000.0;
        let text = engine.get_transcript();
        let max = per_chunk.iter().cloned().fold(0.0, f64::max);
        let mean = per_chunk.iter().sum::<f64>() / per_chunk.len() as f64;
        println!(
            "{}",
            serde_json::json!({"file": wav, "text": text.trim(), "chunks": per_chunk.len(), "chunk_mean_ms": mean, "chunk_max_ms": max, "total_ms": total, "audio_ms": samples.len() as f64 / 16.0, "load_ms": load_ms})
        );
    }
    Ok(())
}

#[cfg(not(nvidia_engines))]
fn main() {
    eprintln!("NVIDIA engines are unavailable on this target.");
}
