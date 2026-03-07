#[cfg(all(
    feature = "parakeet",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
use std::path::PathBuf;

#[cfg(all(
    feature = "parakeet",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
use glimpse_speech::{
    engines::parakeet::{ParakeetEngine, ParakeetInferenceParams, ParakeetModelParams},
    TranscriptionEngine,
};

#[cfg(all(
    feature = "parakeet",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let model_dir = PathBuf::from(
        args.get(1)
            .map(|value| value.as_str())
            .unwrap_or("models/parakeet-tdt-0.6b-v3-onnx-int8"),
    );
    let wav_path = PathBuf::from(
        args.get(2)
            .map(|value| value.as_str())
            .unwrap_or("samples/dots.wav"),
    );

    let mut engine = ParakeetEngine::new();
    engine.load_model_with_params(&model_dir, ParakeetModelParams::int8())?;

    let result = engine.transcribe_file(&wav_path, Some(ParakeetInferenceParams::default()))?;

    println!("{}", result.text);

    if let Some(segments) = result.segments {
        for segment in segments {
            println!(
                "[{:.2}s - {:.2}s] {}",
                segment.start, segment.end, segment.text
            );
        }
    }

    Ok(())
}

#[cfg(not(all(
    feature = "parakeet",
    not(all(target_os = "macos", target_arch = "x86_64"))
)))]
fn main() {
    eprintln!("The parakeet example is unavailable on Intel macOS builds.");
}
