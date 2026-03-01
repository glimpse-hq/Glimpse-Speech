use std::path::PathBuf;

use glimpse_speech::{
    engines::whisper::{WhisperEngine, WhisperInferenceParams},
    TranscriptionEngine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let model_path = PathBuf::from(
        args.get(1)
            .map(|value| value.as_str())
            .unwrap_or("models/whisper-medium-q4_1.bin"),
    );
    let wav_path = PathBuf::from(
        args.get(2)
            .map(|value| value.as_str())
            .unwrap_or("samples/dots.wav"),
    );

    let mut engine = WhisperEngine::new();
    engine.load_model(&model_path)?;

    let result = engine.transcribe_file(
        &wav_path,
        Some(WhisperInferenceParams {
            dictionary: vec!["Glimpse".to_string(), "Parakeet".to_string()],
            ..Default::default()
        }),
    )?;

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
