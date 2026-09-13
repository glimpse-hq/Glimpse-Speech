//! Transcribe a WAV with a transcribe.cpp GGUF model.
//!
//!     cargo run --example transcribe --features transcribe -- \
//!         models/Qwen3-ASR-0.6B-Q8_0.gguf samples/jfk.wav [language] [dictionary words...]
//!
//! A `<gguf stem>-encoder.mlmodelc` next to the GGUF is used on Apple Silicon.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{path::PathBuf, time::Instant};

    use glimpse_speech::{
        TranscriptionEngine,
        engines::transcribe::{TranscribeEngine, TranscribeInferenceParams, TranscribeModelParams},
    };

    let args: Vec<String> = std::env::args().collect();
    let model = PathBuf::from(
        args.get(1)
            .map_or("models/Qwen3-ASR-0.6B-Q8_0.gguf", String::as_str),
    );
    let wav = PathBuf::from(args.get(2).map_or("samples/jfk.wav", String::as_str));
    let language = args.get(3).cloned();

    let companion = TranscribeEngine::companion_for(&model);
    println!(
        "companion: {}",
        companion
            .as_ref()
            .map_or("none".to_string(), |p| p.display().to_string())
    );
    let mut engine = TranscribeEngine::new();
    let started = Instant::now();
    engine.load_model_with_params(
        &model,
        TranscribeModelParams {
            coreml_encoder: companion,
            ..Default::default()
        },
    )?;
    println!("load: {:.2}s", started.elapsed().as_secs_f32());

    for _ in 0..2 {
        let started = Instant::now();
        let result = engine.transcribe_file(
            &wav,
            Some(TranscribeInferenceParams {
                language: language.clone(),
                dictionary: args.iter().skip(4).cloned().collect(),
                ..Default::default()
            }),
        )?;
        println!(
            "{:.3}s lang={:?} text={}",
            started.elapsed().as_secs_f32(),
            result.language,
            result.text
        );
    }
    Ok(())
}
