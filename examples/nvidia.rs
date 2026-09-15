#[cfg(nvidia_engines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;

    use glimpse_speech::{
        TranscriptionEngine,
        engines::nemotron::NemotronEngine,
        engines::parakeet::{ParakeetEngine, ParakeetInferenceParams, ParakeetModelParams},
    };

    let args: Vec<String> = std::env::args().collect();
    let engine = args.get(1).map_or("parakeet", String::as_str);
    let model_dir = |default: &str| PathBuf::from(args.get(2).map_or(default, String::as_str));
    let wav_path = PathBuf::from(args.get(3).map_or("samples/dots.wav", String::as_str));

    let result = match engine {
        "parakeet" => {
            let mut engine = ParakeetEngine::new();
            engine.load_model_with_params(
                &model_dir("models/parakeet-tdt-0.6b-v3-onnx-int8"),
                ParakeetModelParams::int8(),
            )?;
            engine.transcribe_file(&wav_path, Some(ParakeetInferenceParams::default()))?
        }
        "nemotron" => {
            let mut engine = NemotronEngine::new();
            engine.load_model(&model_dir("models/nemotron-speech-streaming-en-0.6b"))?;
            engine.transcribe_file(&wav_path, None)?
        }
        other => {
            return Err(format!(
                "Unknown NVIDIA engine `{other}`. Expected `parakeet` or `nemotron`."
            )
            .into());
        }
    };

    println!("{}", result.text);
    for segment in result.segments.unwrap_or_default() {
        println!(
            "[{:.2}s - {:.2}s] {}",
            segment.start, segment.end, segment.text
        );
    }

    Ok(())
}

#[cfg(not(nvidia_engines))]
fn main() {
    eprintln!("The NVIDIA example is unavailable on Intel macOS builds.");
}
