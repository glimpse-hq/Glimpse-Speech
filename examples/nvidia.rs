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
    engines::nemotron::NemotronEngine,
    engines::parakeet::{ParakeetEngine, ParakeetInferenceParams, ParakeetModelParams},
    TranscriptionEngine, TranscriptionResult,
};

#[cfg(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let engine = args
        .get(1)
        .map(|value| value.as_str())
        .unwrap_or("parakeet");

    let default_model_dir = match engine {
        "parakeet" => "models/parakeet-tdt-0.6b-v3-onnx-int8",
        "nemotron" => "models/nemotron-speech-streaming-en-0.6b",
        other => {
            return Err(io_error(format!(
                "Unknown NVIDIA engine `{other}`. Expected `parakeet` or `nemotron`."
            )))
        }
    };

    let model_dir = PathBuf::from(
        args.get(2)
            .map(|value| value.as_str())
            .unwrap_or(default_model_dir),
    );
    let wav_path = PathBuf::from(
        args.get(3)
            .map(|value| value.as_str())
            .unwrap_or("samples/dots.wav"),
    );

    let result = match engine {
        "parakeet" => transcribe_with_parakeet(&model_dir, &wav_path)?,
        "nemotron" => transcribe_with_nemotron(&model_dir, &wav_path)?,
        _ => unreachable!(),
    };

    print_result(result);

    Ok(())
}

#[cfg(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
fn transcribe_with_parakeet(
    model_dir: &std::path::Path,
    wav_path: &std::path::Path,
) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
    let mut engine = ParakeetEngine::new();
    engine.load_model_with_params(model_dir, ParakeetModelParams::int8())?;
    engine.transcribe_file(wav_path, Some(ParakeetInferenceParams::default()))
}

#[cfg(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
fn transcribe_with_nemotron(
    model_dir: &std::path::Path,
    wav_path: &std::path::Path,
) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
    let mut engine = NemotronEngine::new();
    engine.load_model(model_dir)?;
    engine.transcribe_file(wav_path, None)
}

#[cfg(all(
    feature = "nvidia",
    not(all(target_os = "macos", target_arch = "x86_64"))
))]
fn print_result(result: TranscriptionResult) {
    println!("{}", result.text);

    if let Some(segments) = result.segments {
        for segment in segments {
            println!(
                "[{:.2}s - {:.2}s] {}",
                segment.start, segment.end, segment.text
            );
        }
    }
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
    eprintln!("The NVIDIA example is unavailable on Intel macOS builds.");
}
