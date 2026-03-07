# glimpse-speech

`glimpse-speech` is a local transcription crate built around:
- `whisper-rs` for GGML Whisper inference
- `parakeet-rs` for ONNX Parakeet TDT inference

The public API is intentionally simple and engine-agnostic:
- `TranscriptionEngine`
- `TranscriptionResult`
- `TranscriptionSegment`
- `audio::read_wav_samples`
- `engines::*`

## Features

| Feature | Purpose |
| --- | --- |
| `whisper` | Enable `engines::whisper::WhisperEngine` |
| `parakeet` | Enable `engines::parakeet::ParakeetEngine` on supported targets |
| `all` | Enables `whisper` and `parakeet` |

Parakeet is disabled on Intel macOS (`x86_64-apple-darwin`) because `parakeet-rs` currently pulls an `ort` stack that does not ship the required prebuilt ONNX Runtime binary for that target.

## Installation

```toml
[dependencies]
glimpse-speech = { git = "https://github.com/LegendarySpy/Glimpse-Speech.git", tag = "1.0.2", features = ["whisper", "parakeet"] }
```

On Intel macOS, keep `whisper` enabled and treat `parakeet` as unavailable even if the feature is listed.

For local development in this repository:

```toml
[dependencies]
glimpse-speech = { path = "../Glimpse-Speech", features = ["whisper", "parakeet"] }
```

## Usage

### Whisper (local GGML)

```rust
use glimpse_speech::{
    engines::whisper::{WhisperEngine, WhisperInferenceParams},
    TranscriptionEngine,
};
use std::path::PathBuf;

let mut engine = WhisperEngine::new();
engine.load_model(&PathBuf::from("models/ggml-large-v3-turbo-q8_0.bin"))?;
let result = engine.transcribe_file(
    &PathBuf::from("audio.wav"),
    Some(WhisperInferenceParams {
        dictionary: vec!["Glimpse".into(), "Parakeet".into()],
        ..Default::default()
    }),
)?;
println!("{}", result.text);
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Parakeet (ONNX)

```rust
use glimpse_speech::{
    engines::parakeet::{ParakeetEngine, ParakeetInferenceParams, ParakeetModelParams},
    TranscriptionEngine,
};
use std::path::PathBuf;

let mut engine = ParakeetEngine::new();
engine.load_model_with_params(
    &PathBuf::from("models/parakeet-tdt-0.6b-v3-onnx-int8"),
    ParakeetModelParams::int8(),
)?;
let result = engine.transcribe_file(
    &PathBuf::from("audio.wav"),
    Some(ParakeetInferenceParams::default()),
)?;
println!("{}", result.text);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Model directory expectations:

- TDT Int8:
  - `encoder-model.int8.onnx`
  - `decoder_joint-model.int8.onnx`
  - `vocab.txt`
- TDT FP32:
  - `encoder-model.onnx`
  - `encoder-model.onnx.data`
  - `decoder_joint-model.onnx`
  - `vocab.txt`
## Further notes

Example commands:

```bash
cargo run --example whisper --features whisper -- <model.bin> <audio.wav>
cargo run --example parakeet --features parakeet -- <model-dir> <audio.wav>
```

The Parakeet example is only available on non-Intel macOS targets.

## Acknowledgments

- [whisper-rs](https://github.com/tazz4843/whisper-rs) (Unlicense) for Whisper bindings
- [parakeet-rs](https://github.com/altunenes/parakeet-rs) (MIT OR Apache-2.0) for ONNX Parakeet support
