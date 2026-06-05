# glimpse-speech

`glimpse-speech` is a local transcription crate built around:
- `whisper-rs` for GGML Whisper inference
- `parakeet-rs` for NVIDIA ONNX speech models, including Parakeet and Nemotron

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
| `nvidia` | Enable NVIDIA-backed transcription engines on supported targets, including `engines::parakeet::ParakeetEngine` and `engines::nemotron::NemotronEngine` |
| `api` | Enable the local OpenAI-compatible transcription API helpers |
| `cli` | Enable the `glimpse-speech` command-line binary; includes `api` |
| `all` | Enables `whisper` and `nvidia` |

NVIDIA-backed engines are disabled on Intel macOS (`x86_64-apple-darwin`) because `parakeet-rs` currently pulls an `ort` stack that does not ship the required prebuilt ONNX Runtime binary for that target.

The legacy `parakeet` feature remains as a compatibility alias for `nvidia`.

## Installation

```toml
[dependencies]
glimpse-speech = { git = "https://github.com/LegendarySpy/Glimpse-Speech.git", tag = "1.3.4", features = ["whisper", "nvidia"] }
```

On Intel macOS, keep `whisper` enabled and treat `nvidia` as unavailable even if the feature is listed.

For local development in this repository:

```toml
[dependencies]
glimpse-speech = { path = "../Glimpse-Speech", features = ["whisper", "nvidia", "api", "cli"] }
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

Speaker diarization is currently disabled while the API is being cleaned up.

### Nemotron (streaming ONNX)

```rust
use glimpse_speech::{
    engines::nemotron::NemotronEngine,
    TranscriptionEngine,
};
use std::path::PathBuf;

let mut engine = NemotronEngine::new();
engine.load_model(&PathBuf::from("models/nemotron-speech-streaming-en-0.6b"))?;
let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
println!("{}", result.text);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Nemotron also exposes streaming helpers:

- `transcribe_chunk(&[f32])`
- `get_transcript()`
- `reset()`

Model directory expectations:

- `encoder.onnx`
- `encoder.onnx.data`
- `decoder_joint.onnx`
- `tokenizer.model`

## Further notes

Example commands:

```bash
cargo run --example whisper --features whisper -- <model.bin> <audio.wav>
cargo run --example nvidia --features nvidia -- parakeet <model-dir> <audio.wav>
cargo run --example nvidia --features nvidia -- nemotron <model-dir> <audio.wav>
cargo run --features cli,whisper -- transcribe <audio.wav> --model <model.bin> --engine whisper
cargo run --features cli,nvidia -- transcribe <audio.wav> --model <model-dir> --engine parakeet
cargo run --features cli,nvidia -- transcribe <audio.wav> --model <model-dir> --engine nemotron
```

If you omit the engine argument, the NVIDIA example defaults to `parakeet`.
The CLI defaults to `--engine whisper`; Parakeet and Nemotron models are resolved as directories
and validated by their engine loaders.
On macOS, the CLI uses Glimpse's shared model cache by default:
`~/Library/Application Support/com.glimpse.data/models`. Override it with `--cache-dir` or
`GLIMPSE_SPEECH_CACHE_DIR`.

The NVIDIA-backed example is only available on non-Intel macOS targets.

## Acknowledgments

- [whisper-rs](https://github.com/tazz4843/whisper-rs) (Unlicense) for Whisper bindings
- [parakeet-rs](https://github.com/altunenes/parakeet-rs) (MIT OR Apache-2.0) for NVIDIA ONNX speech model support
