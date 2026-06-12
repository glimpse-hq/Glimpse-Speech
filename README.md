# glimpse-speech

Local speech-to-text for Rust. One crate, three engines, an OpenAI-compatible HTTP API, and a CLI.

- **Whisper** (GGML via [whisper-rs](https://github.com/tazz4843/whisper-rs)): Metal and Core ML/ANE on Apple Silicon, Vulkan on Windows and Linux
- **Parakeet TDT** (NVIDIA ONNX via [parakeet-rs](https://github.com/altunenes/parakeet-rs)): fast batch transcription, int8 and fp32
- **Nemotron** (NVIDIA ONNX): streaming transcription with incremental results

## Cargo features

| Feature | Enables |
| --- | --- |
| `whisper` | `engines::whisper::WhisperEngine` |
| `nvidia` | `engines::parakeet::ParakeetEngine` and `engines::nemotron::NemotronEngine` |
| `api` | The OpenAI-compatible HTTP server (`api::serve`) |
| `remote` | Proxying to a remote OpenAI-compatible endpoint, with local fallback |
| `cli` | The `glimpse-speech` binary (implies `api`) |
| `all` | `whisper` + `nvidia` |

NVIDIA engines are unavailable on Intel macOS (`x86_64-apple-darwin`) because ONNX Runtime ships no prebuilt binary for that target. `parakeet` remains as a compatibility alias for `nvidia`.

## Installation

```toml
[dependencies]
glimpse-speech = { git = "https://github.com/LegendarySpy/Glimpse-Speech.git", tag = "1.4.7", features = ["whisper", "nvidia"] }
```

Use the latest tag. For local development:

```toml
glimpse-speech = { path = "../Glimpse-Speech", features = ["whisper", "nvidia", "api", "cli", "remote"] }
```

## CLI

```bash
# Transcribe a file (any format ffmpeg can decode, or 16 kHz mono PCM16 WAV directly)
glimpse-speech transcribe audio.wav --model ggml-large-v3-turbo-q8_0.bin
glimpse-speech transcribe audio.m4a --model parakeet-tdt-int8 --engine parakeet
glimpse-speech transcribe audio.wav --model <model> --response-format srt --timestamps

# Manage models in the shared cache
glimpse-speech models list
glimpse-speech models install whisper_base --url https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin --sha256 <hash>
glimpse-speech models delete whisper_base

# Serve the HTTP API
glimpse-speech serve --port 11435 --model <model>
glimpse-speech serve --port 11435 --remote-endpoint https://api.openai.com/v1 --remote-api-key sk-... --remote-model whisper-1
```

Useful flags:

- `--engine whisper|parakeet|nemotron` (default `whisper`)
- `--response-format text|json|verbose_json|srt|vtt` (default `text`)
- `--language`, `--prompt`, `--dictionary <term>` (repeatable), `--timestamps`
- `--cache-dir <path>` or `GLIMPSE_SPEECH_CACHE_DIR` to override the model cache
- `--json` for machine-readable output

On macOS the default model cache is `~/Library/Application Support/com.glimpse.data/models`. Whisper models resolve to a file (by path, cache name, or single file in a cache directory). Parakeet and Nemotron models resolve to a directory.

## HTTP API

`glimpse-speech serve` exposes an OpenAI-compatible surface:

| Endpoint | Description |
| --- | --- |
| `POST /v1/audio/transcriptions` | Multipart transcription, OpenAI-compatible |
| `GET /v1/models` | Available models |

Multipart fields: `file` (required), `model` (required), `language`, `prompt`, `response_format` (`json`, `text`, `verbose_json`, `srt`, `vtt`), `timestamp_granularities[]` (`segment`, `word`, requires `verbose_json`), `dictionary` (comma separated terms biased into recognition).

```bash
curl -F file=@audio.wav -F model=<model> -F response_format=verbose_json \
     -F "timestamp_granularities[]=word" http://127.0.0.1:11435/v1/audio/transcriptions
```

Word timestamps from Whisper are token-aligned acoustic boundaries, not interpolated estimates.

Auth and networking:

- Loopback by default; binding to LAN requires `--api-key`
- Keys are accepted as `Authorization: Bearer <key>` or `x-api-key: <key>`
- `--cors` enables permissive CORS for browser clients

With `--remote-endpoint` set, transcription requests proxy to the remote service. Endpoint quirks (Mistral, self-hosted servers) are detected automatically, WAV uploads are converted to FLAC to cut upload size, and transient remote failures fall back to the local engine when a local model is installed.

## Library

### Service layer

`SpeechService` manages the model cache, engine loading, and warmup. `api::serve` and the CLI are built on it.

```rust
use glimpse_speech::service::{AudioInput, SpeechService, TranscribeRequest};
use glimpse_speech::models::ModelEngine;

let service = SpeechService::new_loose_with_engine(cache_dir, ModelEngine::Whisper);
let transcription = service.transcribe(TranscribeRequest {
    audio: AudioInput::WavPath("audio.wav".into()),
    model_id: "ggml-large-v3-turbo-q8_0.bin".into(),
    language: None,
    prompt: None,
    dictionary: vec!["Glimpse".into()],
    timestamps: false,
    timestamp_granularity: None,
})?;
println!("{}", transcription.text);
# Ok::<(), anyhow::Error>(())
```

`ModelInstallManager` (in `models`) handles downloads with resume, in-flight sha256 verification, zip extraction, and cancellation.

### Engines directly

```rust
use glimpse_speech::{engines::whisper::WhisperEngine, TranscriptionEngine};
use std::path::PathBuf;

let mut engine = WhisperEngine::new();
engine.load_model(&PathBuf::from("models/ggml-large-v3-turbo-q8_0.bin"))?;
let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
println!("{}", result.text);
# Ok::<(), Box<dyn std::error::Error>>(())
```

```rust
use glimpse_speech::{
    engines::parakeet::{ParakeetEngine, ParakeetModelParams},
    TranscriptionEngine,
};
use std::path::PathBuf;

let mut engine = ParakeetEngine::new();
engine.load_model_with_params(
    &PathBuf::from("models/parakeet-tdt-0.6b-v3-onnx-int8"),
    ParakeetModelParams::int8(),
)?;
let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
println!("{}", result.text);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Nemotron additionally exposes streaming: `transcribe_chunk(&[f32])`, `get_transcript()`, and `reset()`. Chunks are 560 ms at 16 kHz (`STREAMING_CHUNK_SAMPLES`).

### Expected model files

| Engine | Required files |
| --- | --- |
| Whisper | a single GGML `.bin` file |
| Parakeet TDT int8 | `encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx`, `vocab.txt` |
| Parakeet TDT fp32 | `encoder-model.onnx`, `encoder-model.onnx.data`, `decoder_joint-model.onnx`, `vocab.txt` |
| Nemotron | `encoder.onnx`, `encoder.onnx.data`, `decoder_joint.onnx`, `tokenizer.model` |

For Core ML acceleration on Apple Silicon, place the matching `ggml-<name>-encoder.mlmodelc` directory next to the Whisper model file. The first load runs a one-time ANE compilation pass that the OS caches.

## Examples

```bash
cargo run --example whisper --features whisper -- <model.bin> <audio.wav>
cargo run --example nvidia --features nvidia -- parakeet <model-dir> <audio.wav>
cargo run --example nvidia --features nvidia -- nemotron <model-dir> <audio.wav>
```

## Acknowledgments

- [whisper-rs](https://github.com/tazz4843/whisper-rs) (Unlicense) for Whisper bindings
- [parakeet-rs](https://github.com/altunenes/parakeet-rs) (MIT OR Apache-2.0) for NVIDIA ONNX speech model support
