# glimpse-speech

Local speech-to-text for Rust. One crate, an OpenAI-compatible HTTP API, and a CLI.

Every local model runs on [transcribe.cpp](https://github.com/glimpse-hq/transcribe.cpp) (our fork): Metal on macOS, optional Core ML/ANE encoders on Apple Silicon, and Vulkan on Windows and Linux.

- **Whisper**: GGUF or whisper.cpp GGML `.bin` files, with word timestamps from an alignment pass
- **Qwen3-ASR** and **Parakeet TDT V3**: batch transcription
- **Parakeet Unified** and **Nemotron Streaming** (English and 3.5 multilingual): streaming transcription
- **Nemotron-3 Diarization**: speaker detection, batch or live

Parakeet and Nemotron boost dictionary words during decoding.

## Cargo features

| Feature | Enables |
| --- | --- |
| `transcribe` | `engines::transcribe::TranscribeEngine`, `diarization` (`diarize` and `LiveDiarizer` with [Nemotron-3 Diarization](https://huggingface.co/Glimpse-Dictation/Nemotron-3-Diarization-gguf), up to 8 speakers; `diarize` also takes Sortformer v2.1), and Silero VAD (`vad`, pure Rust, Intel Macs included). Builds transcribe.cpp from source: CMake and a C++ toolchain, plus the Vulkan SDK on Windows and Linux |
| `whisper` | Whisper as the default loose engine (implies `transcribe`) |
| `api` | The OpenAI-compatible HTTP server (`api::serve`) |
| `remote` | Proxying to a remote OpenAI-compatible endpoint, with local fallback |
| `cli` | The `glimpse-speech` binary (implies `api`) |
| `apple-speech` | `engines::apple`, the macOS 26 SpeechAnalyzer engine (Apple Silicon) |
| `cleanup-apple` | Transcript cleanup with Apple's on-device Foundation Models (Apple Silicon) |
| `all` | `whisper` + `transcribe` |

## Installation

```toml
[dependencies]
glimpse-speech = { git = "https://github.com/glimpse-hq/Glimpse-Speech.git", tag = "2.0.0", features = ["whisper"] }
```

The transcribe.cpp dependency is pinned to an exact Git revision, including its
Core ML bindings. Consumers should pin a Glimpse-Speech revision that includes
these features. Local development overrides belong in Cargo configuration, not
in committed dependency paths. For example, from the Glimpse app checkout:

```bash
cargo check --manifest-path src-tauri/Cargo.toml \
  --config 'patch."https://github.com/glimpse-hq/Glimpse-Speech.git".glimpse-speech.path="../Glimpse-Speech"'
```

Long audio is decoded in chunks split at quiet boundaries, without overlap or
omitted samples: 28 seconds for Whisper, 15 seconds for Qwen3-ASR and for
Parakeet with a Core ML encoder (the encoder's capacity). Input-length and
output-truncation errors retry smaller chunks; other errors propagate normally.
Partial transcripts are never reported as complete. Parakeet TDT re-decodes
stretches of detected speech that came back without words, which can happen
after long pauses.

Parakeet TDT V3 supports either a full GGUF for CPU/GPU inference or a compact
decoder-only GGUF paired with its matching Core ML encoder on Apple Silicon.
Keep the compiled `<model stem>-encoder.mlmodelc` directory beside the GGUF;
decoder-only files also recognize the stem without `-decoder`. The compact
package requires its encoder and cannot serve as a standalone CPU/GPU model.
Parakeet and Whisper expose word timestamps; Qwen does not. Neither Qwen nor
Parakeet TDT exposes streaming.

The request dictionary is a recognition hint, not a forced replacement. Qwen and
Whisper get it as vocabulary context. Parakeet and Nemotron boost up to 64
entries (newest first, whole words only); Parakeet TDT switches to a 4-wide beam
search when words are set, while Nemotron and streaming decode greedily.

Core ML companions accelerate the encoder only. A Whisper encoder serves every
quantization of its family, and distil-Whisper models use their teacher's
encoder. A Whisper encoder marked as still compiling (`.<name>.compiling` beside
it) is skipped unless the service was built with
`SpeechService::loading_compiling_encoders()`; `TranscribeEngine::is_compiling`
and `TranscribeEngine::companion_for` expose the same checks. Nemotron runs on
the CPU. Installing a
companion beside an already-loaded model requires `SpeechService::unload()`
before loading or warming it again. The service cache tracks model id and path,
not changes to companion files.

## CLI

```bash
# Transcribe a file (WAV is decoded in-process; other formats need ffmpeg)
glimpse-speech transcribe audio.wav --model ggml-large-v3-turbo-q8_0.bin
glimpse-speech transcribe audio.m4a --model parakeet-tdt-0.6b-v3-Q8_0.gguf
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

- `--engine whisper|transcribe` (default `whisper`; `parakeet` and `nemotron` are accepted as aliases for `transcribe`, and a `.gguf` model path or a Parakeet or Nemotron model id selects it automatically)
- `--response-format text|json|verbose_json|srt|vtt` (default `text`; `verbose_json`, `srt` and `vtt` turn on segment timestamps)
- `--language`, `--prompt`, `--dictionary <term>` (repeatable), `--timestamps`
- `--cache-dir <path>` or `GLIMPSE_SPEECH_CACHE_DIR` to override the model cache
- `--json` for machine-readable output

On macOS the default model cache is `~/Library/Application Support/com.glimpse.data/models`. Whisper models resolve to a file (by path, cache name, or single file in a cache directory). transcribe.cpp models resolve to a GGUF file the same way.

## HTTP API

`glimpse-speech serve` exposes an OpenAI-compatible surface:

| Endpoint | Description |
| --- | --- |
| `POST /v1/audio/transcriptions` | Multipart transcription, OpenAI-compatible |
| `GET /v1/models` | Available models |
| `GET /health` | Liveness check, no auth |

Multipart fields: `file` (required), `model` (required), `language`, `prompt`, `response_format` (`json`, `text`, `verbose_json`, `srt`, `vtt`), `timestamp_granularities[]` (`segment`, `word`, requires `verbose_json`), `dictionary` (comma separated terms biased into recognition).

```bash
curl -F file=@audio.wav -F model=<model> -F response_format=verbose_json \
     -F "timestamp_granularities[]=word" http://127.0.0.1:11435/v1/audio/transcriptions
```

Word timestamps from Whisper come from cross-attention alignment, computed only when word granularity is requested.

Auth and networking:

- Loopback by default; binding to LAN requires `--api-key`
- Keys are accepted as `Authorization: Bearer <key>` or `x-api-key: <key>`
- `GLIMPSE_SPEECH_API_KEY` and `GLIMPSE_SPEECH_REMOTE_API_KEY` stand in for `--api-key` and `--remote-api-key`, keeping keys out of the process list
- `--cors` enables permissive CORS for browser clients

With `--remote-endpoint` set, transcription requests proxy to the remote service. Endpoint quirks (Mistral, OpenRouter, xAI, ElevenLabs, Deepgram, self-hosted servers) are detected automatically, WAV uploads are converted to FLAC to cut upload size, and transient remote failures fall back to the local engine when a local model is installed. Speaker diarization is requested from endpoints that support it (Mistral, xAI, ElevenLabs, Deepgram, Fireworks, and OpenAI's diarize model).

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
use glimpse_speech::{engines::transcribe::TranscribeEngine, TranscriptionEngine};
use std::path::PathBuf;

let mut engine = TranscribeEngine::new();
engine.load_model(&PathBuf::from("models/ggml-large-v3-turbo-q8_0.bin"))?;
let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
println!("{}", result.text);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`TranscribeEngine` streams Nemotron and Parakeet Unified GGUFs: `configure_stream(language, dictionary)`, then `transcribe_chunk(&[f32])` for each chunk, `finalize()` for the final text, and `reset()`.

`diarization::LiveDiarizer` detects speakers while audio arrives: `feed(&[f32])` returns the turns so far as `LiveTurns`, and `finish()` returns the final `SpeakerTurn`s.

### Expected model files

| Engine | Required files |
| --- | --- |
| Whisper | a single GGUF (for example `whisper-small-Q8_0.gguf`) or GGML `.bin` file |
| Qwen3-ASR, Parakeet, Nemotron | a single GGUF (for example `parakeet-tdt-0.6b-v3-Q8_0.gguf`) |

Parakeet and Nemotron ONNX directories from 1.x no longer load; install the GGUF instead.

For Core ML acceleration on Apple Silicon, place a transcribe.cpp Whisper encoder named `whisper-<family>-encoder.mlmodelc` (for example `whisper-small-encoder.mlmodelc`) next to the `whisper-<family>-<quant>.gguf` or `ggml-<family>[-qX_Y].bin` file. whisper.cpp's own `ggml-<family>-encoder.mlmodelc` encoders are not compatible; an encoder that fails to load is skipped and the ggml encoder runs instead.

## Examples

```bash
cargo run --example transcribe --features transcribe -- <model.bin or model.gguf> <audio.wav>
cargo run --example diarize --features transcribe -- <nemotron-3-diarization-Q8_0.gguf> <audio.wav>
```

## Acknowledgments

- [Silero VAD](https://github.com/snakers4/silero-vad) (MIT) v6.2 model weights, bundled as `src/silero_vad_16k_op15.onnx` and run by the pure-Rust `vad` module
