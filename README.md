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
| `nvidia` | Enable all NVIDIA-backed engines in this crate on supported targets, including `engines::parakeet::ParakeetEngine`, Sortformer-based diarization helpers, and `engines::nemotron::NemotronEngine` |
| `all` | Enables `whisper` and `nvidia` |

NVIDIA-backed engines are disabled on Intel macOS (`x86_64-apple-darwin`) because `parakeet-rs` currently pulls an `ort` stack that does not ship the required prebuilt ONNX Runtime binary for that target.

The legacy `parakeet` feature remains as a compatibility alias for `nvidia`.

## Installation

```toml
[dependencies]
glimpse-speech = { git = "https://github.com/LegendarySpy/Glimpse-Speech.git", tag = "1.2.5", features = ["whisper", "nvidia"] }
```

On Intel macOS, keep `whisper` enabled and treat `nvidia` as unavailable even if the feature is listed.

For local development in this repository:

```toml
[dependencies]
glimpse-speech = { path = "../Glimpse-Speech", features = ["whisper", "nvidia"] }
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

### Parakeet + Sortformer diarization

`glimpse-speech` bundles Sortformer support into the `nvidia` feature, so downstream crates do not need a separate crate feature to access diarization.

```rust
use glimpse_speech::{
    diarization::attribute_speakers,
    engines::{
        parakeet::{ParakeetEngine, ParakeetInferenceParams, ParakeetModelParams, TimestampGranularity},
        sortformer::{SortformerEngine, SortformerModelParams},
    },
    SpeakerDiarizationEngine, TranscriptionEngine,
};
use std::path::PathBuf;

let mut transcription_engine = ParakeetEngine::new();
transcription_engine.load_model_with_params(
    &PathBuf::from("models/parakeet-tdt-0.6b-v3-onnx-int8"),
    ParakeetModelParams::int8(),
)?;

let mut diarization_engine = SortformerEngine::new();
diarization_engine.load_model_with_params(
    &PathBuf::from("models/diar_streaming_sortformer_4spk-v2.onnx"),
    SortformerModelParams::callhome(),
)?;

let transcription = transcription_engine.transcribe_file(
    &PathBuf::from("audio.wav"),
    Some(ParakeetInferenceParams {
        timestamp_granularity: TimestampGranularity::Segment,
        ..Default::default()
    }),
)?;
let speaker_segments = diarization_engine.diarize_file(&PathBuf::from("audio.wav"))?;
let result = attribute_speakers(transcription, speaker_segments)?;

for segment in result.segments {
    println!(
        "[{:.2}s - {:.2}s] speaker={:?} {}",
        segment.start, segment.end, segment.speaker_id, segment.text
    );
}
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
- Sortformer diarization:
  - `diar_streaming_sortformer_4spk-v2.onnx` or `diar_streaming_sortformer_4spk-v2.1.onnx`

Diarization-specific pieces are model-agnostic:

- `SpeakerDiarizationEngine`
- `diarization::SpeakerDiarizationSegment`
- `diarization::DiarizedTranscriptionResult`
- `diarization::attribute_speakers()`
- `engines::sortformer::SortformerEngine`

`SortformerModelParams` defaults to the upstream CallHome tuning and reads streaming values from ONNX metadata when present. To lower latency, override the runtime window after construction:

```rust
use glimpse_speech::engines::sortformer::SortformerModelParams;

let params = SortformerModelParams::callhome()
    .with_streaming_overrides(62, 62, 94, 1);
```

Sortformer can stream arbitrarily long audio, but Parakeet-TDT still has sequence-length limits. For long recordings, run diarization across the full file and transcribe shorter chunks before merging timestamps back together.

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
cargo run --example diarization --features nvidia -- <parakeet-model-dir> <sortformer.onnx> <audio.wav>
```

If you omit the engine argument, the NVIDIA example defaults to `parakeet`.

The NVIDIA-backed example is only available on non-Intel macOS targets.

## Acknowledgments

- [whisper-rs](https://github.com/tazz4843/whisper-rs) (Unlicense) for Whisper bindings
- [parakeet-rs](https://github.com/altunenes/parakeet-rs) (MIT OR Apache-2.0) for NVIDIA ONNX speech model support
