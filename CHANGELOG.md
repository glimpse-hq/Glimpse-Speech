# Changelog

## 2.0.3

### Changed

- Parakeet with a Core ML encoder decodes a recording's chunks, and its dropped-speech retries, as one transcribe.cpp batch, so the Neural Engine encodes the next chunk while the current one decodes. On an M2 Pro, Parakeet TDT V3 with the same model files transcribes a 554 s file in 2.26 s instead of 4.21 s, and short clips in 47 ms instead of 68 ms.
- Silero VAD computes its spectrum with an FFT.
- transcribe.cpp is pinned to the Parakeet speed work: quantized decoder step weights for Parakeet (Nemotron keeps fp32), smaller Core ML encoder functions for short audio when the encoder has them (macOS 15), and int8 weights in the Core ML converter.

## 2.0.2

### Fixed

- A full Parakeet model with a Core ML encoder (Parakeet Unified) now loads on `Backend::Auto` instead of the CPU. Its streaming still runs the ggml encoder, and on an M2 Pro the CPU made release latency about three times slower (about 255 ms against 90 ms). Whole-file transcription with the Core ML encoder is as fast either way. Decoder-only files (Parakeet TDT V3) keep the CPU.

## 2.0.1

### Fixed

- A service now skips any still-compiling Core ML encoder when the model file can run without it, not only Whisper's. Parakeet Unified keeps its full GGUF, so loading it no longer waits minutes on a first Neural Engine compile. Decoder-only files still load their encoder.

## 2.0.0

Every local model now runs on [transcribe.cpp](https://github.com/LegendarySpy/transcribe.cpp) (our fork). whisper-rs, parakeet-rs and ONNX Runtime are gone, so there is one native engine to build and ship.

### Breaking

- whisper-rs, parakeet-rs and `ort` are removed. Whisper, Qwen3-ASR, Parakeet and Nemotron all run through `engines::transcribe::TranscribeEngine`.
- The `nvidia` and `parakeet` features are removed. `whisper` now implies `transcribe`, and `all` is `whisper` + `transcribe`.
- The `vad` module is only built with `transcribe`.
- `engines::whisper`, `engines::parakeet`, `engines::nemotron` and `take_coreml_log()` are removed.
- `ModelEngine::Parakeet` and `ModelEngine::Nemotron` are removed. The old names still deserialize as `ModelEngine::Transcribe` and are written back as `"transcribe"`.
- `ModelLayout`, `InstallSpec.layout` and `ResolvedModel.layout` are removed. Every model is a single file, optionally with a Core ML companion beside it.
- `TranscribeInferenceParams` gains `prompt` and `word_timestamps`.
- `ModelInstallManager::install()` returns an error when expected files are missing after the download.
- Parakeet and Nemotron ONNX model directories from 1.x no longer load.

### Added

- `diarization::LiveDiarizer` and `LiveTurns`: speaker detection while audio is recorded (Nemotron-3 Diarization).
- `SpeechService::loading_compiling_encoders()`, `TranscribeEngine::is_compiling()` and `TranscribeEngine::companion_for()`. A service skips a Whisper Core ML encoder that the app is still compiling, so loading never waits minutes on a first Neural Engine compile.
- Streaming for Nemotron and Parakeet Unified through `TranscribeEngine`.
- Whisper word timestamps, computed by an alignment pass only when word granularity is requested.
- Dictionary boosting on Parakeet and Nemotron: up to 64 entries, newest first, whole words only, default weight 3.0. Parakeet TDT uses a 4-wide beam search when words are set; Nemotron and streaming stay greedy. On the maintainer's own dictations, dictionary-word recall went from 44% to 86% with no changes on control clips.

### Changed

- Silero VAD runs in pure Rust and now works on Intel Macs too.
- Whisper decodes long audio in 28 second chunks.
- distil-Whisper models use their teacher's Core ML encoder.
- Nemotron runs on the CPU, which was faster and steadier for its 560 ms stream chunks.

### Fixed

- Parakeet TDT re-decodes speech it skipped after long pauses.
- Cancelling a download stops it immediately, including during the initial request.
- Cancelling a fresh install deletes only the files that install created.
- Whisper falls back to its ggml encoder when the Core ML encoder is rejected.

### Removed

- The ONNX Parakeet and Nemotron engines and their model layouts.

### Migrating from 1.x

- Features: replace `nvidia` or `parakeet` with `transcribe`. `whisper` already pulls in `transcribe`.
- Engines: match on `ModelEngine::Transcribe` instead of `Parakeet` or `Nemotron`. Stored settings with the old names keep working.
- Model specs: drop the `layout` field from `InstallSpec` and stop reading `ResolvedModel.layout`.
- Installs: handle the new `install()` error for incomplete downloads, and offer the GGUF model to users who still have an ONNX Parakeet or Nemotron directory.
- `TranscribeInferenceParams`: set `prompt` and `word_timestamps` (or use `..Default::default()`).
