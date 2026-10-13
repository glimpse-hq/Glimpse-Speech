# Changelog

## 2.0.5

### Added

- `vad::analyze` runs Silero over a whole recording: the share of speech frames, per-frame decisions on any frame size, and padded speech regions (speech starts at 0.5, lasts until below 0.35, widened by about half a second) for cutting audio. It first raises a quiet recording so its loudest 100 ms sits near -20 dBFS, by at most 30 dB. In model-bench's replay of Glimpse's recorder check, single words cut from real speech pass 92.3% instead of 66.7% at -20 dB and 96.7% instead of 86.3% at -10 dB, while room tone and typing pass 0 to 22.5% (WebRTC VAD: 62.5 to 100%).
- `vad::SpeechStream` runs Silero on live audio as it arrives, with the same padded regions. It does not raise quiet audio, so it matches `vad::analyze` on recordings whose loudest 100 ms reaches -20 dBFS.

### Changed

- `cleanup-apple` runs on a small Swift shim (swift/apple_cleanup.swift) instead of the `fm-rs` crate. fm-rs builds its Swift for macOS 26, which hard-linked Foundation symbols missing on older systems, so apps using the feature crashed at launch on macOS 14 and 15. The shim builds for macOS 14 with `#available` checks.
- The on-device model uses Apple's permissive guardrails for text transformations, and temperature 0 samples greedily.
- `apple_generate` takes a timeout and cancels the request when it passes, and returns an `AppleError` whose `kind` separates input problems (too long for the context window, refused, unsupported language) from the model being unavailable or busy.
- transcribe.cpp is pinned to upstream 0.4.0: ggml 0.26.0, plus language ID (ECAPA-TDNN) and Silero VAD sessions in the C API. Glimpse Speech does not use the new language ID or VAD sessions yet.

### Removed

- `CleanupProvider` and its built-in cleanup prompt. Use `apple_availability`, `apple_prewarm` and `apple_generate`.

## 2.0.4

### Changed

- transcribe.cpp is pinned to `ec5f294d` (glimpse-hq main: the decoder speed work plus upstream main through 95d82e51), and the dependency now points at github.com/glimpse-hq/transcribe.cpp.
- Parakeet recordings longer than two windows are cut in the middle of the latest pause of at least 0.2 s in each window's back half, so no word straddles a cut. Shorter audio keeps the quietest-point cut. On an M2 Pro, Parakeet TDT V3 on the Neural Engine transcribes a 554 s file in 1.50 s instead of 1.76 s, with long-form WER 3.26% instead of 3.47% and 5.50% instead of 6.19%. On Windows (Vulkan) the same file takes 4.82 s instead of 5.21 s.
- Dictionary words a model spells another way can be boosted again (transcribe.cpp boost floor 1e-5 to 1e-7). Parakeet TDT V3 dictionary recall is 95.1% instead of 91.0% on the Neural Engine and 89.3% instead of 85.2% on Windows, with false insertions unchanged or up by 1 of 797 control clips. Parakeet Ultra writes "Tauri" again.
- Diarization runs through transcribe.cpp's diarize sessions; live diarization rows are unchanged.
- Nemotron streaming on Metal handles short clips in 186 ms instead of 266 ms.
- Parakeet token confidence is vectorized on Apple, about 4% faster on short clips on the Neural Engine.

### Fixed

- Parakeet no longer prints `<unk>`, `<pad>` or `<|...|>` pieces.

## 2.0.3

### Changed

- Parakeet with a Core ML encoder decodes a recording's chunks, and its dropped-speech retries, as one transcribe.cpp batch, so the Neural Engine encodes the next chunk while the current one decodes. On an M2 Pro, Parakeet TDT V3 with the same model files transcribes a 554 s file in 2.26 s instead of 4.21 s, and short clips in 47 ms instead of 68 ms.
- Silero VAD computes its spectrum with an FFT.
- transcribe.cpp is pinned to the Parakeet speed work: quantized decoder step weights for Parakeet (Nemotron keeps fp32), smaller Core ML encoder functions for short audio when the encoder has them (macOS 15), and int8 weights in the Core ML converter.
- Inference uses one thread per performance core, up to 24, instead of a flat 8 (Windows and Intel Macs included). With a GPU or Core ML encoder doing the heavy work the CPU side keeps 8. On a 24-core Xeon, Parakeet TDT V3 on the CPU transcribes a 554 s file in 24 s instead of 119 s together with the 60 second windows below.
- Parakeet-family models without a Core ML encoder transcribe long recordings in 60 second windows. Faster on the CPU and Metal (Parakeet TDT V3 on Metal: 7.5 s instead of 19.8 s for 554 s) and a lower long-form error rate.
- Whole Nemotron recordings of 30 seconds or more run on a GPU copy of the model, loaded on the first such transcription, while live streaming and shorter audio (including dictation that falls back from streaming) stay on the CPU. On an RTX 4000 Ada, Nemotron EN transcribes a 554 s file in 2.8 s instead of 92 s. Nemotron 3.5 keeps whole recordings on the CPU on Metal.
- transcribe.cpp's mel fallback without BLAS (Windows) is about 4x faster with identical output.

### Fixed

- Parakeet Unified returned an empty transcript for long recordings passed in one piece without a Core ML encoder. The 60 second windows fix it.
- On Windows, loading any local model crashed (MSVCP140.dll+0x12c10) when an older MSVCP140.dll was the one loaded, for example another app's copy on PATH. transcribe.cpp now builds with `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR`, like whisper.cpp did before 2.0.0.

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
