#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <cache-dir> <model-id> <audio.wav>" >&2
  exit 2
fi

cache_dir="$1"
model_id="$2"
audio_path="$3"

cargo run --features cli,whisper --bin glimpse-speech -- \
  --cache-dir "$cache_dir" \
  --json \
  transcribe "$audio_path" \
  --model "$model_id" \
  --language en \
  --prompt "Transcribe the sample audio." \
  --dictionary Glimpse \
  --response-format json |
  python3 -c 'import json,sys; data=json.load(sys.stdin); assert "text" in data; print("CLI smoke test passed")'
