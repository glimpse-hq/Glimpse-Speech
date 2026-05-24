#!/usr/bin/env python3
import argparse
import json
import mimetypes
import os
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def multipart_body(fields, files):
    boundary = f"----glimpse-speech-{uuid.uuid4().hex}"
    chunks = []

    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )

    for name, path in files:
        path = Path(path)
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{path.name}"\r\n'
                ).encode(),
                f"Content-Type: {mime_type}\r\n\r\n".encode(),
                path.read_bytes(),
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode())
    return boundary, b"".join(chunks)


def post_transcription(base_url, fields, audio_path):
    boundary, body = multipart_body(fields, [("file", audio_path)])
    request = Request(
        f"{base_url.rstrip('/')}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return response.status, response.headers.get("content-type", ""), response.read()
    except HTTPError as error:
        return error.code, error.headers.get("content-type", ""), error.read()
    except URLError as error:
        raise SystemExit(f"Failed to reach API: {error}") from error


def require_ok(name, status, body):
    if status != 200:
        raise SystemExit(f"{name} failed with HTTP {status}: {body.decode(errors='replace')}")


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test the OpenAI-compatible Glimpse Speech transcription endpoint."
    )
    parser.add_argument("--base-url", default=os.getenv("GLIMPSE_SPEECH_API", "http://127.0.0.1:11435"))
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    status, content_type, body = post_transcription(
        args.base_url,
        [
            ("model", args.model),
            ("language", "en"),
            ("prompt", "Transcribe the sample audio."),
            ("dictionary", "Glimpse,Parakeet,Nemotron"),
            ("response_format", "json"),
        ],
        args.audio,
    )
    require_ok("json transcription", status, body)
    payload = json.loads(body)
    if "text" not in payload:
        raise SystemExit(f"json response missing text: {payload}")

    status, content_type, body = post_transcription(
        args.base_url,
        [
            ("model", args.model),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "segment"),
            ("timestamp_granularities[]", "word"),
            ("dictionary[]", "Glimpse"),
        ],
        args.audio,
    )
    require_ok("verbose_json transcription", status, body)
    payload = json.loads(body)
    for key in ("text", "segments", "duration", "task"):
        if key not in payload:
            raise SystemExit(f"verbose_json response missing {key}: {payload}")

    status, content_type, body = post_transcription(
        args.base_url,
        [("model", args.model), ("response_format", "text")],
        args.audio,
    )
    require_ok("text transcription", status, body)
    if not content_type.startswith("text/plain"):
        raise SystemExit(f"text response used unexpected content-type: {content_type}")

    print("OpenAI-compatible API smoke test passed")


if __name__ == "__main__":
    main()
