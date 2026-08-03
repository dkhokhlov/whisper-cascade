#!/usr/bin/env python3
"""Transcribe audio with a Whisper model and print JSON to stdout.

The script loads the Hugging Face automatic-speech-recognition pipeline once.
Then it transcribes each input file. It prints one JSON array to stdout.
Each element has "file", "text", "model", and a "stats" block with
duration_s, elapsed_s, rtf, tokens, words, and chars.

Usage:
    python transcribe.py [path ...]

If no path is given, the script resolves input from the AUDIO env var when set.
AUDIO accepts one or more whitespace-separated tokens, each a Hugging Face URL
(hf://datasets/<ns>/<repo>/<file>), a local file, a directory, or a glob.
Otherwise the default multilingual samples (English, Spanish, Hindi) are
resolved from the Hugging Face cache (dataset Narsil/asr_dummy) via
huggingface_hub. The MODEL_ASR env var selects the model (default
openai/whisper-tiny, multilingual). The input order becomes the output order.
"""

import glob
import json
import logging
import os
import sys
import time
import warnings

# Silence transformers' noisy warnings (deprecated input name, attention mask,
# cache-migration logs) so stdout is clean JSON and stderr stays quiet.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import hqq_asr
import numpy as np
import soundfile as sf
import transformers
from huggingface_hub import hf_hub_download

transformers.logging.set_verbosity_error()

MODEL_ASR = os.environ.get("MODEL_ASR", "openai/whisper-tiny")
# Optional quantization. Set QUANT=hqq to load MODEL_ASR as a saved HQQ 4-bit
# model (local dir or Hugging Face repo). When unset, the model loads as fp32,
# or loads an already-quantized model when MODEL_ASR points at one.
QUANT = os.environ.get("QUANT", "").strip().lower()
DATASET = "Narsil/asr_dummy"
# Default multilingual sample set: English, Spanish, Hindi. All are loose files
# in the dataset and resolve from the HF cache via huggingface_hub.
DEFAULT_FILES = ("mlk.flac", "4.flac", "hindi.ogg")
# Audio extensions used when AUDIO points at a directory.
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def resolve_samples(files: tuple) -> list:
    """Resolve sample files from the Hugging Face cache.

    Download each file on first use, then reuse the cache. This is the same
    mechanism the model uses, so samples need no separate download step.
    Return a list of (display_name, path) pairs: the short dataset filename for
    display, and the cached path for reading.
    """
    return [
        (f, hf_hub_download(repo_id=DATASET, filename=f, repo_type="dataset"))
        for f in files
    ]


def download_hf_url(url: str) -> str:
    """Download one hf:// file and return its cached local path.

    hf_hub_download takes repo_id + filename + repo_type, not a URL, so this
    parses the URL. Forms (the repo-type segment is optional; models is the
    default):
        hf://datasets/<ns>/<repo>/<path...>
        hf://models/<ns>/<repo>/<path...>
    A repo-level URL (no file path) is not a single file and is rejected.
    """
    parts = url[len("hf://"):].split("/")
    repo_type = "model"
    if parts and parts[0] in ("datasets", "models", "spaces"):
        repo_type = {"datasets": "dataset", "models": "model", "spaces": "space"}[parts.pop(0)]
    if len(parts) < 3:
        raise ValueError(f"hf:// URL needs <ns>/<repo>/<file>: {url}")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def expand_audio(spec: str) -> list:
    """Expand one AUDIO token to (display_name, path) pairs.

    An hf:// URL is passed through as its own path placeholder and downloaded
    lazily in the per-file loop (so a bad URL or a missing file becomes a
    per-file error, not a crash). A directory becomes its audio files (by
    extension). A glob expands to its matches. Anything else is a literal
    file path.
    """
    if spec.startswith("hf://"):
        return [(spec, spec)]
    if os.path.isdir(spec):
        names = sorted(
            f for f in os.listdir(spec)
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS
        )
        return [(os.path.join(spec, f), os.path.join(spec, f)) for f in names]
    if any(c in spec for c in "*?["):
        return [(m, m) for m in sorted(glob.glob(spec))]
    return [(spec, spec)]


def get_inputs() -> list:
    """Return the list of (display_name, path) pairs to transcribe.

    Precedence: command line arguments, then the AUDIO env var (hf:// URL(s),
    file(s), dir, glob), then the default sample set from the HF cache
    (SAMPLES_SET).
    """
    if len(sys.argv) > 1:
        return [(a, a) for a in sys.argv[1:]]

    audio = os.environ.get("AUDIO", "").strip()
    if audio:
        out = []
        for tok in audio.split():
            out.extend(expand_audio(tok))
        return out

    return resolve_samples(DEFAULT_FILES)


def main() -> int:
    """Run the transcription and print the JSON array.

    Return 0 when all files succeed. Return 1 when one or more files fail.
    """
    inputs = get_inputs()

    if not inputs:
        # get_inputs() only returns [] when AUDIO was set but every token
        # expanded to nothing (an unmatched glob or an empty directory). A
        # default run resolves the built-in samples and is never empty. Treat
        # the no-match case as an error, not a silent success, so a typo in
        # AUDIO= stops the pipeline (with `set -o pipefail`) instead of
        # passing `null` through jq into the next stage.
        print("[]")
        print("error: AUDIO matched no files", file=sys.stderr)
        return 1

    pipe = hqq_asr.build_pipeline(MODEL_ASR, QUANT)

    results = []
    had_error = False

    for display, path in inputs:
        try:
            # An hf:// placeholder is downloaded here (not in get_inputs) so a
            # bad URL or a missing file becomes a per-file error, not a crash.
            if path.startswith("hf://"):
                path = download_hf_url(path)
            # Load and decode with soundfile (no ffmpeg needed). The pipeline
            # resamples to 16 kHz internally via torchaudio.
            data, sr = sf.read(path)
            if data.ndim > 1:            # stereo or more: average to mono
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            duration = len(data) / sr if sr else 0.0

            t0 = time.perf_counter()
            output = pipe({"array": data, "sampling_rate": sr})
            elapsed = time.perf_counter() - t0
            text = output["text"].strip()

            try:
                tokens = len(pipe.tokenizer.encode(text, add_special_tokens=False))
            except Exception:  # noqa: BLE001 - tokenizer access is not guaranteed
                tokens = None
            words = len(text.split())
            chars = len(text)

            results.append({
                "file": display,
                "text": text,
                "model": MODEL_ASR,
                "stats": {
                    "duration_s": round(duration, 3),
                    "elapsed_s": round(elapsed, 3),
                    "rtf": round(elapsed / duration, 3) if duration else None,
                    "tokens": tokens,
                    "words": words,
                    "chars": chars,
                },
            })
        except Exception as exc:  # noqa: BLE001 - report, do not abort the batch
            had_error = True
            results.append({"file": display, "error": str(exc), "model": MODEL_ASR})

    print(json.dumps(results, ensure_ascii=False, indent=2))

    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())