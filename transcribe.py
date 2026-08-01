#!/usr/bin/env python3
"""Transcribe audio with a Whisper model and print JSON to stdout.

The script loads the Hugging Face automatic-speech-recognition pipeline once.
Then it transcribes each input file. It prints one JSON array to stdout.
Each element has the keys "file", "text", and "model".

Usage:
    python transcribe.py [path ...]

If no path is given, the script resolves the default samples from the Hugging
Face cache (dataset Narsil/asr_dummy) via huggingface_hub. The MODEL env var
selects the model (default openai/whisper-tiny.en). The SAMPLES_SET env var
selects the sample set: "en" (English, default) or "ml" (multilingual).
The input order becomes the output order.
"""

import json
import logging
import os
import sys
import warnings

# Silence transformers' noisy warnings (deprecated input name, attention mask,
# cache-migration logs) so stdout is clean JSON and stderr stays quiet.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import transformers
from huggingface_hub import hf_hub_download
from transformers import pipeline

transformers.logging.set_verbosity_error()

MODEL = os.environ.get("MODEL", "openai/whisper-tiny.en")
DATASET = "Narsil/asr_dummy"
EN_FILES = ("mlk.flac", "1.flac", "2.flac")
ML_FILES = ("4.flac", "hindi.ogg")


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


def get_inputs() -> list:
    """Return the list of (display_name, path) pairs to transcribe.

    Use the command line arguments when the user gives them.
    Otherwise resolve the default sample set from the HF cache.
    """
    if len(sys.argv) > 1:
        return [(a, a) for a in sys.argv[1:]]

    files = ML_FILES if os.environ.get("SAMPLES_SET") == "ml" else EN_FILES
    return resolve_samples(files)


def main() -> int:
    """Run the transcription and print the JSON array.

    Return 0 when all files succeed. Return 1 when one or more files fail.
    """
    inputs = get_inputs()

    if not inputs:
        print("[]")
        return 0

    pipe = pipeline(task="automatic-speech-recognition", model=MODEL)

    results = []
    had_error = False

    for display, path in inputs:
        try:
            # Load and decode with soundfile (no ffmpeg needed). The pipeline
            # resamples to 16 kHz internally via torchaudio.
            data, sr = sf.read(path)
            if data.ndim > 1:            # stereo or more: average to mono
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            output = pipe({"array": data, "sampling_rate": sr})
            text = output["text"].strip()
            results.append({"file": display, "text": text, "model": MODEL})
        except Exception as exc:  # noqa: BLE001 - report, do not abort the batch
            had_error = True
            results.append({"file": display, "error": str(exc), "model": MODEL})

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())