#!/usr/bin/env python3
"""Transcribe WAV files with the whisper-tiny.en model and print JSON to stdout.

This script loads the Hugging Face automatic-speech-recognition pipeline once.
Then it transcribes each input file. It prints one JSON array to stdout.
Each element has the keys "file", "text", and "model".

Usage:
    python transcribe.py [path.wav ...]

If no path is given, the script transcribes every .wav file in the samples
directory. The samples directory comes from the SAMPLES_DIR environment variable.
The default value is "./samples". The input order becomes the output order.
"""

import glob
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
from transformers import pipeline

transformers.logging.set_verbosity_error()

MODEL = os.environ.get("MODEL", "openai/whisper-tiny.en")
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", "./samples")


def get_inputs() -> list:
    """Return the list of input file paths.

    Use the command line arguments when the user gives them.
    Otherwise use the audio files in the samples directory.
    """
    if len(sys.argv) > 1:
        return list(sys.argv[1:])

    found = []
    for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
        found.extend(glob.glob(os.path.join(SAMPLES_DIR, ext)))
    return sorted(found)


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

    for path in inputs:
        try:
            # Load and decode with soundfile (no ffmpeg needed). The pipeline
            # resamples to 16 kHz internally via torchaudio.
            data, sr = sf.read(path)
            if data.ndim > 1:            # stereo or more: average to mono
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            output = pipe({"array": data, "sampling_rate": sr})
            text = output["text"].strip()
            results.append({"file": path, "text": text, "model": MODEL})
        except Exception as exc:  # noqa: BLE001 - report, do not abort the batch
            had_error = True
            results.append({"file": path, "error": str(exc), "model": MODEL})

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())