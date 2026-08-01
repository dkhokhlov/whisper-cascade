#!/usr/bin/env python3
"""Synthesize speech from text and write a WAV file.

Reads text from the TEXT env var, a file argument (or "-" for stdin), or stdin,
and writes the synthesized waveform to OUTPUT (default tts.wav). A JSON summary
goes to stdout (tts is the pipeline terminus, so stdout JSON is safe). The
model is a VITS model selected by the MODEL_TTS env var (default
facebook/mms-tts-eng, English). transformers 4.44.2 has no text-to-speech
pipeline, so this calls the model directly with AutoTokenizer + VitsModel.

Usage:
    python tts.py [file | -]
    echo "Hello" | python tts.py
    TEXT="Hello" python tts.py
    TEXT="Hello" OUTPUT=out.wav python tts.py
"""

import json
import logging
import os
import sys
import time
import warnings

# Silence transformers' warnings so stdout is clean JSON and stderr stays
# quiet except errors (same pattern as transcribe.py).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import AutoTokenizer, VitsModel

transformers.logging.set_verbosity_error()

MODEL_TTS = os.environ.get("MODEL_TTS", "facebook/mms-tts-eng")
OUTPUT = os.environ.get("OUTPUT", "tts.wav")


def main() -> int:
    """Run the synthesis and write the WAV file.

    Return 0 on success, 1 on empty input or a synthesis error.
    """
    from textio import get_text

    text = get_text()
    if not text:
        print(json.dumps({"error": "no input text (set TEXT=, pass a file, or pipe stdin)", "model": MODEL_TTS}))
        return 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TTS)
    model = VitsModel.from_pretrained(MODEL_TTS)
    inputs = tokenizer(text, return_tensors="pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
    elapsed = time.perf_counter() - t0

    waveform = out.waveform.squeeze().numpy().astype(np.float32)
    sr = model.config.sampling_rate
    sf.write(OUTPUT, waveform, sr)
    duration = len(waveform) / sr if sr else 0.0

    print(json.dumps({
        "output": OUTPUT,
        "model": MODEL_TTS,
        "text": text,
        "stats": {
            "chars": len(text),
            "words": len(text.split()),
            "duration_s": round(duration, 3),
            "elapsed_s": round(elapsed, 3),
            "rtf": round(elapsed / duration, 3) if duration else None,
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())