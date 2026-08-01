#!/usr/bin/env python3
"""Translate text to English and print a JSON result to stdout.

Reads text from the TEXT env var, a file argument (or "-" for stdin), or stdin,
and writes a JSON object to stdout with "text" (the English translation),
"model", and a "stats" block. To pipe into make tts, extract the text with
`jq -r '.text'`. The model is a MarianMT model selected by the MODEL_TRANSLATE
env var (default Helsinki-NLP/opus-mt-mul-en, multilingual to English). The
target is named by the target language: `make en` runs this script; `make es`
etc. are one-line Make targets that set MODEL_TRANSLATE to a different model.

Usage:
    python translate.py [file | -]
    echo "Hola" | python translate.py | jq -r '.text'
    TEXT="Hola" python translate.py
"""

import json
import logging
import os
import sys
import time
import warnings

# Silence transformers' warnings so stdout is pure JSON and stderr stays quiet
# (same pattern as transcribe.py).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import transformers
from transformers import pipeline

transformers.logging.set_verbosity_error()

MODEL_TRANSLATE = os.environ.get("MODEL_TRANSLATE", "Helsinki-NLP/opus-mt-mul-en")
# Target language code, printed in the JSON output as "lang". Defaults to "en"
# (make en). A future make es etc. sets this (and MODEL_TRANSLATE) per target.
TARGET_LANG = os.environ.get("TARGET_LANG", "en")


def main() -> int:
    """Run the translation and print the JSON result.

    Return 0 on success, 1 on empty input or a translation error.
    """
    # Import here so unit tests can stub the pipeline before main() runs.
    from textio import get_text

    text = get_text()
    if not text:
        print(json.dumps({"error": "no input text (set TEXT=, pass a file, or pipe stdin)", "lang": TARGET_LANG, "model": MODEL_TRANSLATE}))
        return 1

    try:
        translator = pipeline(task="translation", model=MODEL_TRANSLATE)
        # Time inference only, not model loading (matches asr/tts).
        t0 = time.perf_counter()
        result = translator(text)
        elapsed = time.perf_counter() - t0

        translated = result[0]["translation_text"].strip()
    except Exception as exc:  # noqa: BLE001 - emit a JSON error, not a traceback
        print(json.dumps({"error": str(exc), "lang": TARGET_LANG, "model": MODEL_TRANSLATE}))
        return 1

    print(json.dumps({
        "text": translated,
        "lang": TARGET_LANG,
        "model": MODEL_TRANSLATE,
        "stats": {
            "chars": len(text),
            "words": len(text.split()),
            "out_chars": len(translated),
            "out_words": len(translated.split()),
            "elapsed_s": round(elapsed, 3),
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())