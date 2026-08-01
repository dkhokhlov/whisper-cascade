#!/usr/bin/env python3
"""Translate text to English and print it to stdout.

Reads text from the TEXT env var, a file argument (or "-" for stdin), or stdin,
and writes the English translation to stdout as plain text (so it pipes into
make tts). A one-line summary goes to stderr. The model is a MarianMT model
selected by the MODEL_TRANSLATE env var (default Helsinki-NLP/opus-mt-mul-en,
multilingual to English). The target is named by the target language:
`make en` runs this script; `make es` etc. are one-line Make targets that set
MODEL_TRANSLATE to a different model.

Usage:
    python translate.py [file | -]
    echo "Hola" | python translate.py
    TEXT="Hola" python translate.py
"""

import logging
import os
import sys
import time
import warnings

# Silence transformers' warnings so stdout is pure text and stderr stays quiet
# except the summary line (same pattern as transcribe.py).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import transformers
from transformers import pipeline

transformers.logging.set_verbosity_error()

MODEL_TRANSLATE = os.environ.get("MODEL_TRANSLATE", "Helsinki-NLP/opus-mt-mul-en")


def main() -> int:
    """Run the translation and print the result.

    Return 0 on success, 1 on empty input or a translation error.
    """
    # Import here so unit tests can stub the pipeline before main() runs.
    from textio import get_text

    text = get_text()
    if not text:
        print("[translate] no input text (set TEXT=, pass a file, or pipe stdin)", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    translator = pipeline(task="translation", model=MODEL_TRANSLATE)
    result = translator(text)
    elapsed = time.perf_counter() - t0

    translated = result[0]["translation_text"].strip()
    print(translated)

    print(
        f"[translate] model={MODEL_TRANSLATE} chars={len(text)} "
        f"elapsed={elapsed:.2f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())