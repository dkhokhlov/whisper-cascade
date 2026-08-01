"""Shared text input helper for the translate and tts scripts.

Both scripts accept the same input forms: the TEXT env var, a file path given
on the command line (or "-" for stdin), or stdin when nothing else is set.
get_text() resolves that precedence and returns the text as one string.
"""

import os
import sys


def get_text() -> str:
    """Return the input text for translate/tts.

    Precedence:
        1. the TEXT env var when set and non-empty;
        2. the first command-line argument (a file path, or "-" for stdin);
        3. stdin.

    Return the text stripped of leading and trailing whitespace. Return "" for
    empty input so the caller can exit with a clear message.
    """
    text = os.environ.get("TEXT", "").strip()
    if text:
        return text

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "-":
            return sys.stdin.read().strip()
        with open(arg, "r", encoding="utf-8") as fh:
            return fh.read().strip()

    return sys.stdin.read().strip()