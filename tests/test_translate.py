"""Tests for translate.py (make en): stdout is JSON with text + stats.

Unit tests stub the pipeline so they need no model and no network. The
integration test (marked `integration`) loads the real MarianMT model.
"""

import json
import re
import sys

import pytest

import translate


# Skip an integration test only for environmental (network/cache) errors. Any
# other exception is re-raised so a real logic bug fails the test instead of
# being silently skipped (a broad `except: skip` masks implementation bugs).
_ENV_ERR = re.compile(
    r"(connection|offline|local entry|resolve|hostname|timeout|trust_remote"
    r"|unreachable|temporarily unavailable|huggingface|hf_hub|http error)",
    re.IGNORECASE,
)


def _skip_if_env(exc):
    if _ENV_ERR.search(str(exc)) or isinstance(exc, (OSError, ConnectionError)):
        pytest.skip(f"environment/model-load error: {exc}")
    raise exc


# --- unit: JSON object with text, model, stats ------------------------------

def test_translate_prints_json_with_text_and_stats(monkeypatch, capsys):
    def fake_pipeline(task, model):
        assert task == "translation"
        return lambda text: [{"translation_text": "Hello friend"}]

    monkeypatch.setattr(translate, "pipeline", fake_pipeline)
    monkeypatch.setattr(sys, "argv", ["translate.py"])
    monkeypatch.delenv("TEXT", raising=False)
    monkeypatch.setattr(sys, "stdin", _Stdin("Hola amigo"))

    rc = translate.main()
    cap = capsys.readouterr()

    assert rc == 0
    data = json.loads(cap.out)            # stdout is a JSON object
    assert data["text"] == "Hello friend"
    assert data["lang"] == "en"
    assert data["model"] == "Helsinki-NLP/opus-mt-mul-en"
    assert data["stats"]["chars"] == len("Hola amigo")
    assert data["stats"]["out_chars"] == len("Hello friend")
    assert data["stats"]["elapsed_s"] >= 0
    assert cap.err == ""                   # no stderr summary; stats are in JSON


def test_translate_empty_input_returns_1(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["translate.py"])
    monkeypatch.setenv("TEXT", "   ")
    monkeypatch.setattr(sys, "stdin", _Stdin(""))

    assert translate.main() == 1
    assert json.loads(capsys.readouterr().out)["error"]  # JSON error element


def test_translate_inference_error_emits_json_error(monkeypatch, capsys):
    # A runtime error during model load or inference must emit a JSON error and
    # return 1, not a traceback (consistent with asr per-file errors).
    def fake_pipeline(task, model):
        def call(text):
            raise RuntimeError("boom")
        return call

    monkeypatch.setattr(translate, "pipeline", fake_pipeline)
    monkeypatch.setattr(sys, "argv", ["translate.py"])
    monkeypatch.setenv("TEXT", "Hola")
    monkeypatch.setattr(sys, "stdin", _Stdin(""))

    rc = translate.main()
    cap = capsys.readouterr()
    assert rc == 1
    data = json.loads(cap.out)
    assert data["error"] == "boom"
    assert data["lang"] == "en"
    assert data["model"] == "Helsinki-NLP/opus-mt-mul-en"


def test_translate_target_lang_from_env(monkeypatch, capsys):
    def fake_pipeline(task, model):
        return lambda text: [{"translation_text": "Hallo"}]

    monkeypatch.setattr(translate, "pipeline", fake_pipeline)
    monkeypatch.setattr(sys, "argv", ["translate.py"])
    monkeypatch.setenv("TEXT", "Hello")
    monkeypatch.setenv("TARGET_LANG", "de")
    monkeypatch.setattr(translate, "TARGET_LANG", "de")  # main reads module global
    monkeypatch.setattr(sys, "stdin", _Stdin(""))

    rc = translate.main()
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert data["lang"] == "de"


def test_translate_model_from_env(monkeypatch):
    import importlib

    monkeypatch.setenv("MODEL_TRANSLATE", "Helsinki-NLP/opus-mt-es-en")
    importlib.reload(translate)
    assert translate.MODEL_TRANSLATE == "Helsinki-NLP/opus-mt-es-en"
    # Restore the default for the rest of the session.
    monkeypatch.delenv("MODEL_TRANSLATE", raising=False)
    importlib.reload(translate)
    assert translate.MODEL_TRANSLATE == "Helsinki-NLP/opus-mt-mul-en"


# --- integration: real MarianMT model on cached weights (opt-in) -------------

@pytest.mark.integration
def test_translate_spanish_to_english(monkeypatch, capsys):
    import importlib.util

    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers not installed")

    monkeypatch.setattr(sys, "argv", ["translate.py"])
    monkeypatch.setenv("TEXT", "Hola, ¿cómo estás?")
    monkeypatch.delenv("MODEL_TRANSLATE", raising=False)

    try:
        rc = translate.main()
    except Exception as exc:  # noqa: BLE001 - re-raise unless environmental
        _skip_if_env(exc)

    cap = capsys.readouterr()
    assert rc == 0
    data = json.loads(cap.out)
    assert data["text"].strip()  # non-empty English text
    assert data["stats"]["chars"] > 0


class _Stdin:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data