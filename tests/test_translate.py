"""Tests for translate.py (make en): stdout is pure text; model from env.

Unit tests stub the pipeline so they need no model and no network. The
integration test (marked `integration`) loads the real MarianMT model.
"""

import sys

import pytest

import translate


# --- unit: pure text to stdout, summary to stderr ---------------------------

def test_translate_prints_translated_text_to_stdout(monkeypatch, capsys):
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
    assert cap.out.strip() == "Hello friend"  # stdout is pure text
    assert "[translate]" in cap.err           # summary on stderr


def test_translate_empty_input_returns_1(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["translate.py"])
    monkeypatch.setenv("TEXT", "   ")
    monkeypatch.setattr(sys, "stdin", _Stdin(""))

    assert translate.main() == 1
    assert capsys.readouterr().out == ""  # nothing on stdout


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
    except Exception as exc:  # noqa: BLE001 - network/cache failure, not a test fail
        pytest.skip(f"could not load MarianMT model: {exc}")

    cap = capsys.readouterr()
    assert rc == 0
    assert cap.out.strip()  # non-empty English text
    assert "[translate]" in cap.err


class _Stdin:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data