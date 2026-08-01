"""Tests for tts.py (make tts): output path from env; stats JSON to stdout.

Unit tests stub the model so they need no model and no network. The
integration test (marked `integration`) loads the real VITS model.
"""

import json
import re
import sys

import numpy as np
import pytest

import tts


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


# --- fakes for the model/tokenizer ------------------------------------------

class _FakeTokenizer:
    """Stub tokenizer: from_pretrained returns a callable that echoes text."""

    @staticmethod
    def from_pretrained(name):
        return lambda text, return_tensors="pt": {"input_ids": text}


class _FakeModel:
    """Stub VITS model: returns a fixed 1-second waveform at 16 kHz."""

    @staticmethod
    def from_pretrained(name):
        return _FakeModelInst()


class _FakeModelInst:
    config = type("c", (), {"sampling_rate": 16000})()

    def __call__(self, **kwargs):
        return type("out", (), {"waveform": _FakeWaveform()})()


class _FakeWaveform:
    def squeeze(self):
        return _FakeArray()


class _FakeArray:
    def numpy(self):
        return np.zeros(16000, dtype="float32")


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --- unit: OUTPUT env, JSON summary on stdout -------------------------------

def test_tts_output_path_default(monkeypatch):
    monkeypatch.delenv("OUTPUT", raising=False)
    import importlib
    importlib.reload(tts)
    assert tts.OUTPUT == "tts.wav"


def test_tts_output_path_empty_string_falls_back(monkeypatch):
    # `make tts` exports OUTPUT with no value. An empty string must fall back to
    # the default (the #1 bug: os.environ.get("OUTPUT", "tts.wav") returned "").
    monkeypatch.setenv("OUTPUT", "")
    import importlib
    importlib.reload(tts)
    assert tts.OUTPUT == "tts.wav"
    monkeypatch.delenv("OUTPUT", raising=False)
    importlib.reload(tts)


def test_tts_output_path_from_env(monkeypatch):
    monkeypatch.setenv("OUTPUT", "/tmp/out.wav")
    import importlib
    importlib.reload(tts)
    assert tts.OUTPUT == "/tmp/out.wav"
    monkeypatch.delenv("OUTPUT", raising=False)
    importlib.reload(tts)


def test_tts_empty_input_returns_1(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tts.py"])
    monkeypatch.setenv("TEXT", "   ")
    monkeypatch.setattr(sys, "stdin", _Stdin(""))

    assert tts.main() == 1
    assert json.loads(capsys.readouterr().out)["error"]


def test_tts_writes_wav_and_json(monkeypatch, capsys, tmp_path):
    out_path = tmp_path / "out.wav"
    written = {}

    monkeypatch.setattr(tts, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(tts, "VitsModel", _FakeModel)
    monkeypatch.setattr(tts.torch, "no_grad", lambda: _NoGrad())
    monkeypatch.setattr(tts.sf, "write",
                        lambda path, data, sr: written.update(path=path, sr=sr))

    monkeypatch.setattr(sys, "argv", ["tts.py"])
    monkeypatch.setenv("TEXT", "hello")
    monkeypatch.setattr(tts, "OUTPUT", str(out_path))

    rc = tts.main()
    cap = capsys.readouterr()

    assert rc == 0
    assert written["path"] == str(out_path)
    assert written["sr"] == 16000
    data = json.loads(cap.out)
    assert data["output"] == str(out_path)
    assert data["text"] == "hello"
    assert data["stats"]["duration_s"] == 1.0


# --- integration: real VITS model on cached weights (opt-in) -----------------

@pytest.mark.integration
def test_tts_synthesizes_hello(monkeypatch, capsys, tmp_path):
    import importlib.util

    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers not installed")

    out_path = tmp_path / "tts.wav"
    monkeypatch.setattr(sys, "argv", ["tts.py"])
    monkeypatch.setenv("TEXT", "Hello world")
    monkeypatch.setattr(tts, "OUTPUT", str(out_path))
    monkeypatch.delenv("MODEL_TTS", raising=False)

    try:
        rc = tts.main()
    except Exception as exc:  # noqa: BLE001 - re-raise unless environmental
        _skip_if_env(exc)

    cap = capsys.readouterr()
    assert rc == 0
    assert out_path.exists() and out_path.stat().st_size > 0
    assert json.loads(cap.out)["stats"]["duration_s"] > 0


class _Stdin:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data