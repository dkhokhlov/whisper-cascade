"""Tests for transcribe.py input resolution (AUDIO, hf:// URLs, dir/glob).

Unit tests cover the pure logic and need no model and no network. The
integration test (marked `integration`) loads the real Whisper model and runs
only under `make test-integration`; the default `make test` excludes it.
"""

import os
import sys

import pytest

import transcribe


# --- expand_audio: literal, hf://, directory, glob ---------------------------

def test_expand_audio_literal(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"")
    assert transcribe.expand_audio(str(f)) == [(str(f), str(f))]


def test_expand_audio_hf_url_is_a_placeholder():
    # An hf:// token must NOT download here; it is resolved lazily in the loop.
    u = "hf://datasets/Narsil/asr_dummy/1.flac"
    assert transcribe.expand_audio(u) == [(u, u)]


def test_expand_audio_dir_filters_by_extension_and_sorts(tmp_path):
    (tmp_path / "b.flac").write_bytes(b"")
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "ignore.txt").write_bytes(b"")
    got = transcribe.expand_audio(str(tmp_path))
    assert [os.path.basename(p) for _, p in got] == ["a.wav", "b.flac"]


def test_expand_audio_glob(tmp_path):
    (tmp_path / "x1.flac").write_bytes(b"")
    (tmp_path / "x2.flac").write_bytes(b"")
    (tmp_path / "other.ogg").write_bytes(b"")
    got = transcribe.expand_audio(str(tmp_path / "x*.flac"))
    assert [os.path.basename(p) for _, p in got] == ["x1.flac", "x2.flac"]


# --- download_hf_url: URL parsing (hf_hub_download is stubbed) ---------------

def _stub_hub(monkeypatch, seen):
    def fake(repo_id, filename, repo_type):
        seen["repo_id"] = repo_id
        seen["filename"] = filename
        seen["repo_type"] = repo_type
        return "/cached"
    monkeypatch.setattr(transcribe, "hf_hub_download", fake)


def test_download_hf_url_parses_dataset(monkeypatch):
    seen = {}
    _stub_hub(monkeypatch, seen)
    out = transcribe.download_hf_url("hf://datasets/Narsil/asr_dummy/1.flac")
    assert out == "/cached"
    assert seen == {"repo_id": "Narsil/asr_dummy", "filename": "1.flac",
                    "repo_type": "dataset"}


def test_download_hf_url_parses_nested_path(monkeypatch):
    seen = {}
    _stub_hub(monkeypatch, seen)
    transcribe.download_hf_url("hf://datasets/ns/repo/a/b/c.wav")
    assert seen == {"repo_id": "ns/repo", "filename": "a/b/c.wav",
                    "repo_type": "dataset"}


def test_download_hf_url_models_segment(monkeypatch):
    seen = {}
    _stub_hub(monkeypatch, seen)
    transcribe.download_hf_url("hf://models/ns/repo/file.safetensors")
    assert seen == {"repo_id": "ns/repo", "filename": "file.safetensors",
                    "repo_type": "model"}


def test_download_hf_url_default_repo_type_is_model(monkeypatch):
    seen = {}
    _stub_hub(monkeypatch, seen)
    transcribe.download_hf_url("hf://ns/repo/config.json")
    assert seen == {"repo_id": "ns/repo", "filename": "config.json",
                    "repo_type": "model"}


def test_download_hf_url_repo_level_rejected():
    # A repo-level URL is not a single file; reject without a network call.
    with pytest.raises(ValueError):
        transcribe.download_hf_url("hf://datasets/Narsil/asr_dummy")


# --- get_inputs: precedence argv > AUDIO > default sample set ---------------

def test_get_inputs_argv_wins(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["transcribe.py", "x.wav", "y.wav"])
    monkeypatch.setenv("AUDIO", "hf://datasets/ns/repo/z.wav")
    assert transcribe.get_inputs() == [("x.wav", "x.wav"), ("y.wav", "y.wav")]


def test_get_inputs_audio_beats_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["transcribe.py"])
    monkeypatch.setenv("AUDIO", "hf://datasets/ns/repo/a.flac b.wav")
    assert transcribe.get_inputs() == [
        ("hf://datasets/ns/repo/a.flac", "hf://datasets/ns/repo/a.flac"),
        ("b.wav", "b.wav"),
    ]


def test_get_inputs_empty_audio_uses_default_set(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["transcribe.py"])
    monkeypatch.setenv("AUDIO", "   ")
    seen = {}

    def fake(files):
        seen["files"] = files
        return [(f, f"/c/{f}") for f in files]

    monkeypatch.setattr(transcribe, "resolve_samples", fake)
    got = transcribe.get_inputs()
    assert seen["files"] == transcribe.DEFAULT_FILES
    assert got == [(f, f"/c/{f}") for f in transcribe.DEFAULT_FILES]


def test_get_inputs_no_audio_no_argv_uses_default_set(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["transcribe.py"])
    monkeypatch.delenv("AUDIO", raising=False)
    seen = {}

    def fake(files):
        seen["files"] = files
        return [(f, f) for f in files]

    monkeypatch.setattr(transcribe, "resolve_samples", fake)
    transcribe.get_inputs()
    assert seen["files"] == transcribe.DEFAULT_FILES


# --- integration: real model on a cached sample (opt-in) ---------------------

@pytest.mark.integration
def test_main_on_cached_sample(monkeypatch, capsys):
    import glob as _glob
    import json

    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Narsil--asr_dummy/snapshots/*/mlk.flac"
    )
    paths = _glob.glob(pattern)
    if not paths:
        pytest.skip("Narsil/asr_dummy mlk.flac not in HF cache")

    monkeypatch.setattr(sys, "argv", ["transcribe.py"])
    monkeypatch.setenv("AUDIO", paths[0])
    monkeypatch.delenv("MODEL_ASR", raising=False)  # default whisper-tiny

    rc = transcribe.main()
    out = capsys.readouterr().out
    data = json.loads(out)

    assert rc == 0
    assert len(data) == 1
    assert data[0]["text"].strip()
    assert data[0]["stats"]["duration_s"] > 0
    assert data[0]["model"] == "openai/whisper-tiny"