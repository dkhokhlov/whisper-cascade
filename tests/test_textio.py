"""Tests for textio.get_text() precedence: TEXT env > file arg > stdin.

Unit tests only: no model, no network.
"""

import os
import sys

import textio


def test_text_env_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["t.py", str(tmp_path / "file.txt")])
    monkeypatch.setenv("TEXT", "from env")
    assert textio.get_text() == "from env"


def test_file_arg_when_no_text_env(monkeypatch, tmp_path):
    f = tmp_path / "in.txt"
    f.write_text("from file", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["t.py", str(f)])
    monkeypatch.delenv("TEXT", raising=False)
    assert textio.get_text() == "from file"


def test_dash_arg_reads_stdin(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["t.py", "-"])
    monkeypatch.delenv("TEXT", raising=False)
    monkeypatch.setattr(sys, "stdin", _Stdin("piped text\n"))
    assert textio.get_text() == "piped text"


def test_stdin_when_nothing_set(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["t.py"])
    monkeypatch.delenv("TEXT", raising=False)
    monkeypatch.setattr(sys, "stdin", _Stdin("  trim me  \n"))
    assert textio.get_text() == "trim me"


def test_empty_input_returns_empty(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["t.py"])
    monkeypatch.setenv("TEXT", "   ")
    monkeypatch.setattr(sys, "stdin", _Stdin(""))
    assert textio.get_text() == ""


class _Stdin:
    """Minimal stdin stub with .read()."""

    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data