"""Tests for eval_wer helpers: the post-hoc repetition-loop guard and the
8 kHz -> 16 kHz resampler (the mu-bench telephone path).

Unit tests only: no model load, no network. Importing eval_wer pulls
torch/torchaudio/transformers/datasets (all in the venv) but no model and
no network call.
"""

import numpy as np
import pytest

import eval_wer


# --- is_loop: the post-hoc gzip compression-ratio guard --------------------
# A looping hypothesis (the failure mode that inflated talkbank WER) must be
# detected; a normal transcription must not be flagged.

def test_is_loop_flags_repetition():
    loop = "my dad ain't my dad ain't " * 20
    assert eval_wer.is_loop(loop) is True


def test_is_loop_clears_normal_text():
    assert eval_wer.is_loop("I have a dream that one day this nation will rise up") is False


def test_is_loop_short_text_is_false():
    # Below the 12-char floor the guard is a no-op (would misfire on noise).
    assert eval_wer.is_loop("hi there") is False


def test_is_loop_empty_is_false():
    assert eval_wer.is_loop("") is False
    assert eval_wer.is_loop(None) is False


def test_is_loop_threshold_boundary():
    # A text just above the 2.4 ratio is a loop; one below is not. The exact
    # ratio is gzip-dependent, so only assert the two sides classify right.
    assert eval_wer.is_loop("word " * 200, threshold=2.0) is True
    assert eval_wer.is_loop("word " * 200, threshold=100.0) is False


# --- resample: the 8 kHz -> 16 kHz mu-bench path --------------------------

def test_resample_doubles_length_8k_to_16k():
    # 1 second of 8 kHz mono float32 -> 16000 samples at 16 kHz.
    arr = np.sin(np.linspace(0, 2 * np.pi * 440, 8000, endpoint=False)).astype(np.float32)
    out = eval_wer.resample(arr, 8000, 16000)
    assert out.dtype == np.float32
    assert out.shape == (16000,)


def test_resample_noop_when_rates_equal():
    arr = np.zeros(1000, dtype=np.float32)
    out = eval_wer.resample(arr, 16000, 16000)
    assert out.shape == (1000,)


def test_resample_downsample_16k_to_8k():
    arr = np.zeros(16000, dtype=np.float32)
    out = eval_wer.resample(arr, 16000, 8000)
    assert out.shape == (8000,)