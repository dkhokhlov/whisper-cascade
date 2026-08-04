#!/usr/bin/env python3
"""Measure Word Error Rate (WER) of a Whisper ASR model on a HF dataset.

The script loads the Hugging Face automatic-speech-recognition pipeline once
(the same pipeline as transcribe.py). Then it transcribes a subset of a
Hugging Face audio dataset that has reference transcripts. It computes the
normalized WER with jiwer. It prints one JSON object to stdout with the model,
the dataset, the sample count, the WER, the total audio duration, the total
elapsed time, the average real-time factor, and the first sample references
and hypotheses.

Two dataset sources are supported (dispatched by EVAL_DATASET):

  - google/fleurs            : streamed via datasets, audio bytes are wav,
                               reference field "transcription". 16 kHz.
  - diabolocom/talkbank_4_stt: streamed via datasets, audio bytes are mp3,
                               reference field "transcript". 16 kHz. Use the
                               "segment" split (the "switch" split has long
                               silences and a much higher WER).

Usage:
    python eval_wer.py

Environment:
    MODEL_ASR        ASR model id (default openai/whisper-tiny).
    QUANT            Optional quantization. Set QUANT=hqq to quantize on the
                     fly with HQQ 4-bit (nbits=4, group_size=64). Default: none
                     (load the model as fp32, or load an already-quantized model
                     when MODEL_ASR points at one).
    EVAL_DATASET     HF dataset id (default google/fleurs).
    EVAL_CONFIG      Dataset config / language code (default en_us).
    EVAL_SPLIT       Dataset split (default test). For talkbank use "segment".
    EVAL_LIMIT       Max number of samples. Default 50. Set 0 to use all.
    EVAL_OUT         Optional path to write the JSON result (in addition to
                     stdout).
"""

import io
import gzip
import json
import logging
import os
import sys
import tempfile
import time
import warnings
from itertools import islice

# Silence transformers' noisy logs so stdout is clean JSON and stderr stays
# quiet except for the progress lines this script prints.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
import torchaudio
import transformers
from datasets import Audio, load_dataset
from jiwer import wer as jiwer_wer
from jiwer.transforms import (
    Compose,
    ReduceToListOfListOfWords,
    RemoveMultipleSpaces,
    RemovePunctuation,
    Strip,
    ToLowerCase,
)

import hqq_asr

# Whisper expects 16 kHz mono float32. Both supported sources ship 16 kHz;
# resample is kept as a safety net for any source that does not.
TARGET_SR = 16000

transformers.logging.set_verbosity_error()

MODEL_ASR = os.environ.get("MODEL_ASR", "openai/whisper-tiny")
QUANT = os.environ.get("QUANT", "").strip().lower()
# Compute device: "cpu" (default) keeps the original CPU behavior; "cuda" runs
# the model on GPU (needs CUDA torch, e.g. the .venv-gpu env for the A10).
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cpu").strip().lower() or "cpu"
EVAL_DATASET = os.environ.get("EVAL_DATASET", "google/fleurs")
EVAL_CONFIG = os.environ.get("EVAL_CONFIG", "en_us")
EVAL_SPLIT = os.environ.get("EVAL_SPLIT", "test")
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "50"))
EVAL_OUT = os.environ.get("EVAL_OUT", "").strip()
# Optional forced language (e.g. "spanish"). When set, the pipeline is told
# language + task=transcribe so the model does not auto-detect. Leave unset
# for auto-detect (the default multilingual Whisper behavior).
EVAL_LANG = os.environ.get("EVAL_LANG", "").strip().lower() or None

# WER normalization applied to both the reference and the hypothesis before
# the alignment. Whisper prints capitalized text with punctuation. The fleurs
# reference has a different case and punctuation style. This transform makes
# both sides comparable: lowercase, remove punctuation, collapse spaces.
WER_NORM = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
    ReduceToListOfListOfWords(),
])


def resample(array, sr, target_sr):
    """Resample a mono float32 numpy array from sr to target_sr."""
    t = torch.from_numpy(array).unsqueeze(0)  # (1, T)
    t = torchaudio.functional.resample(t, sr, target_sr)
    return t.squeeze(0).numpy().astype(np.float32)


# Whisper's compression_ratio_threshold (the openai CLI default 2.4), applied
# post-hoc. On short/noisy telephone segments greedy decoding can loop,
# emitting hundreds of repeated words that dominate corpus WER via insertions.
# transformers 4.44.2's in-generation fallback path raises UnboundLocalError
# when return_timestamps is False, so the guard is applied here instead: a
# hypothesis whose gzip compression ratio exceeds the threshold is a
# repetition loop (the model failed), and is treated as empty so it counts as
# deletions on its own reference, not as hundreds of insertions.
LOOP_THRESHOLD = 2.4


def is_loop(text, threshold=LOOP_THRESHOLD):
    """True if text is a repetition loop (gzip compression ratio > threshold)."""
    if not text or len(text) < 12:
        return False
    raw = text.encode("utf-8")
    return len(raw) / max(1, len(gzip.compress(raw))) > threshold


def _decode_bytes(raw, path):
    """Decode audio bytes to (mono float32 numpy array, sample_rate).

    Try soundfile first (wav/flac/ogg via libsndfile). If that fails (e.g. an
    mp3, which libsndfile 1.2.0 cannot read), fall back to torchaudio via a
    temp file.
    """
    try:
        array, sr = sf.read(io.BytesIO(raw))
        if array.ndim > 1:            # stereo or more: average to mono
            array = array.mean(axis=1)
        return array.astype(np.float32), sr
    except Exception:
        suffix = os.path.splitext(path or "")[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(raw)
            tmp = tf.name
        try:
            wav, sr = torchaudio.load(tmp)  # (C, T)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav.squeeze(0).numpy().astype(np.float32), sr
        finally:
            os.unlink(tmp)


def _hf_token():
    return os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")


def _iter_fleurs(dataset, config, split):
    """Stream fleurs; audio bytes are wav, reference field is "transcription"."""
    ds = load_dataset(dataset, config, split=split, streaming=True,
                      token=_hf_token())
    ds = ds.cast_column("audio", Audio(decode=False))
    for ex in ds:
        ref = (ex.get("transcription") or "").strip()
        audio = ex.get("audio")
        if not audio or not audio.get("bytes"):
            yield ref, None, None
            continue
        array, sr = _decode_bytes(audio["bytes"], audio.get("path"))
        yield ref, array, sr


def _iter_talkbank(dataset, config, split):
    """Stream talkbank_4_stt; audio bytes are mp3, reference field is "transcript"."""
    ds = load_dataset(dataset, config, split=split, streaming=True,
                      token=_hf_token())
    ds = ds.cast_column("audio", Audio(decode=False))
    for ex in ds:
        ref = (ex.get("transcript") or "").strip()
        audio = ex.get("audio")
        if not audio or not audio.get("bytes"):
            yield ref, None, None
            continue
        array, sr = _decode_bytes(audio["bytes"], audio.get("path"))
        yield ref, array, sr


def iter_samples(dataset, config, split):
    """Yield (reference, audio_array, sample_rate) for the requested source.

    Dispatch by dataset id. Each source handles its own loading path and
    reference field name; the caller resamples to 16 kHz and runs the pipeline.
    """
    if dataset == "diabolocom/talkbank_4_stt":
        return _iter_talkbank(dataset, config, split)
    return _iter_fleurs(dataset, config, split)


def main() -> int:
    """Run the WER evaluation and print the JSON result.

    Return 0 when at least one sample succeeded. Return 1 when every sample
    failed (so a bad dataset or model does not pass a meaningless result).
    """
    pipe = hqq_asr.build_pipeline(MODEL_ASR, QUANT, device=ASR_DEVICE)

    print(
        f"loading {EVAL_DATASET}/{EVAL_CONFIG} split={EVAL_SPLIT} "
        f"(limit={EVAL_LIMIT or 'all'})",
        file=sys.stderr,
    )
    samples = iter_samples(EVAL_DATASET, EVAL_CONFIG, EVAL_SPLIT)
    if EVAL_LIMIT and EVAL_LIMIT > 0:
        samples = islice(samples, EVAL_LIMIT)

    refs = []
    hyps = []
    details = []
    total_duration = 0.0
    total_elapsed = 0.0
    n_ok = 0
    n_fail = 0

    for ref, array, sr in samples:
        if not ref or array is None or not sr:
            n_fail += 1
            continue
        if sr != TARGET_SR:
            array = resample(array, sr, TARGET_SR)
            sr = TARGET_SR
        duration = len(array) / sr
        total_duration += duration

        try:
            t0 = time.perf_counter()
            call_kwargs = {"array": array, "sampling_rate": sr}
            gen_kwargs = {"language": EVAL_LANG, "task": "transcribe"} if EVAL_LANG else None
            output = pipe(call_kwargs, generate_kwargs=gen_kwargs)
            elapsed = time.perf_counter() - t0
            hyp = output["text"].strip()
            if is_loop(hyp):
                # Repetition loop: the model failed on this segment. Treat as
                # empty so it counts as deletions on its own reference, not as
                # hundreds of insertions that would dominate corpus WER.
                hyp = ""
            total_elapsed += elapsed
            refs.append(ref)
            hyps.append(hyp)
            n_ok += 1
            if len(details) < 5:
                details.append({"ref": ref, "hyp": hyp})
            print(
                f"[{n_ok}] wer={jiwer_wer([ref], [hyp], WER_NORM, WER_NORM):.3f} "
                f"rtf={elapsed / duration:.2f}",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001 - report, do not abort the run
            n_fail += 1
            print(f"[fail] {exc}", file=sys.stderr)

    if n_ok == 0:
        print(json.dumps({"error": "no samples succeeded", "n_fail": n_fail}))
        return 1

    corpus_wer = jiwer_wer(refs, hyps, WER_NORM, WER_NORM)
    avg_rtf = total_elapsed / total_duration if total_duration else None

    result = {
        "model": MODEL_ASR,
        "quant": QUANT or None,
        "dataset": EVAL_DATASET,
        "config": EVAL_CONFIG,
        "split": EVAL_SPLIT,
        "language": EVAL_LANG,
        "n": n_ok,
        "n_fail": n_fail,
        "wer": round(corpus_wer, 4),
        "total_duration_s": round(total_duration, 3),
        "total_elapsed_s": round(total_elapsed, 3),
        "avg_rtf": round(avg_rtf, 3) if avg_rtf is not None else None,
        "samples": details,
    }

    out = json.dumps(result, ensure_ascii=False, indent=2)
    print(out)
    if EVAL_OUT:
        with open(EVAL_OUT, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())