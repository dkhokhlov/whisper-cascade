#!/usr/bin/env python3
"""Measure Word Error Rate (WER) of a Whisper ASR model on a HF dataset.

The script loads the Hugging Face automatic-speech-recognition pipeline once
(the same pipeline as transcribe.py). Then it transcribes a subset of a
Hugging Face audio dataset that has reference transcripts. It computes the
normalized WER with jiwer. It prints one JSON object to stdout with the model,
the dataset, the sample count, the WER, the total audio duration, the total
elapsed time, the average real-time factor, and the first sample references
and hypotheses.

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
    EVAL_SPLIT       Dataset split (default test).
    EVAL_LIMIT       Max number of samples. Default 50. Set 0 to use all.
    EVAL_OUT         Optional path to write the JSON result (in addition to
                     stdout).
"""

import io
import json
import logging
import os
import sys
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

transformers.logging.set_verbosity_error()

MODEL_ASR = os.environ.get("MODEL_ASR", "openai/whisper-tiny")
QUANT = os.environ.get("QUANT", "").strip().lower()
EVAL_DATASET = os.environ.get("EVAL_DATASET", "google/fleurs")
EVAL_CONFIG = os.environ.get("EVAL_CONFIG", "en_us")
EVAL_SPLIT = os.environ.get("EVAL_SPLIT", "test")
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "50"))
EVAL_OUT = os.environ.get("EVAL_OUT", "").strip()

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


def main() -> int:
    """Run the WER evaluation and print the JSON result.

    Return 0 when at least one sample succeeded. Return 1 when every sample
    failed (so a bad dataset or model does not pass a meaningless result).
    """
    pipe = hqq_asr.build_pipeline(MODEL_ASR, QUANT)

    print(
        f"loading {EVAL_DATASET}/{EVAL_CONFIG} split={EVAL_SPLIT} "
        f"(limit={EVAL_LIMIT or 'all'})",
        file=sys.stderr,
    )
    ds = load_dataset(
        EVAL_DATASET, EVAL_CONFIG, split=EVAL_SPLIT, streaming=True
    )
    # Decode audio ourselves with soundfile (libsndfile 1.2.0 reads the fleurs
    # wav files). datasets 5.x would otherwise require torchcodec + ffmpeg,
    # which the project does not depend on.
    ds = ds.cast_column("audio", Audio(decode=False))
    if EVAL_LIMIT and EVAL_LIMIT > 0:
        ds = islice(ds, EVAL_LIMIT)

    refs = []
    hyps = []
    details = []
    total_duration = 0.0
    total_elapsed = 0.0
    n_ok = 0
    n_fail = 0

    for ex in ds:
        ref = (ex.get("transcription") or "").strip()
        audio = ex.get("audio")
        if not ref or not audio or not audio.get("bytes"):
            n_fail += 1
            continue
        array, sr = sf.read(io.BytesIO(audio["bytes"]))
        if array.ndim > 1:            # stereo or more: average to mono
            array = array.mean(axis=1)
        array = array.astype(np.float32)
        duration = len(array) / sr if sr else 0.0
        total_duration += duration

        try:
            t0 = time.perf_counter()
            output = pipe({"array": array, "sampling_rate": sr})
            elapsed = time.perf_counter() - t0
            hyp = output["text"].strip()
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