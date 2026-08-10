#!/usr/bin/env python3
"""Render TorchLens block graphs for the three cascade models to SVG.

For each of the default ASR / MT / TTS models, render one SVG per high-level
block (encoder, decoder, etc.) into build/viz/<model>/. No browser is opened;
the SVGs are written for offline inspection. Model ids come from the same env
vars the Makefile uses (MODEL_ASR, MODEL_TRANSLATE, MODEL_TTS), with the same
defaults.

Usage:
    make viz
    make viz MODEL_TTS=facebook/mms-tts-deu
    MODEL_ASR=openai/whisper-base python viz_torchlens.py
"""

import json
import logging
import os
import sys
import warnings

# Keep stdout clean (the script prints one path per SVG) and stderr quiet,
# matching transcribe.py / translate.py / tts.py.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import torch
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    VitsModel,
    WhisperForConditionalGeneration,
)
import torchlens as tl

transformers.logging.set_verbosity_error()

MODEL_ASR = os.environ.get("MODEL_ASR", "openai/whisper-tiny")
MODEL_TRANSLATE = os.environ.get("MODEL_TRANSLATE", "Helsinki-NLP/opus-mt-mul-en")
MODEL_TTS = os.environ.get("MODEL_TTS", "facebook/mms-tts-eng")

VIZ_ROOT = os.path.join("build", "viz")


def slug(model_id: str) -> str:
    """Last path segment of a model id (openai/whisper-tiny -> whisper-tiny)."""
    return model_id.rstrip("/").split("/")[-1]


def render_block(block, args=(), kwargs=None, outpath: str = "") -> str:
    """Run one TorchLens forward pass on `block` and write <outpath>.svg.

    `args` / `kwargs` are replayed into the block's forward. `outpath` is given
    without extension; TorchLens appends .svg for vis_fileformat="svg".
    """
    args = list(args)
    kwargs = kwargs or None
    # A single positional tensor with no kwargs is passed bare (the form TorchLens
    # accepts directly); otherwise pass the arg list.
    input_args = args[0] if (len(args) == 1 and kwargs is None) else args
    tl.show_model_graph(
        block,
        input_args,
        input_kwargs=kwargs,
        renderer="graphviz",
        view="unrolled",
        vis_outpath=outpath,
        vis_fileformat="svg",
        vis_save_only=True,
    )
    return outpath + ".svg"


def viz_whisper(model_id: str, outdir: str) -> list[str]:
    """Whisper: encoder (mel) and decoder (ids + real encoder output)."""
    m = WhisperForConditionalGeneration.from_pretrained(model_id)
    m.eval()
    mel = torch.randn(1, 80, 3000)
    with torch.no_grad():
        enc = m.model.encoder(mel)
    dec_ids = torch.full((1, 4), m.config.decoder_start_token_id, dtype=torch.long)
    produced = []
    produced.append(render_block(m.model.encoder, args=(mel,), outpath=os.path.join(outdir, "encoder")))
    produced.append(
        render_block(
            m.model.decoder,
            kwargs=dict(input_ids=dec_ids, encoder_hidden_states=enc.last_hidden_state),
            outpath=os.path.join(outdir, "decoder"),
        )
    )
    return produced


def viz_marian(model_id: str, outdir: str) -> list[str]:
    """MarianMT: encoder (ids) and decoder (ids + real encoder output)."""
    m = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    m.eval()
    src_ids = torch.randint(1, max(2, m.config.vocab_size), (1, 12), dtype=torch.long)
    with torch.no_grad():
        enc = m.model.encoder(input_ids=src_ids)
    dec_ids = torch.full((1, 4), m.config.decoder_start_token_id, dtype=torch.long)
    produced = []
    produced.append(render_block(m.model.encoder, kwargs=dict(input_ids=src_ids), outpath=os.path.join(outdir, "encoder")))
    produced.append(
        render_block(
            m.model.decoder,
            kwargs=dict(input_ids=dec_ids, encoder_hidden_states=enc.last_hidden_state),
            outpath=os.path.join(outdir, "decoder"),
        )
    )
    return produced


def viz_vits(model_id: str, outdir: str) -> list[str]:
    """VITS: capture each block's input from one inference forward, then render.

    text_encoder / duration_predictor / flow / decoder are on the inference path
    and are called by the model with shape-specific inputs (e.g. text_encoder's
    padding_mask is (B, L, 1)) that are awkward to reconstruct by hand. A forward
    pre-hook captures the exact (args, kwargs) the model uses; each block is then
    rendered in isolation by replaying them. posterior_encoder is training-only,
    so it is fed a dummy spectrogram of the configured width.
    """
    v = VitsModel.from_pretrained(model_id)
    v.eval()
    cfg = v.config
    length = 23
    input_ids = torch.randint(0, cfg.vocab_size, (1, length), dtype=torch.long)
    attn = torch.ones(1, length, dtype=torch.long)

    captured: dict[str, tuple] = {}

    def pre_hook(module, args, kwargs):
        # Tag by identity, since the same hook closure is registered per block.
        captured[id(module)] = (args, kwargs)
        return None

    blocks = ["text_encoder", "duration_predictor", "flow", "decoder"]
    handles = [getattr(v, n).register_forward_pre_hook(pre_hook, with_kwargs=True) for n in blocks]
    try:
        with torch.no_grad():
            v(input_ids=input_ids, attention_mask=attn)
    finally:
        for h in handles:
            h.remove()

    produced = []
    for name in blocks:
        block = getattr(v, name)
        args, kwargs = captured[id(block)]
        produced.append(render_block(block, args=args, kwargs=kwargs, outpath=os.path.join(outdir, name)))

    # posterior_encoder: not on the inference path; dummy spectrogram + mask.
    spec = torch.randn(1, cfg.spectrogram_bins, 64)
    spec_mask = torch.ones(1, 64)
    produced.append(
        render_block(v.posterior_encoder, args=(spec, spec_mask), outpath=os.path.join(outdir, "posterior_encoder"))
    )
    return produced


def main() -> int:
    jobs = [
        ("asr", MODEL_ASR, viz_whisper),
        ("mt", MODEL_TRANSLATE, viz_marian),
        ("tts", MODEL_TTS, viz_vits),
    ]
    produced: list[str] = []
    failures: list[str] = []
    for label, model_id, fn in jobs:
        outdir = os.path.join(VIZ_ROOT, slug(model_id))
        os.makedirs(outdir, exist_ok=True)
        print(f"[{label}] {model_id} -> {outdir}", file=sys.stderr)
        try:
            produced.extend(fn(model_id, outdir))
        except Exception as exc:  # noqa: BLE001 - keep going so one model does not block the rest
            failures.append(f"{label} ({model_id}): {type(exc).__name__}: {exc}")
            print(f"  FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)

    for p in produced:
        print(p)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
    print(f"\n{len(produced)} SVG(s) written under {VIZ_ROOT}/", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())