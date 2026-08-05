"""Phase A6: int8-compute Whisper forward + staged-WER harness.

Wraps the per-block int8 reference (`int8_compute.py`, validated A1-A5) into drop-in
`nn.Module` replacements and swaps them into a loaded HQQ Whisper model, so the normal
`model.generate()` path runs the int8-compute graph. The WER is then measured one block
at a time (A1 matmul -> +A2 layernorm -> +A3 GELU -> +A4 softmax -> +A5 conv/attn-scale)
against the same-model fp (HQQ dequant) baseline + the 4-bit reference manifest, isolating
which fixed-point block costs WER. This is the torch decision gate before Phase B (ONNX).

Block enable set (`--blocks`): a comma list of {matmul,ln,gelu,sm,conv}. A1-only is the
default first stage (`matmul`); each added block swaps one more op family to int8.
"""

from __future__ import annotations

import os

os.environ.setdefault("HQQ_COMPUTE_DTYPE", "fp32")  # fp reference == the fp32 benchmark

import torch
import torch.nn as nn

import int8_compute as i8
from hqq.core.quantize import HQQLinear

# --------------------------------------------------------------------------- #
# Int8Linear: drop-in replacement for an HQQLinear (A1 per-group int8 matmul)   #
# --------------------------------------------------------------------------- #


class Int8Linear(nn.Module):
    """int8-compute Linear wrapping an HQQLinear's stored params.

    Unpacks W_q to int8 levels once (i8.unpack_levels), holds the per-group scale/zero
    (fp32, [N,1] flat as HQQ stores them) and bias, and runs the per-group int8 matmul
    (i8.int8_matmul_fixed_point) with per-token dynamic activation quant. Drop-in for the
    HQQLinear forward (x [*, in_features] -> [*, out_features]).
    """

    def __init__(self, hqq: HQQLinear):
        super().__init__()
        meta = hqq.meta
        self.in_features = hqq.in_features
        self.out_features = hqq.out_features
        self.group_size = int(meta["group_size"])
        self.shape = tuple(int(s) for s in meta["shape"])
        self.nbits = int(meta["nbits"])
        self.packing = meta["packing"]
        self.axis = int(meta["axis"])
        # unpack once -> int8 levels [O, I] int32 (the MatMulInteger weight)
        with torch.no_grad():
            qr = i8.unpack_levels(
                hqq.W_q, self.nbits, self.packing, self.shape, self.group_size, self.axis
            )
        self.register_buffer("qr", qr.to(torch.int32))                 # [O, I]
        self.register_buffer("zero", meta["zero"].to(torch.float32).reshape(-1, 1))   # [N,1]
        self.register_buffer("scale", meta["scale"].to(torch.float32).reshape(-1, 1)) # [N,1]
        self.register_buffer("bias", hqq.bias.to(torch.float32) if hqq.bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        I = x.shape[-1]
        x2 = x.reshape(-1, I).to(torch.float32)                          # [B, I]
        x_int, x_scale, x_zp = i8.quantize_act_per_token(x2)            # per-token over I
        out = i8.int8_matmul_fixed_point(
            x_int, x_scale, x_zp, self.qr, self.zero, self.scale, self.bias, self.group_size
        )                                                              # [B, O] fp32
        return out.reshape(*lead, self.out_features)


# --------------------------------------------------------------------------- #
# Staged swap: replace selected op families with int8-compute modules         #
# --------------------------------------------------------------------------- #


def _set_child(parent: nn.Module, dotted: str, child: nn.Module) -> None:
    """setattr a dotted module path (e.g. 'self_attn.q_proj') on parent."""
    *prefix, leaf = dotted.split(".")
    for p in prefix:
        parent = getattr(parent, p)
    setattr(parent, leaf, child)


def swap_to_int8(model: nn.Module, blocks: set[str]) -> dict:
    """Swap selected op families in `model` to int8-compute modules. Returns a report dict.

    blocks: subset of {"matmul","ln","gelu","sm","conv"}.
      matmul: every HQQLinear -> Int8Linear (A1).
      ln/gelu/conv/sm: TODO (A2-A5 wrappers); raise NotImplementedError if requested.
    The swap is in-place; the original fp modules are returned in the report for restore.
    """
    report = {"matmul": 0, "skipped": []}
    if "matmul" in blocks:
        for name, mod in list(model.named_modules()):
            if isinstance(mod, HQQLinear):
                wrapped = Int8Linear(mod)
                _set_child(model, name, wrapped)
                report["matmul"] += 1
    for b in ("ln", "gelu", "sm", "conv"):
        if b in blocks:
            report["skipped"].append(f"{b}: not implemented yet (A2-A5 wrappers)")
    return report


# --------------------------------------------------------------------------- #
# Per-layer validation: Int8Linear vs the HQQLinear fp forward                 #
# --------------------------------------------------------------------------- #


def validate_int8_linear(model_dir: str, n_layers: int = 8, seed: int = 0) -> None:
    """For the first n_layers HQQLinears, compare Int8Linear.forward to the HQQ fp forward
    (the dequant+matmul that the 4-bit/fp32 benchmark runs) on random input. Expected per-
    layer rel_err in the A1 range (~1.7-5%)."""
    import hqq_asr

    torch.manual_seed(seed)
    model = hqq_asr.WhisperHQQModel.from_quantized(model_dir, device="cpu", compute_dtype=torch.float32).eval()
    gen = torch.manual_seed(seed)
    print(f"Int8Linear vs HQQ fp forward (first {n_layers} HQQLinears), {model_dir}:")
    print(f"{'layer':<52} {'nbits':>5} {'rel_err':>10}")
    print("-" * 72)
    seen = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, HQQLinear) or seen >= n_layers:
                if seen >= n_layers:
                    break
                continue
            I, O = mod.in_features, mod.out_features
            x = torch.randn(4, 19, I, generator=gen) * 1.0              # [B,T,I] like a hidden
            fp = mod(x.to(mod.compute_dtype))                          # HQQ fp dequant+matmul
            wrapped = Int8Linear(mod)
            out = wrapped(x)
            err = i8._rel_err(out, fp.float())
            print(f"{name:<52} {mod.meta['nbits']:>5} {err:>10.2e}")
            seen += 1


# --------------------------------------------------------------------------- #
# Staged WER: fp baseline vs int8 stage on the same samples (A6 decision gate) #
# --------------------------------------------------------------------------- #


def _run_pass(pipe, samples, n, lang):
    """Run `pipe` over up to n (ref, array, sr) samples. Returns (refs, hyps)."""
    import time
    from itertools import islice
    import eval_wer as e

    refs, hyps = [], []
    if n and n > 0:
        samples = islice(samples, n)
    for ref, array, sr in samples:
        if not ref or array is None or not sr:
            continue
        if sr != e.TARGET_SR:
            array = e.resample(array, sr, e.TARGET_SR)
            sr = e.TARGET_SR
        try:
            gen_kwargs = {"language": lang, "task": "transcribe"} if lang else None
            output = pipe({"array": array, "sampling_rate": sr}, generate_kwargs=gen_kwargs)
            hyp = output["text"].strip()
            if e.is_loop(hyp):
                hyp = ""
        except Exception as exc:  # noqa: BLE001
            print(f"  sample fail: {exc}", file=__import__("sys").stderr)
            continue
        refs.append(ref)
        hyps.append(hyp)
    return refs, hyps


def run_wer(model_dir: str, blocks: set[str], n: int, dataset: str, config: str,
            split: str, lang) -> None:
    """Build the HQQ pipeline once, run the fp baseline (no swap), then swap `blocks`
    to int8 and re-run on the SAME samples. Report both corpus WERs + the paired delta.

    The fp baseline (all-fp HQQ dequant compute) is the same-storage anchor: it isolates
    the int8-compute cost from the storage-bit effect. The 4-bit manifest is a separate
    cross-model reference (not run here)."""
    import json, sys
    from jiwer import wer as jiwer_wer
    import eval_wer as e
    import hqq_asr

    print(f"loading {model_dir} (fp baseline) ...", file=sys.stderr)
    pipe = hqq_asr.build_pipeline(model_dir, quant="hqq", device="cpu", compute_dtype=torch.float32)

    # materialize the sample list once so both passes see identical samples
    samples = list(e.iter_samples(dataset, config, split))
    print(f"fp baseline pass over {n} samples ...", file=sys.stderr)
    refs, fp_hyps = _run_pass(pipe, samples, n, lang)
    fp_wer = jiwer_wer(refs, fp_hyps, e.WER_NORM, e.WER_NORM)
    print(f"  fp baseline WER = {fp_wer:.4f} over {len(refs)} samples", file=sys.stderr)

    report = swap_to_int8(pipe.model, blocks)
    print(f"swapped: {report}", file=sys.stderr)
    if report["skipped"]:
        print("  aborting: requested block not implemented: " + ", ".join(report["skipped"]))
        return
    print(f"int8 ({sorted(blocks)}) pass over the same {len(refs)} samples ...", file=sys.stderr)
    _, i8_hyps = _run_pass(pipe, samples, n, lang)
    n_cmp = min(len(i8_hyps), len(fp_hyps))
    i8_wer = jiwer_wer(refs[:n_cmp], i8_hyps[:n_cmp], e.WER_NORM, e.WER_NORM)
    delta = i8_wer - fp_wer
    n_match = sum(1 for a, b in zip(fp_hyps[:n_cmp], i8_hyps[:n_cmp]) if a == b)
    result = {
        "model": model_dir,
        "blocks": sorted(blocks),
        "n_samples": n_cmp,
        "fp_wer": round(fp_wer, 4),
        "int8_wer": round(i8_wer, 4),
        "delta_abs": round(delta, 4),
        "text_identical_to_fp": n_match,
        "text_identical_frac": round(n_match / n_cmp, 4) if n_cmp else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    import argparse, sys

    p = argparse.ArgumentParser(description="int8-compute forward harness (A6)")
    p.add_argument("--model", default="build/whisper-tiny-hqq-2bit")
    p.add_argument("--validate", action="store_true", help="per-layer Int8Linear vs HQQ fp")
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wer", action="store_true", help="staged WER: fp baseline vs int8 stage")
    p.add_argument("--blocks", default="matmul", help="comma list: matmul,ln,gelu,sm,conv")
    p.add_argument("--n", type=int, default=20, help="number of fleurs samples")
    p.add_argument("--dataset", default="google/fleurs")
    p.add_argument("--config", default="en_us")
    p.add_argument("--split", default="test")
    p.add_argument("--lang", default="", help="forced language (e.g. spanish); empty=auto")
    args = p.parse_args()
    if args.validate:
        validate_int8_linear(args.model, args.n_layers, args.seed)
        return
    if args.wer:
        blocks = {b.strip() for b in args.blocks.split(",") if b.strip()}
        run_wer(args.model, blocks, args.n, args.dataset, args.config, args.split,
                args.lang or None)
        return
    p.print_help()


if __name__ == "__main__":
    main()