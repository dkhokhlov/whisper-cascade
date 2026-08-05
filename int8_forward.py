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
# Eager attention (like the fp16 ONNX export): the attention scale stays a Mul (1/sqrt(d)
# = >>3 here), not an SDPA Sqrt/Div. Int8Attention implements the eager path; SDPA would
# fuse scale+softmax+mask and bypass it. Must be set before hqq_asr import.
os.environ.setdefault("HQQ_ATTN_IMPL", "eager")

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
# Int8LayerNorm / Int8GELU / Int8Conv1d: drop-in int8-compute wrappers (A2/A3/A5)
# --------------------------------------------------------------------------- #


class Int8LayerNorm(nn.Module):
    """int8-compute LayerNorm (A2). Drop-in for nn.LayerNorm: quantizes the input per-token
    to int8, runs the fixed-point mean/var + int rsqrt + fixed-point gamma/beta, returns fp."""

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = ln.normalized_shape
        self.eps = ln.eps
        self.register_buffer("weight", ln.weight.to(torch.float32))
        self.register_buffer("bias", ln.bias.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        D = x.shape[-1]
        x2 = x.reshape(-1, D).to(torch.float32)
        x_int, x_scale, x_zp = i8.quantize_act_per_token(x2)
        y = i8.int8_layernorm(x_int, x_scale, x_zp, self.weight, self.bias, self.eps, stage=4)
        return y.reshape(*lead, D)


class Int8GELU(nn.Module):
    """int8-compute GELU (A3). Drop-in for transformers GELUActivation: quantizes the input
    per-token to int8, runs x*Phi(x) via the int normal-CDF LUT, returns fp."""

    def __init__(self, act):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        D = x.shape[-1]
        x2 = x.reshape(-1, D).to(torch.float32)
        x_int, x_scale, x_zp = i8.quantize_act_per_token(x2)
        y = i8.int8_gelu(x_int, x_scale, x_zp, stage=2)
        return y.reshape(*lead, D)


class Int8Conv1d(nn.Module):
    """int8-compute Conv1d (A5). Drop-in for nn.Conv1d (kernel 3, pad 1): per-BATCH act
    quant (per-token is not factorable across the window), per-channel int8 weight, int32
    window matmul, returns fp."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        w_int, w_scale = i8._quant_weight_per_channel(conv.weight.to(torch.float32))
        self.register_buffer("w_int", w_int.to(torch.int32))
        self.register_buffer("w_scale", w_scale.to(torch.float32))
        self.register_buffer("bias", conv.bias.to(torch.float32) if conv.bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_int, x_scale, x_zp = i8.quantize_act_per_batch(x.to(torch.float32))   # per-batch [B,1,1]
        return i8.int8_conv1d(x_int, x_scale, x_zp, self.w_int, self.w_scale, self.bias,
                              stage=2, stride=self.stride[0], kernel=self.kernel_size[0],
                              padding=self.padding[0])


# --------------------------------------------------------------------------- #
# Int8Attention: int8 softmax (A4) + int8 attention scale (A5) in the eager path
# --------------------------------------------------------------------------- #


class Int8Attention(nn.Module):
    """int8-compute attention (A4 softmax + A5 attention scale), eager path.

    Wraps a WhisperAttention: reuses its q/k/v/out projections (HQQLinear or Int8Linear if
    the matmul block was swapped) and KV-cache logic, but replaces the fp softmax with the
    int8 softmax (quantize scores -> int32 -> exp LUT + int reciprocal) and the fp attention
    scale with the integer 1/sqrt(d_head) (exact >>3 for d_head=64). When neither is enabled
    this is equivalent to the fp eager attention.

    The forward mirrors transformers 4.44.2 WhisperAttention.forward (pinned) with only the
    scale-fold and the softmax replaced, so model.generate()'s KV-cache path is preserved.
    """

    def __init__(self, attn, use_int_softmax: bool, use_int_scale: bool):
        super().__init__()
        self.embed_dim = attn.embed_dim
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scaling = attn.scaling
        self.dropout = attn.dropout
        self.layer_idx = getattr(attn, "layer_idx", None)
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.out_proj = attn.out_proj
        self.use_int_softmax = use_int_softmax
        self.use_int_scale = use_int_scale

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.contiguous().view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                cache_position=None):
        import torch.nn.functional as F
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # Fold the attention scale into q in fp UNLESS the int8 scale path is on (then the
        # exact integer 1/sqrt(d_head) is applied to the int scores below).
        q_in = self.q_proj(hidden_states)
        q_scaled = q_in if self.use_int_scale else q_in * self.scaling
        query_states = self._shape(q_scaled, tgt_len, bsz)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        raw_scores = torch.matmul(query_states, key_states.transpose(2, 3))   # [B,h,Tq,Tk] fp
        valid = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]     # 0 valid, finfo.min masked
            valid = (causal_mask == 0).expand_as(raw_scores)                   # [B,h,Tq,Tk]

        if self.use_int_softmax:
            # Quantize the scores to int32 per row over the VALID entries only: the causal/
            # padding mask adds ~finfo.min to masked positions, and including that in the
            # per-row min/max explodes the scale and collapses the valid scores to a tiny int
            # range (the int8 softmax then leaks into masked positions -> decoder garbage).
            # Fix: scale from valid entries, set masked entries to a large negative int (so
            # subtract-max + exp LUT push them toward 0), then zero + renormalize after softmax.
            lead = raw_scores.shape[:-1]
            K = raw_scores.shape[-1]
            s2 = raw_scores.reshape(-1, K).to(torch.float32)
            if valid is not None:
                v2 = valid.reshape(-1, K)
                BIG = torch.finfo(torch.float32).max / 4.0
                minv = s2.masked_fill(~v2, BIG).amin(-1, keepdim=True)
                maxv = s2.masked_fill(~v2, -BIG).amax(-1, keepdim=True)
            else:
                minv, maxv = s2.amin(-1, keepdim=True), s2.amax(-1, keepdim=True)
            s_scale = ((maxv - minv) / 255.0).clamp(min=1e-8)
            zp = torch.round(-minv / s_scale - 128.0)
            s_int = torch.round(s2 / s_scale + zp).clamp(-128, 127).to(torch.int32)
            if valid is not None:
                s_int = s_int.masked_fill(~v2, -(2 ** 20))
            if self.use_int_scale:
                s_int, s_scale = i8.int_attn_scale(s_int, s_scale, self.head_dim)
            probs = i8.int8_softmax(s_int, s_scale, stage=4).reshape(*lead, K)
            if valid is not None:
                probs = probs * valid.to(torch.float32).reshape(*lead, K)
                probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-8)
            attn_weights = probs
        else:
            attn_weights = raw_scores + (attention_mask[:, :, :, : key_states.shape[-2]]
                                         if attention_mask is not None else 0.0)
            if self.use_int_scale:
                lead = attn_weights.shape[:-1]; K = attn_weights.shape[-1]
                s2 = attn_weights.reshape(-1, K).to(torch.float32)
                s_int, s_scale, _ = i8.quantize_act_per_token(s2)
                s_int, s_scale = i8.int_attn_scale(s_int, s_scale, self.head_dim)
                attn_weights = i8.dequant_act(s_int, s_scale, torch.zeros_like(s_scale)).reshape(*lead, K)
            attn_weights = F.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights, past_key_value


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

    blocks: subset of {"matmul","ln","gelu","sm","conv"} (the staged-WER block names):
      matmul: every HQQLinear -> Int8Linear (A1, per-group int8 matmul).
      ln:     every nn.LayerNorm -> Int8LayerNorm (A2, fixed-point + int rsqrt).
      gelu:   every GELUActivation -> Int8GELU (A3, int sigmoid LUT).
      sm:     every WhisperAttention -> int8 softmax (A4); combined with conv for int scale.
      conv:   every nn.Conv1d -> Int8Conv1d (A5) AND every WhisperAttention uses the integer
              attention scale 1/sqrt(d_head) (A5); the int scale only has effect with sm.
    The swap is in-place. The int8 attention wraps the existing q/k/v/out_proj modules
    (which are Int8Linear if matmul is in blocks, else the original HQQLinear).
    """
    import transformers.models.whisper.modeling_whisper as w

    report = {"matmul": 0, "ln": 0, "gelu": 0, "conv1d": 0, "attn": 0}
    use_int_sm = "sm" in blocks
    use_int_scale = "conv" in blocks
    for name, mod in list(model.named_modules()):
        if "matmul" in blocks and isinstance(mod, HQQLinear):
            _set_child(model, name, Int8Linear(mod)); report["matmul"] += 1
        elif "ln" in blocks and isinstance(mod, nn.LayerNorm):
            _set_child(model, name, Int8LayerNorm(mod)); report["ln"] += 1
        elif "gelu" in blocks and type(mod).__name__ == "GELUActivation":
            _set_child(model, name, Int8GELU(mod)); report["gelu"] += 1
        elif "conv" in blocks and isinstance(mod, nn.Conv1d):
            _set_child(model, name, Int8Conv1d(mod)); report["conv1d"] += 1
        elif (use_int_sm or use_int_scale) and isinstance(mod, w.WhisperAttention):
            _set_child(model, name, Int8Attention(mod, use_int_sm, use_int_scale))
            report["attn"] += 1
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