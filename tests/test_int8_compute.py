"""Stage 1b int-canonical output requant (int8_compute.int8_output_requant_intscale).

Validates the zero-fp boundary requant: given the per-group matmul accumulator (int64 @
2^-F) + the Q1.16 input act scale + bias, it produces (y_int8, y_mul, y_shift, y_zp) all
in integer. Checks:
  (a) self-consistency: re-dequant (y_int8 - y_zp)*y_mul*2^-y_shift == the fp out_fp within
      one int8 step (abs err < 2^-6);
  (b) y_mul normalized to [2^15, 2^16);
  (c) differential vs the A7.1 float32 reference (quantize_act_per_token_intscale(out_fp),
      half-to-even): <= 0.1% of y_int8 differ by 1 LSB (ties + float32-vs-exact-int range).
Runs under .venv (no onnx needed).
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HQQ_COMPUTE_DTYPE", "fp32")
os.environ.setdefault("HQQ_ATTN_IMPL", "eager")

import int8_compute as i8  # noqa: E402
import hqq_asr  # noqa: E402
from hqq.core.quantize import HQQLinear  # noqa: E402

MODEL = os.environ.get("INT8_ONNX_MODEL", "build/whisper-tiny-hqq-3bit")


def _acc(mod, xi, am, sh, xz):
    """Rebuild the int64 accumulator @ 2^-F (mirror int8_matmul_fixed_point int-act path)."""
    O, I = (int(s) for s in mod.meta["shape"])
    g = int(mod.meta["group_size"])
    G = I // g
    qr = i8.unpack_levels(mod.W_q, int(mod.meta["nbits"]), mod.meta["packing"], (O, I), g, 1).to(torch.int32)
    A, Bt, C, D = i8._per_group_int_terms(xi, qr, g)
    scale_g = mod.meta["scale"].to(torch.float32).reshape(O, G)
    mul_s, sh_s = i8.fixed_point_per_group(scale_g)
    mul_z, sh_z = i8.fixed_point_per_group(scale_g * mod.meta["zero"].to(torch.float32).reshape(O, G))
    F = int(min(sh_s.amin().item(), sh_z.amin().item()))
    x_zp = xz.to(torch.int32).reshape(-1, 1, 1)
    T1 = (A - x_zp * Bt.unsqueeze(0)).to(torch.int64)
    T2 = (C.unsqueeze(1) - x_zp * D).to(torch.int64)
    p1 = i8._rshift_round(T1 * mul_s.unsqueeze(0).to(torch.int64), (sh_s - F).unsqueeze(0).to(torch.int64))
    p2 = i8._rshift_round(T2 * mul_z.unsqueeze(0).to(torch.int64), (sh_z - F).unsqueeze(0).to(torch.int64))
    return (p1 - p2).sum(-1).to(torch.int64), F


def _first_layers(n=4):
    m = hqq_asr.load_whisper_hqq(MODEL, device="cpu", compute_dtype=torch.float32)
    out = [(n, mod) for n, mod in m.named_modules()
           if isinstance(mod, HQQLinear) and int(mod.meta["nbits"]) in (3, 8)]
    return out[:n]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_output_requant_self_consistent(seed):
    """Re-dequant of (y_int8, y_mul, y_shift, y_zp) matches out_fp within one int8 step."""
    torch.manual_seed(seed)
    for _name, mod in _first_layers(4):
        I = int(mod.meta["shape"][1])
        x = torch.randn(8, I) * 2.0
        xi, am, sh, xz = i8.quantize_act_per_token_intscale(x)
        acc, F = _acc(mod, xi, am, sh, xz)
        out_fp = i8.int8_matmul_fixed_point(
            xi, None, xz,
            i8.unpack_levels(mod.W_q, int(mod.meta["nbits"]), mod.meta["packing"],
                             (int(mod.meta["shape"][0]), I), int(mod.meta["group_size"]), 1).to(torch.int32),
            mod.meta["zero"].to(torch.float32).reshape(-1, 1),
            mod.meta["scale"].to(torch.float32).reshape(-1, 1),
            mod.bias, int(mod.meta["group_size"]), act_mul=am, act_shift=sh)
        y, mul, sh_o, zp = i8.int8_output_requant_intscale(acc, am, sh, F, mod.bias)
        # (a) y_mul normalized to [2^15, 2^16)
        assert int(mul.min()) >= (1 << 15) and int(mul.max()) < (1 << 16), "y_mul out of [2^15,2^16)"
        # (b) re-dequant == out_fp within one int8 step (per-token step = range/255; the
        # round-to-nearest int8 bin gives <= 0.5 step; allow 1.0 step for the Q1.16 scale round)
        scale = mul.to(torch.float64).reshape(-1, 1) * (2.0 ** (-sh_o.to(torch.float64).reshape(-1, 1)))
        recon = (y.to(torch.float64) - zp.to(torch.float64).reshape(-1, 1)) * scale
        err = (recon - out_fp.to(torch.float64)).abs()
        step = (out_fp.amax(-1, keepdim=True).to(torch.float64)
                - out_fp.amin(-1, keepdim=True).to(torch.float64)) / 255.0
        assert bool((err < step).all()), f"{_name}: re-dequant err {err.max().item()} >= step {step.max().item()}"


@pytest.mark.parametrize("seed", [0, 1])
def test_output_requant_differential(seed):
    """vs the A7.1 float32 reference: <= 0.1% of y_int8 differ by 1 LSB (ties)."""
    torch.manual_seed(seed)
    tot_y = tot_n = 0
    worst = 0
    for _name, mod in _first_layers(4):
        I = int(mod.meta["shape"][1])
        x = torch.randn(8, I) * 2.0
        xi, am, sh, xz = i8.quantize_act_per_token_intscale(x)
        acc, F = _acc(mod, xi, am, sh, xz)
        out_fp = i8.int8_matmul_fixed_point(
            xi, None, xz,
            i8.unpack_levels(mod.W_q, int(mod.meta["nbits"]), mod.meta["packing"],
                             (int(mod.meta["shape"][0]), I), int(mod.meta["group_size"]), 1).to(torch.int32),
            mod.meta["zero"].to(torch.float32).reshape(-1, 1),
            mod.meta["scale"].to(torch.float32).reshape(-1, 1),
            mod.bias, int(mod.meta["group_size"]), act_mul=am, act_shift=sh)
        y, _, _, _ = i8.int8_output_requant_intscale(acc, am, sh, F, mod.bias)
        y_ref, _, _, _ = i8.quantize_act_per_token_intscale(out_fp)
        d = (y != y_ref).sum().item()
        tot_y += d
        tot_n += y.numel()
        worst = max(worst, int((y - y_ref).abs().max()))
    rate = tot_y / tot_n
    assert rate < 1e-3, f"y_int8 mismatch rate {rate} exceeds 0.1% (ties)"
    assert worst <= 1, f"y_int8 worst |d| {worst} > 1 LSB"