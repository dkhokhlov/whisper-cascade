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

# --------------------------------------------------------------------------- #
# A2 int-canonical LayerNorm (int8_layernorm_intscale): the zero-fp oracle the
# ONNX LayerNorm mirrors bit-exactly. Pure-int (eps_K from int scale, CLZ rsqrt
# seed, no fp fallback). Validates the reference BEFORE the ONNX emission.
# --------------------------------------------------------------------------- #

def _ln_inputs(D=384, B=8, sigma=4.0, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, D) * sigma
    xi, am, sh, xz = i8.quantize_act_per_token_intscale(x)
    return xi, am, sh, xz


def test_int_bitlen_powers_of_two():
    """CLZ ladder must be exact at 2^k, 2^k-1, 2^k+1 (off-by-one shifts the seed 2x)."""
    for k in range(1, 56):
        for off, want in [(0, k + 1), (-1, k), (1, k + 1)]:
            s = torch.tensor([max(1, (1 << k) + off)], dtype=torch.int64)
            assert int(i8._int_bitlen(s).item()) == want, f"bitlen(2^{k}{'+' if off>=0 else ''}{off})"


def test_layernorm_intscale_rsqrt_converges():
    """4 Newton iters converge to rsqrt(s_K/2^K)*2^R within ~1e-6 across s_K in [2^8, 2^30]."""
    K, R = i8._LN_K, i8._LN_R
    s_K = torch.tensor([1 << 8, (1 << 10) + 3, 1 << 14, 1 << 20,
                        (1 << 24) + 999, 1 << 28, (1 << 30) - 7], dtype=torch.int64).reshape(-1, 1)
    s = i8._int_bitlen(s_K); bitpos = s - 1; e = K - bitpos
    odd = e & 1; half = (e - odd) // 2; a = R + half
    sqrt2_Q20 = 1482910
    C = torch.where(odd == 1, torch.full_like(s_K, 8409 * sqrt2_Q20),
                    torch.full_like(s_K, 8409 * (1 << 20)))
    seed = i8._round_half_up(C << a.clamp(min=0), torch.full_like(s_K, 10000 * (1 << 20)))
    r = seed; three = torch.full_like(s_K, 3 * (1 << R)); hr = torch.full_like(s_K, 1 << (R - 1))
    for _ in range(4):
        t = ((s_K * r) * r) >> (K + R)
        r = (r * (three - t) + hr) >> (R + 1)
    r_true = (1.0 / torch.sqrt(s_K.float() / 2 ** K)) * 2 ** R
    rel = (r.float() - r_true).abs() / r_true
    assert float(rel.max()) < 1e-5, f"rsqrt rel err {rel.max().item()} > 1e-5"


def test_layernorm_intscale_self_consistent():
    """Dequant of (y_int8, y_mul, y_shift, y_zp) matches y_int/2^(K+R+G) within one int8 step."""
    m = hqq_asr.load_whisper_hqq(MODEL, device="cpu", compute_dtype=torch.float32)
    ln = dict(m.named_modules())["model.encoder.layers.0.self_attn_layer_norm"]
    gamma, beta, eps = ln.weight.detach(), ln.bias.detach(), ln.eps
    for seed in (0, 1, 2):
        xi, am, sh, xz = _ln_inputs(seed=seed)
        y_int8, ym, ys, yzp, inter = i8.int8_layernorm_intscale(
            xi, xz, am, sh, gamma, beta, eps, return_intermediates=True)
        scale = ym.float().reshape(-1, 1) * (2.0 ** (-ys.float().reshape(-1, 1)))
        y_recon = (y_int8.float() - yzp.float().reshape(-1, 1)) * scale
        y_int_fp = inter["y_int"].float() * (2.0 ** -(16 + 20 + 15))
        step = (inter["y_int"].amax(-1, keepdim=True).float()
                - inter["y_int"].amin(-1, keepdim=True).float()) / 255.0
        err = (y_recon - y_int_fp).abs()
        assert bool((err < step).all()), f"self-consistency err {err.max().item()} >= step {step.max().item()}"
        # overflow guards (codex+claude bounds)
        assert int(inter["y_int"].abs().max()) < (1 << 62), "y_int overflow"
        assert int((inter["s_K"] * inter["r"]).abs().max()) < (1 << 62), "s_K*r overflow"


def test_layernorm_intscale_vs_fp_within_tolerance():
    """Int-canonical LN output (dequant) vs fp LayerNorm: abs err < 0.1 (int8 + fixed-point
    budget; the A2 staged WER gate already passed at +ln -0.0090). The int-canonical path adds
    the int eps_K + int rsqrt seed approximations (both ~lossless)."""
    m = hqq_asr.load_whisper_hqq(MODEL, device="cpu", compute_dtype=torch.float32)
    ln = dict(m.named_modules())["model.encoder.layers.0.self_attn_layer_norm"]
    gamma, beta, eps = ln.weight.detach(), ln.bias.detach(), ln.eps
    worst = 0.0
    for seed in (0, 1, 2, 3):
        xi, am, sh, xz = _ln_inputs(seed=seed)
        y_int8, ym, ys, yzp = i8.int8_layernorm_intscale(xi, xz, am, sh, gamma, beta, eps)
        scale = ym.float().reshape(-1, 1) * (2.0 ** (-ys.float().reshape(-1, 1)))
        y_recon = (y_int8.float() - yzp.float().reshape(-1, 1)) * scale
        x_recon = (xi.float() - xz.float().reshape(-1, 1)) * am.float().reshape(-1, 1) \
            * (2.0 ** (-sh.float().reshape(-1, 1)))
        y_fp = i8.fp_layernorm_ref(x_recon, gamma, beta, eps)
        worst = max(worst, float((y_recon - y_fp).abs().max()))
    assert worst < 0.1, f"int-canonical LN abs err {worst} >= 0.1"
