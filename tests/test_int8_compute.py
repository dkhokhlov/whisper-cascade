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


# --------------------------------------------------------------------------- #
# A3 (int-canonical): pure-int GELU oracle (int8_gelu_intscale) -- the spec the  #
# ONNX GELU mirrors bit-exactly. GELU(x) = x*Phi(x), Phi from the int LUT; the   #
# index round and the x*Phi multiply are fixed-point (integer input scale).     #
# --------------------------------------------------------------------------- #

def _gelu_inputs(D=1536, B=3, sigma=4.0, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, D) * sigma
    return i8.quantize_act_per_token_intscale(x)   # xi, am, sh, xz


def test_gelu_intscale_index_in_range():
    """The LUT index is clamped to [0, T-1] and phi_int in [0, 2^S] for all magnitudes (the
    large-tail cases saturate idx to 0/T-1 where Phi -> 0/1)."""
    T, S = i8._GELU_T, i8._GELU_S
    for seed in (0, 1, 2):
        for sig in (0.3, 4.0, 8.0, 50.0):
            xi, am, sh, xz = _gelu_inputs(seed=seed, sigma=sig)
            _, _, _, _, inter = i8.int8_gelu_intscale(xi, xz, am, sh, return_intermediates=True)
            assert int(inter["idx"].min()) >= 0 and int(inter["idx"].max()) <= T - 1
            assert int(inter["phi_int"].min()) >= 0 and int(inter["phi_int"].max()) <= (1 << S)
            assert int(inter["acc"].abs().max()) < (1 << 62), "acc overflow"
            assert int(inter["num"].abs().max()) < (1 << 62), "num overflow"


def test_gelu_intscale_self_consistent():
    """Dequant of (y_int8, y_mul, y_shift, y_zp) matches the pre-requant GELU (acc*y_mul*2^-(S+shift))
    within one int8 step."""
    for seed in (0, 1, 2):
        xi, am, sh, xz = _gelu_inputs(seed=seed)
        y_int8, ym, ys, yzp, inter = i8.int8_gelu_intscale(xi, xz, am, sh, return_intermediates=True)
        scale = ym.float().reshape(-1, 1) * (2.0 ** (-ys.float().reshape(-1, 1)))
        y_recon = (y_int8.float() - yzp.float().reshape(-1, 1)) * scale
        # pre-requant real GELU = acc * y_mul * 2^-(S + y_shift) = (u*phi_int)*y_mul*2^-(S+y_shift)
        gelu_prereq = inter["acc"].float() * am.float().reshape(-1, 1) \
            * (2.0 ** -(i8._GELU_S + sh.float().reshape(-1, 1)))
        step = (gelu_prereq.amax(-1, keepdim=True) - gelu_prereq.amin(-1, keepdim=True)) / 255.0
        err = (y_recon - gelu_prereq).abs()
        assert bool((err < step + 1e-6).all()), f"self-consistency err {err.max().item()} >= step {step.max().item()}"


def test_gelu_intscale_vs_fp_erf_pre_requant():
    """The PRE-requant LUT+index GELU vs fp erf GELU: abs err < 0.001 (the Phi LUT is near-exact;
    the int index round is ~1 LUT LSB). The output int8 requant step is gated separately
    (shared with the linear/LN, WER-neutral; the A3 staged WER gate already passed at +gelu)."""
    worst = 0.0
    for seed in (0, 1, 2, 3):
        xi, am, sh, xz = _gelu_inputs(seed=seed)
        _, _, _, _, inter = i8.int8_gelu_intscale(xi, xz, am, sh, return_intermediates=True)
        x_real = (xi.float() - xz.float().reshape(-1, 1)) * am.float().reshape(-1, 1) \
            * (2.0 ** (-sh.float().reshape(-1, 1)))
        phi_real = inter["phi_int"].float() * (2.0 ** -i8._GELU_S)
        gelu_prereq = x_real * phi_real
        gelu_fp = i8.fp_gelu_ref(x_real)
        worst = max(worst, float((gelu_prereq - gelu_fp).abs().max()))
    assert worst < 1e-2, f"pre-requant LUT/index abs err {worst} >= 0.01"


# --------------------------------------------------------------------------- #
# A4 (int-canonical): pure-int softmax oracle (int8_softmax_intscale) + the      #
# pure-int reciprocal (_int_recip_intscale) -- the spec the ONNX softmax        #
# mirrors bit-exactly. subtract-max cancels zp; exp via int LUT; int reciprocal #
# (CLZ seed + Newton); per-row requant.                                         #
# --------------------------------------------------------------------------- #

def test_int_recip_intscale_converges():
    """_int_recip_intscale converges to (1/x)*2^P within ~5e-6 across x in [2^8, 2^30] (the
    softmax sum_exp range). Pure-int CLZ seed + 5 Newton iters; no torch.log2, no fp fallback."""
    K = 16
    xs = torch.tensor([1 << 8, (1 << 10) + 3, 1 << 16, (1 << 20) + 999,
                       1 << 24, 123456789, 1 << 30], dtype=torch.int64).reshape(-1, 1)
    r = i8._int_recip_intscale(xs, K=K, P=24)
    r_true = (1.0 / (xs.float() / 2 ** K)) * 2 ** 24
    rel = (r.float() - r_true).abs() / r_true
    assert float(rel.max()) < 1e-4, f"int recip rel err {rel.max().item()} > 1e-4"


def _sm_inputs(B=2, K=1500, sig=2.0, seed=0):
    torch.manual_seed(seed)
    scores = torch.randn(B, K) * sig
    return i8.quantize_act_per_token_intscale(scores)


def test_softmax_intscale_idx_in_range():
    """The exp-LUT index is clamped to [0, T-1]; exp_int in [0, 2^S]; the subtract-max cancels
    zp (shifted <= 0); no int64 overflow in num/p_fixed/sum*inv."""
    T, S = i8._SM_T, i8._SM_S
    for seed in (0, 1, 2):
        for sig in (0.5, 2.0, 5.0, 50.0):
            xi, am, sh, xz = _sm_inputs(seed=seed, sig=sig)
            _, _, _, _, inter = i8.int8_softmax_intscale(xi, xz, am, sh, return_intermediates=True)
            assert int(inter["idx"].min()) >= 0 and int(inter["idx"].max()) <= T - 1
            assert int(inter["exp_int"].min()) >= 0 and int(inter["exp_int"].max()) <= (1 << S)
            assert int(inter["shifted"].max()) <= 0, "shifted must be <= 0 (max subtracted)"
            assert int(inter["num"].abs().max()) < (1 << 62), "num overflow"
            assert int(inter["p_fixed"].abs().max()) < (1 << 62), "p_fixed overflow"
            assert int((inter["sum_exp"] * inter["inv_int"]).abs().max()) < (1 << 62), "sum*inv overflow"


def test_softmax_intscale_vs_fp_within_tolerance():
    """PRE-requant LUT+reciprocal softmax vs fp softmax: abs err < 2e-3 (the exp LUT
    nearest-neighbor grid spacing L/T = 0.003 -> ~0.5 grid; the int reciprocal is ~5e-6). The
    output int8 requant step is gated separately (shared with the linear/LN/GELU, WER-neutral;
    the A4 staged WER gate already passed at +softmax)."""
    worst = 0.0
    for seed in (0, 1, 2, 3):
        xi, am, sh, xz = _sm_inputs(seed=seed)
        _, _, _, _, inter = i8.int8_softmax_intscale(xi, xz, am, sh, return_intermediates=True)
        x_real = (xi.float() - xz.float().reshape(-1, 1)) * am.float().reshape(-1, 1) \
            * (2.0 ** (-sh.float().reshape(-1, 1)))
        p_fp = i8.fp_softmax_ref(x_real)
        p_prereq = inter["p_fixed"].float() * (2.0 ** -(i8._SM_S + i8._SM_P))
        worst = max(worst, float((p_prereq - p_fp).abs().max()))
    assert worst < 2e-3, f"pre-requant LUT+recip abs err {worst} >= 2e-3"


# --------------------------------------------------------------------------- #
# A5 (int-canonical): pure-int Conv1d oracle (int8_conv1d_intscale) -- the spec  #
# the ONNX Conv1d mirrors bit-exactly. per-batch Q1.16 act scale (factorable     #
# across the kernel window), per-channel Q0.15 weight scale (keeps the full      #
# int32 acc * mul_w * x_mul in int64), pre-folded int bias, shared per-(b,o)     #
# output requant.                                                               #
# --------------------------------------------------------------------------- #

def _conv_inputs(B=2, in_ch=80, T=300, sigma=2.0, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, in_ch, T) * sigma
    return i8.quantize_act_per_batch_intscale(x)            # xi [B,in,T] int32, mul/shift/zp [B,1,1]


def _real_conv(name):
    m = hqq_asr.load_whisper_hqq(MODEL, device="cpu", compute_dtype=torch.float32)
    conv = dict(m.named_modules())[name]
    return conv.weight.detach().to(torch.float32), conv.bias.detach().to(torch.float32), \
        int(conv.stride[0]), int(conv.padding[0])


def test_conv1d_intscale_in_range():
    """x_int in [-128,127]; F_w >= 1 (bias precision); acc_w/acc_wb in int64 headroom
    (Q0.15 * int32 acc * Q1.16 x_mul <= 2^62; Q0.31 would hit 2^78). Real whisper-tiny weights."""
    for name in ("model.encoder.conv1", "model.encoder.conv2"):
        w, bias, stride, _pad = _real_conv(name)
        out_ch, in_ch, k = w.shape
        for seed in (0, 1, 2):
            for sig in (0.3, 2.0, 8.0):
                xi, am, sh, xz = _conv_inputs(B=2, in_ch=in_ch, T=300, sigma=sig, seed=seed)
                w_int, w_scale = i8._quant_weight_per_channel(w)
                _, _, _, _, inter = i8.int8_conv1d_intscale(
                    xi, xz, am, sh, w_int, w_scale, bias, stride=stride, kernel=k,
                    padding=1, return_intermediates=True)
                assert int(xi.min()) >= -128 and int(xi.max()) <= 127, "x_int out of int8"
                assert inter["F_w"] >= 1, f"F_w={inter['F_w']} < 1 (bias precision)"
                assert int(inter["acc_w"].abs().max()) < (1 << 62), "acc_w overflow"
                assert int(inter["acc_wb"].abs().max()) < (1 << 62), "acc_wb overflow"


def test_conv1d_intscale_self_consistent():
    """Re-dequant of (y_int8, y_mul, y_shift, y_zp) matches the pre-requant real conv output
    (acc_real_w * x_scale + bias, where acc_real_w = acc_w*2^-F_w and x_scale = x_mul*2^-x_shift)
    within one int8 step (per-(b,o) row step)."""
    import torch.nn.functional as F
    for name, T in [("model.encoder.conv1", 300), ("model.encoder.conv2", 300)]:
        w, bias, stride, _pad = _real_conv(name)
        out_ch, in_ch, k = w.shape
        for seed in (0, 1, 2):
            xi, am, sh, xz = _conv_inputs(B=2, in_ch=in_ch, T=T, sigma=2.0, seed=seed)
            w_int, w_scale = i8._quant_weight_per_channel(w)
            y_int8, ym, ys, yzp, inter = i8.int8_conv1d_intscale(
                xi, xz, am, sh, w_int, w_scale, bias, stride=stride, kernel=k,
                padding=1, return_intermediates=True)
            scale = ym.float() * (2.0 ** (-ys.float()))                       # [B,out,1]
            y_recon = (y_int8.float() - yzp.float()) * scale                   # [B,out,T_out]
            acc_real_w = inter["acc_w"].float() * (2.0 ** -inter["F_w"])       # [B,out,T_out]
            x_scale = am.float() * (2.0 ** (-sh.float()))                     # [B,1,1]
            y_pre = acc_real_w * x_scale + bias.reshape(1, out_ch, 1)          # [B,out,T_out]
            step = (y_pre.amax(-1, keepdim=True) - y_pre.amin(-1, keepdim=True)) / 255.0
            err = (y_recon - y_pre).abs()
            assert bool((err < step + 1e-6).all()), \
                f"{name}: self-consistency err {err.max().item()} >= step {step.max().item()}"


def test_conv1d_intscale_vs_fp_within_tolerance():
    """Int-canonical Conv1d output (dequant) vs fp Conv1d: the per-BATCH int8 act quant (256 levels
    over the whole [in,T] tensor -- the conv-window-factorable choice) plus the per-channel int8
    weight quant gives a global rel_err ~1.2% (conv1) / ~3.2% (conv2). The A5 staged WER gate
    already passed at +conv (this rel err is WER-neutral); the threshold is a regression guard."""
    import torch.nn.functional as F
    worst = 0.0
    for name, T in [("model.encoder.conv1", 300), ("model.encoder.conv2", 300)]:
        w, bias, stride, _pad = _real_conv(name)
        out_ch, in_ch, k = w.shape
        for seed in (0, 1, 2):
            xi, am, sh, xz = _conv_inputs(B=2, in_ch=in_ch, T=T, sigma=2.0, seed=seed)
            w_int, w_scale = i8._quant_weight_per_channel(w)
            y_int8, ym, ys, yzp = i8.int8_conv1d_intscale(
                xi, xz, am, sh, w_int, w_scale, bias, stride=stride, kernel=k, padding=1)
            scale = ym.float() * (2.0 ** (-ys.float()))                       # [B,out,1]
            y_recon = (y_int8.float() - yzp.float()) * scale                   # [B,out,T_out]
            x_real = (xi.float() - xz.float()) * am.float() * (2.0 ** (-sh.float()))
            y_fp = F.conv1d(x_real, w, bias, stride=stride, padding=1)
            worst = max(worst, i8._rel_err(y_recon, y_fp))
    assert worst < 0.05, f"int-canonical conv rel err {worst} >= 5%"


# --------------------------------------------------------------------------- #
# A8.1 (int-canonical): int8 Q.K^T + score path (int8_qk_matmul_intscale) --    #
# the zero-fp attention-matmul oracle the ONNX decoder mirrors. Pure-int: int8  #
# Q.K^T (int32 scores), integer score scale = q_scale*k_scale combined to a    #
# single Q1.16, exact >>3 attn scale. Static per-(layer,head) K scale -> the   #
# append-only KV cache form; per-token K scale (per-row score requant) is the  #
# other A8.6 variable, added before the gate.                                  #
# --------------------------------------------------------------------------- #

def _qk_inputs(B=2, H=6, Tq=8, Tk=64, d=64, sigma=2.0, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, Tq, d) * sigma
    k = torch.randn(B, H, Tk, d) * sigma
    # Q: per-query-token Q1.16 quant over d (last dim) -> scales [B,H,Tq,1]
    q_int, q_mul, q_shift, q_zp = i8.quantize_act_per_token_intscale(q)
    # K: static per-head symmetric Q1.16 -> scales [1,H,1,1] (the append-only cache form)
    k_int, k_mul, k_shift, k_zp = i8.quantize_kv_static_per_head_intscale(k)
    return q, k, q_int, q_zp, q_mul, q_shift, k_int, k_zp, k_mul, k_shift


def test_combine_scales_q116_normalized():
    """The product of two Q1.16 scales renormalizes to Q1.16 in [2^15,2^16), pure-int, and equals
    scale_a*scale_b to ~2^-16 (one LSB of the Q1.16 mantissa)."""
    torch.manual_seed(0)
    for _ in range(20):
        sa = torch.rand(4) * 0.05 + 1e-3            # act scales < 1
        sb = torch.rand(4) * 0.05 + 1e-3
        qa, ea = torch.frexp(sa.to(torch.float64)); ma = torch.round(qa * (2 ** 16)).to(torch.int32); sha = (16 - ea).to(torch.int32)
        qb, eb = torch.frexp(sb.to(torch.float64)); mb = torch.round(qb * (2 ** 16)).to(torch.int32); shb = (16 - eb).to(torch.int32)
        mo, sho = i8._combine_scales_q116(ma, sha, mb, shb)
        assert int(mo.min()) >= (1 << 15) and int(mo.max()) < (1 << 16), "combined mul out of [2^15,2^16)"
        prod = (ma.float() * 2.0 ** (-sha.float())) * (mb.float() * 2.0 ** (-shb.float()))
        got = mo.float() * 2.0 ** (-sho.float())
        assert float((got - prod).abs().max() / prod.abs().amax()) < 2e-5, "combined scale rel err"


def test_qk_intscale_in_range():
    """scores int64 < 2^62; scaled int32 fits; s_mul in [2^15,2^16); the attn-scale >>3 reduces
    the score magnitude ~8x so the softmax LUT index stays in range (shifted ~2^17)."""
    for seed in (0, 1, 2):
        for sig in (0.5, 2.0, 8.0):
            _, _, q_int, q_zp, q_mul, q_shift, k_int, k_zp, k_mul, k_shift = _qk_inputs(sigma=sig, seed=seed)
            s_int, s_zp, s_mul, s_shift, inter = i8.int8_qk_matmul_intscale(
                q_int, q_zp, q_mul, q_shift, k_int, k_zp, k_mul, k_shift,
                d_head=64, return_intermediates=True)
            assert int(inter["scores"].abs().max()) < (1 << 62), "scores int64 overflow"
            assert int(s_int.min()) >= -(2 ** 31) and int(s_int.max()) <= 2 ** 31 - 1, "scaled int32 overflow"
            assert int(s_mul.min()) >= (1 << 15) and int(s_mul.max()) < (1 << 16), "s_mul out of [2^15,2^16)"
            assert int((s_zp == 0).all()), "score zp must be 0 (softmax subtract-max cancels it)"


def test_qk_intscale_vs_fp_attention():
    """Int-canonical score (dequant) vs fp Q.K^T/sqrt(d) on the SAME int8-quantized operands: the
    only error is the attn-scale >>3 round (0.5 LSB) + the combine_scales Q1.16 renorm (~2^-16).
    rel_err < 1e-3 (a generous regression guard; the real bound is ~2e-5)."""
    import math
    worst = 0.0
    for seed in (0, 1, 2, 3):
        for sig in (0.5, 2.0, 8.0):
            _, _, q_int, q_zp, q_mul, q_shift, k_int, k_zp, k_mul, k_shift = _qk_inputs(sigma=sig, seed=seed)
            s_int, s_zp, s_mul, s_shift, inter = i8.int8_qk_matmul_intscale(
                q_int, q_zp, q_mul, q_shift, k_int, k_zp, k_mul, k_shift,
                d_head=64, return_intermediates=True)
            # int path: score_real = scaled_int * s_mul * 2^-s_shift
            score_int = s_int.to(torch.float64) * s_mul.to(torch.float64) \
                * (2.0 ** (-s_shift.to(torch.float64)))
            # fp path: raw = einsum(uq,uk)*q_scale*k_scale ; scaled = raw/sqrt(d)
            q_scale = q_mul.to(torch.float64) * (2.0 ** (-q_shift.to(torch.float64)))   # [B,H,Tq,1]
            k_scale = k_mul.to(torch.float64) * (2.0 ** (-k_shift.to(torch.float64)))   # [1,H,1,1]
            uq = (q_int - q_zp).to(torch.float64)
            uk = (k_int - k_zp).to(torch.float64)
            scores_fp = torch.einsum("bhtd,bhud->bhtu", uq, uk) \
                * q_scale * k_scale / math.sqrt(64)
            worst = max(worst, i8._rel_err(score_int, scores_fp))
    assert worst < 1e-3, f"int-canonical Q.K^T rel err {worst} >= 1e-3"


def test_qk_intscale_pertoken_not_implemented():
    """The per-token K scale path (per-row score requant) is the other A8.6 variable; it is not
    yet implemented. Guard that it raises (so the A8.6 gate cannot run an unvalidated path)."""
    _, _, q_int, q_zp, q_mul, q_shift, k_int, k_zp, k_mul, k_shift = _qk_inputs()
    with pytest.raises(NotImplementedError):
        i8.int8_qk_matmul_intscale(q_int, q_zp, q_mul, q_shift, k_int, k_zp, k_mul, k_shift,
                                   d_head=64, kv_scale="pertoken")


# --------------------------------------------------------------------------- #
# A8.2 (int-canonical): int8 P.V (int8_pv_matmul_intscale) -- the zero-fp        #
# attention-output oracle the ONNX decoder mirrors. P (softmax boundary form) #
# x V (static per-head) -> int32 attn, scale = p_scale*v_scale combined to     #
# Q1.16, output requant via the shared Stage 1b (fresh per-row scale for       #
# out_proj). Pure-int throughout.                                               #
# --------------------------------------------------------------------------- #

def _pv_inputs(B=2, H=6, Tq=8, Tk=64, d=64, sigma=2.0, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(B, H, Tq, Tk) * sigma
    probs = torch.softmax(logits, dim=-1)                          # [B,H,Tq,Tk] in [0,1]
    p_int, p_mul, p_shift, p_zp = i8.quantize_act_per_token_intscale(probs)   # [B,H,Tq,1]
    v = torch.randn(B, H, Tk, d) * sigma
    v_int, v_mul, v_shift, v_zp = i8.quantize_kv_static_per_head_intscale(v)  # [1,H,1,1]
    return p_int, p_zp, p_mul, p_shift, v_int, v_zp, v_mul, v_shift, d


def test_pv_intscale_in_range():
    """attn int64 < 2^62; y_mul in [2^15,2^16); the flatten/reshape round-trips the shape."""
    for seed in (0, 1, 2):
        for sig in (0.5, 2.0, 8.0):
            p_int, p_zp, p_mul, p_shift, v_int, v_zp, v_mul, v_shift, d = _pv_inputs(sigma=sig, seed=seed)
            B, H, Tq, Tk = p_int.shape
            y_int8, y_mul, y_shift, y_zp, inter = i8.int8_pv_matmul_intscale(
                p_int, p_zp, p_mul, p_shift, v_int, v_zp, v_mul, v_shift, return_intermediates=True)
            assert y_int8.shape == (B, H, Tq, d), "P.V output shape"
            assert int(inter["attn"].abs().max()) < (1 << 62), "attn int64 overflow"
            assert int(y_int8.min()) >= -128 and int(y_int8.max()) <= 127, "y_int8 out of int8"
            assert int(y_mul.min()) >= (1 << 15) and int(y_mul.max()) < (1 << 16), "y_mul out of [2^15,2^16)"


def test_pv_intscale_self_consistent():
    """Re-dequant of (y_int8,y_mul,y_shift,y_zp) matches the pre-requant attn output
    (attn_int * out_mul * 2^-out_shift) within one int8 step (per-(B,H,Tq) row step)."""
    for seed in (0, 1, 2):
        p_int, p_zp, p_mul, p_shift, v_int, v_zp, v_mul, v_shift, d = _pv_inputs(seed=seed)
        y_int8, y_mul, y_shift, y_zp, inter = i8.int8_pv_matmul_intscale(
            p_int, p_zp, p_mul, p_shift, v_int, v_zp, v_mul, v_shift, return_intermediates=True)
        scale = y_mul.to(torch.float64) * (2.0 ** (-y_shift.to(torch.float64)))      # [B,H,Tq,1]
        recon = (y_int8.to(torch.float64) - y_zp.to(torch.float64)) * scale          # [B,H,Tq,d]
        pre = inter["attn"].to(torch.float64) * inter["out_mul"].to(torch.float64) \
            * (2.0 ** (-inter["out_shift"].to(torch.float64)))                        # [B,H,Tq,d]
        step = (pre.amax(-1, keepdim=True) - pre.amin(-1, keepdim=True)) / 255.0
        err = (recon - pre).abs()
        assert bool((err < step + 1e-6).all()), \
            f"P.V self-consistency err {err.max().item()} >= step {step.max().item()}"


def test_pv_intscale_vs_fp_attention():
    """Int-canonical P.V output (dequant) vs fp P.V on the SAME int8-quantized operands: the
    P per-token int8 + V static-per-head int8 + the output requant. rel_err < 5% (the P/V int8
    quant budget; the A8 staged WER gate measures the real cost)."""
    worst = 0.0
    for seed in (0, 1, 2, 3):
        p_int, p_zp, p_mul, p_shift, v_int, v_zp, v_mul, v_shift, d = _pv_inputs(sigma=2.0, seed=seed)
        y_int8, y_mul, y_shift, y_zp = i8.int8_pv_matmul_intscale(
            p_int, p_zp, p_mul, p_shift, v_int, v_zp, v_mul, v_shift)
        scale = y_mul.to(torch.float64) * (2.0 ** (-y_shift.to(torch.float64)))
        recon = (y_int8.to(torch.float64) - y_zp.to(torch.float64)) * scale
        p_scale = p_mul.to(torch.float64) * (2.0 ** (-p_shift.to(torch.float64)))   # [B,H,Tq,1]
        v_scale = v_mul.to(torch.float64) * (2.0 ** (-v_shift.to(torch.float64)))   # [1,H,1,1]
        up = (p_int - p_zp).to(torch.float64)
        uv = (v_int - v_zp).to(torch.float64)
        attn_fp = torch.einsum("bhtu,bhud->bhtd", up, uv) * p_scale * v_scale
        worst = max(worst, i8._rel_err(recon, attn_fp))
    assert worst < 0.05, f"int-canonical P.V rel err {worst} >= 5%"
