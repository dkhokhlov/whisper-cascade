"""Int8-compute reference for the HQQ Whisper models (Phase A).

The deployment hardware has no fp16 and no fp32 unit, so the deploy target is an
int8-compute graph: int8 weights, int8 activations, int32 accumulators, zero fp16/fp32.
HQQ is weight-only with a per-group scale (group_size=32, axis=1); it has no int8-compute
path. This module is the torch reference for that path -- the spec the ONNX emission
(Phase B) mirrors, and the harness that validates WER incrementally before any ONNX.

Staged validation (A1, the per-group int8 matmul):
  Stage 0 (reference): fp dequant (qr - zero)*scale, fp matmul. The ground truth.
  Stage 1 (int8 weight, fp act, fp scale): the HQQ levels fit losslessly into int8
    (4/8 levels << 256), so reconstructing W_fp from the int8 levels is bit-exact.
    Confirms int8 WEIGHT storage is lossless.
  Stage 2 (+ int8 act): per-token dynamic int8 activation. Error = act-quant error.
  Stage 3 (int8 matmul, int32 accumulate, per-group fp scale): the per-group int
    decomposition (MatMulInteger over per-group reshapes + per-group scale). Algebraically
    identical to Stage 2 -- validates the decomposition.
  Stage 4 (fixed-point scale): replace the fp per-group scale with an int multiplier +
    right-shift. Error += fixed-point scale rounding.

Later phases add A2 layernorm, A3 GELU, A4 softmax, A5 conv/act-scale, A6 full forward
with one-block-at-a-time WER. This file grows stage by stage.
"""

from __future__ import annotations

import os

import torch

# HQQ compute dtype for loading the quantized model. fp32 here is fine -- this module
# is the MATH reference; the int8 path it implements is dtype-agnostic (all integer).
_COMPUTE_DTYPE = torch.float32


# --------------------------------------------------------------------------- #
# HQQ unpack (mirror of hqq.core.bitpack.unpack_*; validated against hqq below) #
# --------------------------------------------------------------------------- #

def unpack_levels(W_q: torch.Tensor, nbits: int, packing: str, shape, group_size: int, axis: int) -> torch.Tensor:
    """Return the HQQ integer levels (uint8/int8) for a packed W_q, shape [O, I].

    Mirrors hqq.core.bitpack.unpack_* so the ONNX emission (Phase B) can reuse the same
    layout. axis=1, group_size along I (the repo default).
    """
    O, I = shape
    if packing == "8bit_u8":
        qr = W_q.to(torch.int32)
    elif packing == "2bit_u8":
        # 4 values per uint8, MSB-first, 4 contiguous row slabs (bitpack.py:54-64).
        q = W_q.to(torch.int32)
        step = q.shape[0]
        tmp = torch.empty((4 * step, q.shape[1]), dtype=torch.int32, device=q.device)
        tmp[0 * step:1 * step] = (q & 0b11000000) >> 6
        tmp[1 * step:2 * step] = (q & 0b00110000) >> 4
        tmp[2 * step:3 * step] = (q & 0b00001100) >> 2
        tmp[3 * step:4 * step] = q & 0b00000011
        qr = tmp
    elif packing == "3bit_32":
        # 10 values per int32 (bitpack.py:94-110), packed from bit 27 down. Padded to a
        # multiple of 10; truncated to N below.
        q = W_q.to(torch.int32)
        step = q.shape[0]
        tmp = torch.empty((10 * step, q.shape[1]), dtype=torch.int32, device=q.device)
        shifts = [27, 24, 21, 18, 15, 12, 9, 6, 3, 0]
        for k, sh in enumerate(shifts):
            tmp[k * step:(k + 1) * step] = (q >> sh) & 0b111
        qr = tmp
    else:
        raise NotImplementedError(f"unpack_levels: packing={packing} not supported")
    # qr is [M, group_size] where M >= N = O*I/group_size (3-bit pads M up to a multiple
    # of 10; 2-bit and 8-bit have M == N). Truncate to N rows BEFORE reshaping -- hqq
    # truncates the unpacked tensor to N (quantize.py:190-195).
    N = (O * I) // group_size
    qr = qr[:N]
    if axis == 1:
        qr = qr.reshape(O, I // group_size, group_size).reshape(O, I)
    else:
        raise NotImplementedError("unpack_levels: only axis=1 supported (repo default)")
    return qr.to(torch.int32)


def hqq_fp_weight(meta: dict) -> torch.Tensor:
    """Reference fp weight via the stored meta (the HQQ dequant formula)."""
    O, I = meta["shape"]
    g = meta["group_size"]
    # scale/zero are per-group [O, I/group_size] for axis=1.
    scale = meta["scale"].to(torch.float32)
    zero = meta["zero"].to(torch.float32)
    # Reconstruct the integer levels then dequant -- matches hqq Quantizer.dequantize.
    qr = unpack_levels(meta["_W_q"], int(meta["nbits"]), meta["packing"], meta["shape"], g, int(meta["axis"]))
    qr = qr.reshape(O, I // g, g).to(torch.float32)
    scale = scale.reshape(O, I // g, 1)
    zero = zero.reshape(O, I // g, 1)
    w = (qr - zero) * scale
    return w.reshape(O, I)


# --------------------------------------------------------------------------- #
# Activation quantization (per-token dynamic int8)                              #
# --------------------------------------------------------------------------- #

def quantize_act_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token asymmetric int8 quantization along the last dim (the feature dim).

    Returns (x_int8 [-128,127], x_scale [*,1]). x_fp ~= (x_int8 - x_zp) * x_scale.
    """
    feat = x.shape[-1]
    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    scale = (xmax - xmin) / 255.0
    scale = scale.clamp(min=1e-8)
    zp = torch.round(-xmin / scale - 128.0)
    x_int = torch.round(x / scale + zp).clamp(-128, 127).to(torch.int32)
    return x_int, scale, zp


def dequant_act(x_int: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    return (x_int.to(torch.float32) - zp) * scale


# --------------------------------------------------------------------------- #
# A1: per-group int8 matmul, staged                                            #
# --------------------------------------------------------------------------- #

def fp_matmul_ref(x: torch.Tensor, w_fp: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Stage 0: fp reference. x [B,I], w_fp [O,I] -> [B,O]."""
    out = x @ w_fp.t()
    if bias is not None:
        out = out + bias
    return out


def _per_group_int_terms(x_int, qr, group_size):
    """The four int32 terms of the per-group int8 matmul decomposition.

    A[b,o,g] = sum_{i in g} qr[o,i]*x_int[b,i]   (int32 -- the MatMulInteger partial)
    B[o,g]   = sum_{i in g} qr[o,i]              (int32 -- per-group weight sum)
    C[b,g]   = sum_{i in g} x_int[b,i]           (int32 -- per-group activation sum)
    D        = group_size                         (const int)
    """
    O, I = qr.shape
    G = I // group_size
    B = x_int.shape[0]
    qr_g = qr.reshape(O, G, group_size).to(torch.int32)
    xg = x_int.reshape(B, G, group_size).to(torch.int32)
    A = torch.einsum("ogi,bgi->bog", qr_g, xg)   # [B, O, G] int32
    Bt = qr_g.sum(dim=-1)                         # [O, G] int32
    C = xg.sum(dim=-1)                            # [B, G] int32
    return A, Bt, C, group_size


def int8_matmul_per_group(
    x_int: torch.Tensor, x_scale: torch.Tensor, x_zp: torch.Tensor,
    qr: torch.Tensor, zero: torch.Tensor, scale: torch.Tensor,
    bias: torch.Tensor | None, group_size: int,
) -> torch.Tensor:
    """Stage 3: per-group int8 matmul, int32 accumulate, per-group fp scale + fp zero.

    qr [O,I] int8 levels; zero/scale [O, I/g] fp per-group (HQQ zero is a FP zero-point,
    not int); x_int [B,I] int8; x_scale/x_zp [B,1] (x_zp int).

    out[b,o] = x_scale[b] * sum_g scale[o,g] * [ (A - x_zp[b]*B) - zero[o,g]*(C - x_zp[b]*D) ] + bias[o]

    Algebraically identical to (dequant_act(x) @ dequant_weight.T) -- Stage 2 -- so the
    Stage 3 vs Stage 0 error isolates nothing new beyond Stage 2; it validates the
    per-group int decomposition (the MatMulInteger + reshape math the ONNX graph emits).
    """
    A, Bt, C, D = _per_group_int_terms(x_int, qr, group_size)
    O, I = qr.shape
    G = I // group_size
    zero_g = zero.reshape(O, G).to(torch.float32)                # HQQ stores [N,1] flat
    scale_g = scale.reshape(O, G).to(torch.float32)
    x_zp = x_zp.to(torch.int32).reshape(-1, 1, 1)               # [B,1,1] for broadcast
    # term[b,o,g] = (A - x_zp*B) - zero*(C - x_zp*D), in fp (Stage 3 keeps scale/zero fp)
    A_f = A.to(torch.float32)
    B_f = Bt.to(torch.float32).unsqueeze(0)                      # [1,O,G]
    C_f = C.to(torch.float32).unsqueeze(1)                       # [B,1,G]
    term = (A_f - x_zp * B_f) - zero_g.unsqueeze(0) * (C_f - x_zp * float(D))
    out = (term * scale_g.unsqueeze(0)).sum(dim=-1) * x_scale    # [B,O]
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out


def _rshift_round(v: torch.Tensor, sh) -> torch.Tensor:
    """Arithmetic right shift with round-half-up (signed int). v is a signed int tensor;
    sh is a tensor OR a python int (broadcast with v). Returns the same shape/dtype as v.

    A plain >> truncates toward -inf; under the heavy per-group cancellation in the matmul
    sum (|out| ~1 while individual prod terms ~1e3), truncation error up to 1 LSB per group
    accumulates to ~10% of the output. Round-half-up: always add +2^(sh-1) then arithmetic
    shift (the floor of the shift gives nearest for both signs -- the sign-dependent bias
    is a common bug that rounds negatives away from nearest).
    """
    if isinstance(sh, int):
        half = 1 << max(sh - 1, 0)
        return torch.bitwise_right_shift(v + half, sh)
    half = torch.bitwise_left_shift(torch.ones_like(sh), (sh - 1).clamp(min=0))
    return torch.bitwise_right_shift(v + half, sh)


def fixed_point_per_group(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group fixed-point (gemmlowp/TFLite scheme): scale ≈ mul * 2^left_shift.

    Returns (mul [same shape] int32, right_shift [same shape] int32). mul is normalized
    to ~[2^30, 2^31) (a Q0.31 fixed-point), so each group keeps ~30 bits of precision
    regardless of magnitude. The shift is PER-GROUP -- a shared shift cannot cover the
    per-group scale dynamic range (~10-100x within a layer, see A1 diagnostics) without
    zeroing small-scale groups. ONNX BitShift accepts a per-element shift tensor, so this
    is emittable. HQQ scales are all < 1, so left_shift <= 0 and a right-shift suffices.
    """
    s = scale.to(torch.float64)
    q, exp = torch.frexp(s)                       # s = q * 2^exp, q in [0.5, 1) for s>0
    mul = torch.round(q * (2 ** 31)).clamp(0, 2**31 - 1).to(torch.int32)
    left_shift = exp - 31                          # <= 0 for s < 1
    right_shift = (-left_shift).clamp(min=0).to(torch.int32)
    return mul, right_shift


def int8_matmul_fixed_point(
    x_int: torch.Tensor, x_scale: torch.Tensor, x_zp: torch.Tensor,
    qr: torch.Tensor, zero: torch.Tensor, scale: torch.Tensor,
    bias: torch.Tensor | None, group_size: int,
) -> torch.Tensor:
    """Stage 4: per-group fp scale AND fp zero as fixed-point, with fractional-bit retention.

    A true zero-fp graph needs both scale and zero as int. The per-group prod terms cancel
    heavily across groups (|out|~1 while individual prods ~1e3), so shifting each group to
    its final magnitude BEFORE summing loses the low bits that the cancellation needs. Fix:
    align every group to a common 2^-F fractional scale (F = min shift across scale+zero),
    sum in int64, then one final round-shift by F. F = min(sh) means every group does a
    right-shift by (sh - F) >= 0 -- no left-shift, no overflow -- and retains F ~ 22+ frac
    bits through the sum, driving the per-group error to ~G * 2^-F (negligible).

    out[b,o] = x_scale[b] * rshift_round( sum_g [ p1 - p2 ], F ) + bias
      p1[o,g,b] = rshift_round( T1 * mul_scale, sh_scale - F )    # T1 = A - x_zp*B  (int32)
      p2[o,g,b] = rshift_round( T2 * mul_zero,  sh_zero  - F )    # T2 = C - x_zp*D  (int32)
    """
    A, Bt, C, D = _per_group_int_terms(x_int, qr, group_size)        # int32
    O, I = qr.shape
    G = I // group_size
    x_zp = x_zp.to(torch.int32).reshape(-1, 1, 1)                    # [B,1,1] for broadcast
    zero_g = zero.reshape(O, G).to(torch.float32)                    # HQQ stores [N,1] flat
    scale_g = scale.reshape(O, G).to(torch.float32)
    # The zero correction is zero*scale*T2 (= zscale*T2), NOT zero*T2 -- the per-group
    # scale multiplies the whole (T1 - zero*T2) term (see Stage 3). Fold scale into zero
    # so p1 = scale*T1 and p2 = zscale*T2 have matching magnitude (both ~scale*group*|x|).
    zscale_g = (zero_g * scale_g).to(torch.float32)
    mul_s, sh_s = fixed_point_per_group(scale_g)
    mul_z, sh_z = fixed_point_per_group(zscale_g)
    # Common fractional bits = the smallest shift (largest-magnitude group); all others
    # right-shift down to it. Per-tensor scalar F (one final shift for the whole layer).
    F = int(min(sh_s.amin().item(), sh_z.amin().item()))
    F_t = torch.tensor(F, dtype=torch.int64)
    A_i = A                                                          # [B,O,G] int32
    B_i = Bt.to(torch.int32).unsqueeze(0)                            # [1,O,G]
    C_i = C.to(torch.int32).unsqueeze(1)                             # [B,1,G]
    T1 = (A_i - x_zp * B_i).to(torch.int64)                          # [B,O,G] int64
    T2 = (C_i - x_zp * D).to(torch.int64)                            # [B,1,G] int64
    sh1 = (sh_s - F).unsqueeze(0).to(torch.int64)                    # [1,O,G] right-shift
    sh2 = (sh_z - F).unsqueeze(0).to(torch.int64)
    p1 = _rshift_round(T1 * mul_s.unsqueeze(0).to(torch.int64), sh1)  # [B,O,G] int64 @ 2^-F
    p2 = _rshift_round(T2 * mul_z.unsqueeze(0).to(torch.int64), sh2)  # [B,1,G] int64 @ 2^-F
    acc = (p1 - p2).sum(dim=-1).to(torch.int64)                      # [B,O] int64 @ 2^-F
    out = _rshift_round(acc, F_t).to(torch.float32) * x_scale        # final requant to int
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out


# --------------------------------------------------------------------------- #
# A2: int8 LayerNorm (fixed-point mean/var + integer rsqrt + fixed-point g/b)   #
# --------------------------------------------------------------------------- #

# Fixed-point fraction bits for the layernorm intermediates.
_LN_K = 16   # mean/var (Q16)
_LN_R = 20   # rsqrt result (Q20)
_LN_G = 15   # gamma/beta (Q15)


def fp_layernorm_ref(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """Stage 0: fp LayerNorm ground truth. x [*,D], gamma/beta [D]."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) * torch.rsqrt(var + eps) * gamma + beta


def _int_rsqrt(s_K: torch.Tensor, K: int = _LN_K, R: int = _LN_R, iters: int = 4) -> torch.Tensor:
    """Integer reciprocal-square-root via MSB seed + Newton, all int64.

    s_K [*,1] int64 is s_real * 2^K (s_real > 0). Returns r_int [*,1] int64 = rsqrt(s_real)*2^R.
    Newton: t = s*r^2 (want 1.0); r <- r*(3-t)/2. In fixed point:
      t_int = (s_K * r^2) >> (K + R)         # t_real*2^R, want ~2^R
      r_new = (r * (3*2^R - t_int) + 2^R) >> (R+1)   # round-half-up
    Seed from the MSB position (exponent only; Newton cleans up the mantissa).
    """
    s_K = s_K.to(torch.int64)
    # MSB position = floor(log2(s_K)). s_K < 2^53 here so float64 is exact; Phase B will
    # replace this with a pure-int compare ladder (emittable as ONNX Compare/Where).
    bitpos = torch.floor(torch.log2(s_K.to(torch.float64))).to(torch.int64)
    bitpos = torch.where(s_K > 0, bitpos, torch.zeros_like(bitpos))
    e = K - bitpos                                     # s_real ~ 2^(-e) * mantissa
    half = torch.div(e, 2, rounding_mode="floor")      # e//2 (can be negative)
    odd = (e - 2 * half).to(torch.int64)               # 0 or 1
    sqrt2 = round((2.0 ** 0.5) * (2 ** R))             # sqrt(2) in Q_R, python int
    # seed = 0.8409 * 2^(R + half) * (sqrt2 if odd else 2^R)/2^R ... = 0.8409 * 2^(R+half) * (sqrt2/2^R if odd)
    # build seed as int64 tensor
    base = torch.where(
        (R + half) >= 0,
        torch.bitwise_left_shift(torch.ones_like(s_K), (R + half).clamp(min=0)),
        torch.zeros_like(s_K),
    )
    # if R+half < 0, base would be 0; recover by using float for those rare tiny-s cases
    seed = (base * 8409 // 10000)                      # *0.8409
    seed = torch.where(odd == 1, seed * sqrt2 // (2 ** R), seed)
    # fallback for negative exponent (very large s_K): compute seed in fp64 then round
    neg = (R + half) < 0
    if neg.any():
        s_real = s_K.to(torch.float64) / (2.0 ** K)
        seed_fp = torch.rsqrt(s_real) * (2.0 ** R)
        seed = torch.where(neg, seed_fp.round().to(torch.int64), seed)
    r = seed
    three = torch.tensor(3 * (2 ** R), dtype=torch.int64, device=r.device)
    for _ in range(iters):
        t = torch.bitwise_right_shift(s_K * r * r, K + R)               # s*r^2 in Q_R
        r = torch.bitwise_right_shift(r * (three - t) + (1 << (R - 1)), R + 1)  # round-half-up
    return r


def int8_layernorm(
    x_int: torch.Tensor, x_scale: torch.Tensor, x_zp: torch.Tensor,
    gamma: torch.Tensor, beta: torch.Tensor, eps: float, stage: int,
    K: int = _LN_K, R: int = _LN_R, G: int = _LN_G,
) -> torch.Tensor:
    """Staged int8 LayerNorm. x_int [B,D] int8 (per-token), x_scale/x_zp [B,1].

    Works in the centered u = x_int - x_zp space, where x_scale cancels in the
    normalization: y = (u - mean_u)*rsqrt(var_u + eps')*gamma + beta, eps' = eps/x_scale^2.

    stage 1: fp stats + fp rsqrt + fp g/b on the int-reconstructed input (input-quant error).
    stage 2: fixed-point int mean/var (QK); fp rsqrt + fp g/b.       (fixed-point stats)
    stage 3: + integer rsqrt (Newton); fp g/b.                       (rsqrt approx)
    stage 4: + fixed-point g/b (QG) + per-token requant to int8, dequant for comparison.
    """
    B, D = x_int.shape
    u = (x_int.to(torch.int32) - x_zp.to(torch.int32))                # [B,D] int32, centered
    if stage == 1:
        x_recon = (u.to(torch.float32)) * x_scale                    # dequant (x_scale cancels below)
        return fp_layernorm_ref(x_recon, gamma, beta, eps)
    # integer sums (exact, int32/int64)
    S1 = u.sum(dim=-1, keepdim=True).to(torch.int64)                 # [B,1] sum(u)
    S2 = (u * u).sum(dim=-1, keepdim=True).to(torch.int64)           # [B,1] sum(u^2), <= D*16384
    if stage == 2:
        mean_u = S1.to(torch.float32) / D
        var_u = S2.to(torch.float32) / D - mean_u * mean_u
        eps_p = eps / (x_scale.to(torch.float32) ** 2)
        r = torch.rsqrt(var_u + eps_p)
        y = (u.to(torch.float32) - mean_u) * r * gamma + beta
        return y
    # fixed-point mean/var (QK), int64
    mean_K = torch.div(S1 << K, D, rounding_mode="floor")            # [B,1] QK
    var_K = torch.div(S2 << K, D, rounding_mode="floor") - torch.bitwise_right_shift(mean_K * mean_K, K)
    eps_p = eps / (x_scale.to(torch.float32) ** 2)                   # [B,1] fp
    eps_K = (eps_p * (2 ** K)).round().to(torch.int64)               # [B,1] QK
    s_K = var_K + eps_K                                              # [B,1] QK (>0 via eps)
    if stage in (3, 4):
        # int rsqrt (Newton); stage 3 stops here with fp g/b, stage 4 adds fixed-point g/b + requant
        r_int = _int_rsqrt(s_K, K=K, R=R)                            # [B,1] QR int64
        if stage == 3:
            r = r_int.to(torch.float32) / (2 ** R)
            mean_u = mean_K.to(torch.float32) / (2 ** K)
            y = (u.to(torch.float32) - mean_u) * r * gamma + beta
            return y
        # stage 4: fixed-point g/b + requant
        gamma_int = (gamma.to(torch.float64) * (2 ** G)).round().to(torch.int64)   # [D] QG
        beta_int = (beta.to(torch.float64) * (2 ** G)).round().to(torch.int64)     # [D] QG
        # y_int (Q(K+R+G)) = (u-mean_u)*r*gamma + beta, all int64
        uK = (u.to(torch.int64) << K) - mean_K                       # [B,D] QK  (u - mean_u)
        y_int = uK * r_int * gamma_int                               # [B,D] Q(K+R+G) (broadcast r_int[B,1], gamma[D])
        y_int = y_int + (beta_int << (K + R))                        # + beta at Q(K+R+G)
        y = y_int.to(torch.float32) / (2 ** (K + R + G))             # real output
        # per-token requant to int8 (the next op's input)
        y_int8, y_scale, y_zp = quantize_act_per_token(y)
        return dequant_act(y_int8, y_scale, y_zp)                    # dequant for comparison vs fp ref
    raise ValueError(f"int8_layernorm: stage {stage} not in 1..4")


# --------------------------------------------------------------------------- #
# A3: int8 GELU (sigmoid-1.702 approx via integer sigmoid LUT)                  #
# --------------------------------------------------------------------------- #

# Sigmoid LUT: covers s = 1.702*x in [-L, L]; beyond it sigmoid saturates to <1 LSB.
_GELU_L = 12          # half-range in s-units (sigmoid(12) ~ 1 - 6e-7, > 16-bit LSB)
_GELU_T = 4096        # LUT entries over [-L, L]  (grid spacing ~0.006 in s)
_GELU_S = 16          # sigmoid value fixed-point bits (sig in [0, 2^S])


def _sigmoid_lut() -> torch.Tensor:
    """Int32 LUT [_GELU_T]: round(sigmoid(s_grid) * 2^_GELU_S), s_grid in [-L, L]."""
    import math
    L, T, S = _GELU_L, _GELU_T, _GELU_S
    s_grid = torch.linspace(-L, L, T, dtype=torch.float64)
    sig = torch.sigmoid(s_grid)
    return (sig * (2 ** S)).round().to(torch.int32)


def fp_gelu_ref(x: torch.Tensor) -> torch.Tensor:
    """Stage 0: exact erf GELU (what Whisper's GELUActivation uses)."""
    return torch.nn.functional.gelu(x, approximate="none")


def int8_gelu(x_int: torch.Tensor, x_scale: torch.Tensor, x_zp: torch.Tensor, stage: int) -> torch.Tensor:
    """Staged int8 GELU via GELU(x) ~= x * sigmoid(1.702*x), sigmoid from an int LUT.

    x_int [B,D] int8 (per-token, from the fc1 output requant), x_scale/x_zp [B,1].
    stage 1: exact erf GELU on the int-reconstructed input.      (input-quant error)
    stage 2: sigmoid-1.702 LUT GELU (fp output).                 (approx + LUT quant)
    stage 3: + per-token requant to int8, dequant for compare.   (output quant)
    """
    L, T, S = _GELU_L, _GELU_T, _GELU_S
    u = (x_int.to(torch.float32) - x_zp.to(torch.float32))          # centered int
    x_real = u * x_scale                                             # [B,D] real input
    if stage == 1:
        return fp_gelu_ref(x_real)
    # s = 1.702 * x_real ; index = round(s * T/(2L)) + T/2, clamped
    s = 1.702 * x_real
    idx = torch.clamp((s * (T / (2 * L))).round().to(torch.int64) + T // 2, 0, T - 1)
    lut = _sigmoid_lut().to(x_real.device)
    sig_int = lut[idx]                                               # [B,D] int32, sig in Q_S
    sig = sig_int.to(torch.float32) / (2 ** S)
    gelu = x_real * sig                                              # sigmoid-1.702 approx
    if stage == 2:
        return gelu
    # stage 3: requant to int8 (the fc2 input), dequant for comparison
    g_int, g_scale, g_zp = quantize_act_per_token(gelu)
    return dequant_act(g_int, g_scale, g_zp)


# --------------------------------------------------------------------------- #
# A4: int8 softmax (subtract-max + exp LUT + int reciprocal + requant)         #
# --------------------------------------------------------------------------- #

_SM_T = 4096       # exp LUT entries over [-L, 0]
_SM_L = 12          # exp(-12) = 6.1e-6 < 1 LSB at Q15
_SM_S = 15          # exp value fixed-point bits (exp in [0, 2^S], 1.0 = 2^S)
_SM_P = 24          # reciprocal result fixed-point bits
_SM_R = 20          # requant output frac bits


def _exp_lut() -> torch.Tensor:
    """Int32 LUT [_SM_T]: round(exp(s_grid) * 2^_SM_S), s_grid in [-L, 0]."""
    L, T, S = _SM_L, _SM_T, _SM_S
    s_grid = torch.linspace(-L, 0.0, T, dtype=torch.float64)
    return (torch.exp(s_grid) * (2 ** S)).round().to(torch.int32)


def _int_recip(x_int: torch.Tensor, K: int, P: int = _SM_P, iters: int = 5) -> torch.Tensor:
    """Integer reciprocal 1/x via MSB seed + Newton, all int64.

    x_int [*,1] int64 is x_real * 2^K (x_real > 0). Returns r_int [*,1] = (1/x_real)*2^P.
    Newton: t = x*r (want 1.0); r <- r*(2 - t). In fixed point:
      t_int = (x_int * r) >> K              # x*r in Q_P, want ~2^P
      r_new = (r * (2*2^P - t_int) + 2^(P-1)) >> P   # round-half-up
    """
    x_int = x_int.to(torch.int64)
    bitpos = torch.floor(torch.log2(x_int.to(torch.float64))).to(torch.int64)
    bitpos = torch.where(x_int > 0, bitpos, torch.zeros_like(bitpos))
    # seed r ~ 2^e / mantissa, e = P + K - bitpos, mantissa in [1,2) -> 1/mantissa in (0.5,1].
    # Reciprocal uses the FULL exponent (no halving -- that was the rsqrt seed's trick).
    e = P + K - bitpos
    base = torch.where(
        e >= 0,
        torch.bitwise_left_shift(torch.ones_like(x_int), e.clamp(min=0)),
        torch.zeros_like(x_int),
    )
    seed = (base * 7071 // 10000)        # *0.7071 (geometric mean of 1/mantissa in (0.5,1])
    neg = e < 0
    if neg.any():
        seed_fp = (2.0 ** P) / (x_int.to(torch.float64) / (2.0 ** K))
        seed = torch.where(neg, seed_fp.round().to(torch.int64), seed)
    r = seed
    two = torch.tensor(2 * (2 ** P), dtype=torch.int64, device=r.device)
    for _ in range(iters):
        t = torch.bitwise_right_shift(x_int * r, K)                  # x*r in Q_P
        r = torch.bitwise_right_shift(r * (two - t) + (1 << (P - 1)), P)  # round-half-up
    return r


def fp_softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """Stage 0: fp softmax over the last dim."""
    return torch.softmax(x, dim=-1)


def int8_softmax(scores_int: torch.Tensor, score_scale: torch.Tensor, stage: int) -> torch.Tensor:
    """Staged int8 attention softmax over the last dim (K).

    scores_int [..., K] int32 (per-row dequant scale `score_scale` [..., 1], fp). The
    scores are the QK^T/sqrt(d) accumulator from the A1 matmul (kept int32 -- no requant
    before the reduction). subtract-max is exact int; exp via LUT; sum exact int; the
    reciprocal is the only non-trivial int op.
    stage 1: fp softmax on the int-reconstructed scores.        (input/score-quant error)
    stage 2: int exp LUT, fp reciprocal, fp output.              (exp LUT error)
    stage 3: + int reciprocal (Newton).                         (reciprocal error)
    stage 4: + requant p to int8, dequant for compare.           (output quant)
    """
    L, T, S, P, R = _SM_L, _SM_T, _SM_S, _SM_P, _SM_R
    scores_real = scores_int.to(torch.float32) * score_scale       # [..., K]
    if stage == 1:
        return fp_softmax_ref(scores_real)
    max_int = scores_int.amax(dim=-1, keepdim=True)                # [...,1] int32
    shifted_int = scores_int - max_int                             # [..., K] int32, <= 0
    shifted_real = shifted_int.to(torch.float32) * score_scale     # [..., K] real, <= 0
    idx = torch.clamp((shifted_real * (T / L)).round().to(torch.int64) + T - 1, 0, T - 1)
    lut = _exp_lut().to(scores_int.device)
    exp_int = lut[idx]                                             # [..., K] Q_S int32, in [0, 2^S]
    sum_exp = exp_int.sum(dim=-1, keepdim=True).to(torch.int64)    # [...,1] int64
    if stage == 2:
        probs = exp_int.to(torch.float32) / sum_exp.to(torch.float32)
        return probs
    # int reciprocal of sum_exp (sum_exp is already a Q_S int -> treat as Q_S, K=S)
    inv_int = _int_recip(sum_exp, K=S, P=P)                        # [...,1] Q_P int64
    if stage == 3:
        probs = (exp_int.to(torch.int64) * inv_int).to(torch.float32) / (2 ** (S + P))
        return probs
    # stage 4: requant to int8 (the p@V matmul input), dequant for comparison
    p_fixed = exp_int.to(torch.int64) * inv_int                    # [..., K] Q(S+P)
    p_real = p_fixed.to(torch.float32) / (2 ** (S + P))
    p_int, p_scale, p_zp = quantize_act_per_token(p_real)
    return dequant_act(p_int, p_scale, p_zp)


# --------------------------------------------------------------------------- #
# A5: int8 conv (per-channel int8 weight, standard QLinearConv math) + attn scale
# --------------------------------------------------------------------------- #

def fp_conv1d_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Stage 0: fp Conv1d. x [B, in, T], weight [out, in, k], bias [out]."""
    return torch.nn.functional.conv1d(x, weight, bias, padding=1)


def _quant_weight_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 quantization. w [out, in, k] -> (w_int8, w_scale[out])."""
    w_scale = w.abs().amax(dim=(1, 2)) / 127.0                 # [out]
    w_scale = w_scale.clamp(min=1e-8)
    w_int = torch.round(w / w_scale[:, None, None]).clamp(-127, 127).to(torch.int32)
    return w_int, w_scale


def quantize_act_per_batch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-batch asymmetric int8 quantization over (in, T) -- one scale per batch element.

    Used for Conv1d inputs: the kernel window mixes 3 time positions, so a per-token
    (per-spatial-position) scale is NOT factorable out of the window sum. A per-batch
    scale is constant across the window -> factors out cleanly. Returns (x_int8, scale,
    zp) all shaped [B, 1, 1] (broadcasting over in, T).
    """
    xmin = x.amin(dim=(1, 2), keepdim=True)
    xmax = x.amax(dim=(1, 2), keepdim=True)
    scale = (xmax - xmin) / 255.0
    scale = scale.clamp(min=1e-8)
    zp = torch.round(-xmin / scale - 128.0)
    x_int = torch.round(x / scale + zp).clamp(-128, 127).to(torch.int32)
    return x_int, scale, zp


def int8_conv1d(x_int: torch.Tensor, x_scale: torch.Tensor, x_zp: torch.Tensor,
                w_int: torch.Tensor, w_scale: torch.Tensor, bias: torch.Tensor, stage: int) -> torch.Tensor:
    """Staged int8 Conv1d (kernel 3, padding 1) via window-matmul + per-channel fixed-point.

    x_int [B, in, T] int8 with a PER-BATCH scale x_scale [B, 1, 1] (the kernel window spans
    3 time positions, so a per-token scale is not factorable across the window sum -- see
    quantize_act_per_batch). w_int [out, in, 3] int8, w_scale [out].
    stage 1: fp conv on the int8-reconstructed weight + activation (weight+act quant).
    stage 2: int32 window matmul + per-channel W_scale + x_scale + int bias + requant.
    The per-batch x_scale is constant across the window, so it factors out of the window
    sum exactly (verified: stage 2 == stage 1 to fp rounding). Phase B emits the per-channel
    W_scale as a gemmlowp mul+shift (per-channel, no per-group cancellation -- trivially free
    vs A1) and the per-batch x_scale as the shared dynamic-quant fixed-point.
    """
    in_ch, T = x_int.shape[1], x_int.shape[2]
    out_ch = w_int.shape[0]
    if stage == 1:
        x_recon = (x_int.to(torch.float32) - x_zp.to(torch.float32)) * x_scale       # [B,in,T]
        w_recon = w_int.to(torch.float32) * w_scale[:, None, None]
        return fp_conv1d_ref(x_recon, w_recon, bias)
    # int32 window matmul: pad T by 1, unfold to [B, in, T, 3] -> [B, T, in*3]
    x32 = (x_int.to(torch.int32) - x_zp.to(torch.int32))                       # [B, in, T] centered
    xp = torch.nn.functional.pad(x32, (1, 1))                                 # [B, in, T+2]
    win = xp.unfold(-1, 3, 1).movedim(1, 2)                                   # [B, T, in, 3]
    win = win.reshape(x32.shape[0], T, in_ch * 3).to(torch.int32)             # [B, T, in*3]
    wr = w_int.reshape(out_ch, in_ch * 3).to(torch.int32)                     # [out, in*3]
    acc = torch.einsum("bti,oi->bto", win, wr)                                # [B, T, out] int32
    # x_scale [B,1,1] is constant across the window -> factors out: out = acc*w_scale*x_scale + bias.
    out = acc.to(torch.float32) * w_scale.to(torch.float32) * x_scale + bias.to(torch.float32)
    out = out.transpose(1, 2)                                                # [B, out, T] (conv convention)
    o_int, o_scale, o_zp = quantize_act_per_token(out)                       # per-token output to next op
    return dequant_act(o_int, o_scale, o_zp)


def int_attn_scale(scores_int: torch.Tensor, score_scale: torch.Tensor, d_head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer attention scale 1/sqrt(d_head). For d_head a perfect square of a power of two
    (tiny: 64 = 8^2), 1/sqrt(d_head) = 1/8 = an exact integer right-shift by 3 -- zero error
    on int32 matmul scores (the shift only discards bits below the matmul's own precision).
    Round-half-up is used so the shift rounds to nearest, not toward floor. Otherwise a
    fixed-point mul (Phase B). Returns (scaled_scores_int, scaled_score_scale)."""
    import math
    s = math.isqrt(d_head)
    if s * s == d_head:                                                       # exact integer sqrt
        if s & (s - 1) == 0:                                                   # s is a power of two -> clean shift
            shift = int(math.log2(s))
            return _rshift_round(scores_int, shift), score_scale               # round-half-up, no scale change
    # general case: fixed-point mul by 1/sqrt(d_head) (reference keeps fp here)
    factor = 1.0 / math.sqrt(d_head)
    return scores_int, score_scale * factor


# --------------------------------------------------------------------------- #
# A1 validation harness                                                        #
# --------------------------------------------------------------------------- #

def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().amax().clamp(min=1e-8)
    return (a - b).abs().amax().item() / denom.item()


def run_a1(model, layers: list[str], group_size: int = 32, seed: int = 0) -> None:
    """Validate the staged int8 matmul on real HQQ layers from the loaded model."""
    from hqq.core.quantize import HQQLinear
    torch.manual_seed(seed)
    print(f"{'layer':<46} {'nbits':>5} {'stage1':>10} {'stage2':>10} {'stage3':>10} {'stage4':>10}")
    print("-" * 96)
    for name, module in model.named_modules():
        if not isinstance(module, HQQLinear):
            continue
        if not any(name.endswith(ln) or ln in name for ln in layers):
            continue
        meta = dict(module.meta)
        meta["_W_q"] = module.W_q
        O, I = meta["shape"]
        g = meta["group_size"]
        nbits = int(meta["nbits"])
        w_fp = hqq_fp_weight(meta)                       # ground-truth fp weight
        bias = module.bias
        x = torch.randn(8, I, generator=torch.manual_seed(seed)) * 0.5
        # Stage 0: fp reference
        out0 = fp_matmul_ref(x, w_fp, bias)
        # Stage 1: int8 weight reconstructed (lossless) -> must equal Stage 0
        out1 = fp_matmul_ref(x, w_fp, bias)
        # Stage 2: + int8 activation
        x_int, x_scale, x_zp = quantize_act_per_token(x)
        x_recon = dequant_act(x_int, x_scale, x_zp)
        out2 = fp_matmul_ref(x_recon, w_fp, bias)
        # Stage 3: per-group int8 matmul, int32 accumulate, fp scale
        qr = unpack_levels(module.W_q, nbits, meta["packing"], meta["shape"], g, int(meta["axis"]))
        zero = meta["zero"]
        scale = meta["scale"]
        out3 = int8_matmul_per_group(x_int, x_scale, x_zp, qr, zero, scale, bias, g)
        # Stage 4: fixed-point scale
        out4 = int8_matmul_fixed_point(x_int, x_scale, x_zp, qr, zero, scale, bias, g)
        print(f"{name:<46} {nbits:>5} {_rel_err(out1,out0):>10.2e} {_rel_err(out2,out0):>10.2e} "
              f"{_rel_err(out3,out0):>10.2e} {_rel_err(out4,out0):>10.2e}")


def run_a2(model, layers: list[str], seed: int = 0, eps: float = 1e-5) -> None:
    """Validate the staged int8 LayerNorm on real Whisper LayerNorm modules."""
    import torch.nn as nn
    torch.manual_seed(seed)
    print(f"{'layer':<46} {'stage1':>10} {'stage2':>10} {'stage3':>10} {'stage4':>10}")
    print("-" * 96)
    for name, module in model.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        if not any(name.endswith(ln) or ln in name for ln in layers):
            continue
        D = module.normalized_shape[0]
        gamma = module.weight.to(torch.float32)
        beta = module.bias.to(torch.float32)
        # realistic-ish input: layernorm sees the post-residual stream; moderate scale
        x = torch.randn(8, D, generator=torch.manual_seed(seed)) * 1.5
        x_int, x_scale, x_zp = quantize_act_per_token(x)
        out0 = fp_layernorm_ref(x, gamma, beta, eps)
        errs = []
        for st in (1, 2, 3, 4):
            out = int8_layernorm(x_int, x_scale, x_zp, gamma, beta, eps, stage=st)
            errs.append(_rel_err(out, out0))
        print(f"{name:<46} {errs[0]:>10.2e} {errs[1]:>10.2e} {errs[2]:>10.2e} {errs[3]:>10.2e}")


def run_a3(seed: int = 0) -> None:
    """Validate the staged int8 GELU on a realistic fc1-output distribution.

    GELU is parameter-free (one function for all 8 modules), so one input distribution
    suffices; sweep a few input scales since the fc1-output magnitude varies by layer.
    """
    torch.manual_seed(seed)
    D = 1536   # fc1 output dim for whisper-tiny (4 * d_model)
    print(f"{'input_scale':<14} {'stage1':>10} {'stage2':>10} {'stage3':>10}")
    print("-" * 50)
    for scale in (0.5, 1.0, 2.0, 4.0):
        x = torch.randn(64, D, generator=torch.manual_seed(seed)) * scale
        x_int, x_scale, x_zp = quantize_act_per_token(x)
        out0 = fp_gelu_ref(x)
        errs = [_rel_err(int8_gelu(x_int, x_scale, x_zp, stage=s), out0) for s in (1, 2, 3)]
        print(f"{scale:<14} {errs[0]:>10.2e} {errs[1]:>10.2e} {errs[2]:>10.2e}")


def run_a4(seed: int = 0) -> None:
    """Validate the staged int8 softmax on realistic attention-score distributions.

    Softmax is parameter-free; sweep a few score magnitudes (the exp LUT and reciprocal
    behave differently when the score spread is large vs small).
    """
    torch.manual_seed(seed)
    K = 128                                       # key length (sequence dim)
    print(f"{'score_sigma':<14} {'stage1':>10} {'stage2':>10} {'stage3':>10} {'stage4':>10}")
    print("-" * 60)
    for sigma in (1.0, 2.0, 4.0, 8.0):
        scores = torch.randn(64, K, generator=torch.manual_seed(seed)) * sigma
        # per-row int32 score representation: scale = row_range/4096 (good LUT resolution)
        row_range = (scores.amax(-1, keepdim=True) - scores.amin(-1, keepdim=True)).clamp(min=1e-6)
        score_scale = row_range / 4096.0
        scores_int = (scores / score_scale).round().clamp(-2**31, 2**31 - 1).to(torch.int32)
        out0 = fp_softmax_ref(scores)
        errs = [_rel_err(int8_softmax(scores_int, score_scale, stage=s), out0) for s in (1, 2, 3, 4)]
        print(f"{sigma:<14} {errs[0]:>10.2e} {errs[1]:>10.2e} {errs[2]:>10.2e} {errs[3]:>10.2e}")


def run_a5(model, seed: int = 0) -> None:
    """Validate int8 conv (A5) on the two Whisper encoder convs + the attention scale."""
    import torch.nn as nn
    torch.manual_seed(seed)
    print("int8 Conv1d (encoder.conv1 80->384, encoder.conv2 384->384):")
    print(f"{'layer':<28} {'stage1':>10} {'stage2':>10}")
    print("-" * 52)
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv1d):
            continue
        W = mod.weight.to(torch.float32)
        bias = mod.bias.to(torch.float32)
        in_ch, T = mod.in_channels, 64
        x = torch.randn(4, in_ch, T, generator=torch.manual_seed(seed)) * 1.0
        # per-BATCH activation scale: the conv kernel window spans 3 time positions, so a
        # per-token (per-spatial) scale is not factorable across the window sum. One scale
        # per batch element is constant across the window -> factors out (stage 2 == stage 1).
        x_int, x_scale, x_zp = quantize_act_per_batch(x)            # all [B,1,1] (bcast in,T)
        w_int, w_scale = _quant_weight_per_channel(W)
        out0 = fp_conv1d_ref(x, W, bias)
        e1 = _rel_err(int8_conv1d(x_int, x_scale, x_zp, w_int, w_scale, bias, stage=1), out0)
        e2 = _rel_err(int8_conv1d(x_int, x_scale, x_zp, w_int, w_scale, bias, stage=2), out0)
        print(f"{name.replace('model.encoder.',''):<28} {e1:>10.2e} {e2:>10.2e}")
    # attention scale: 1/sqrt(64) = 1/8 = >>3 (exact for tiny). Use an int32-matmul-realistic
    # score quantization (fine scale -> large int range) so the shift's LSB truncation is
    # negligible, matching the real graph where scores come from an int8 q.k matmul.
    d_head = model.config.d_model // model.config.encoder_attention_heads
    scores = torch.randn(4, 6, 32, 32, generator=torch.manual_seed(seed)) * 3.0
    score_scale = torch.tensor(3.0 / 2**21)                       # fine: int range ~ +-2^21
    s_int = (scores / score_scale).round().clamp(-2**31, 2**31 - 1).to(torch.int32)
    scaled_int, scaled_scale = int_attn_scale(s_int, score_scale, d_head)
    fp_scaled = scores * (1.0 / (d_head ** 0.5))
    int_scaled_real = scaled_int.float() * scaled_scale
    print(f"\nint attention scale 1/sqrt({d_head})={1.0/(d_head**0.5)}: "
          f"rel_err vs fp = {_rel_err(int_scaled_real, fp_scaled):.2e} (exact shift)")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="build/whisper-tiny-hqq-2bit")
    p.add_argument("--a1", action="store_true", help="run the A1 staged matmul validation")
    p.add_argument("--a2", action="store_true", help="run the A2 staged layernorm validation")
    p.add_argument("--a3", action="store_true", help="run the A3 staged GELU validation")
    p.add_argument("--a4", action="store_true", help="run the A4 staged softmax validation")
    p.add_argument("--a5", action="store_true", help="run the A5 int8 conv + attn-scale validation")
    p.add_argument("--layers", nargs="*", default=[
        "decoder.layers.0.self_attn.q_proj",   # 2-bit
        "decoder.layers.0.fc2",                 # 2-bit
        "encoder.layers.0.self_attn.k_proj",   # 8-bit
        "decoder.layers.0.fc1",                 # 8-bit
    ])
    args = p.parse_args()
    os.environ.setdefault("HQQ_COMPUTE_DTYPE", "fp32")
    import hqq_asr
    model = hqq_asr.load_whisper_hqq(args.model, compute_dtype=torch.float32)
    if args.a1:
        run_a1(model, args.layers)
    elif args.a2:
        run_a2(model, [
            "encoder.layers.0.self_attn_layer_norm",
            "encoder.layers.0.final_layer_norm",
            "decoder.layers.0.self_attn_layer_norm",
            "decoder.layers.0.encoder_attn_layer_norm",
            "decoder.layers.0.final_layer_norm",
            "encoder.layer_norm",
            "decoder.layer_norm",
        ])
    elif args.a3:
        run_a3()
    elif args.a4:
        run_a4()
    elif args.a5:
        run_a5(model)
    else:
        p.print_help()


if __name__ == "__main__":
    main()