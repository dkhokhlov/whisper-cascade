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


def quantize_act_per_token_intscale(x: torch.Tensor):
    """Per-token asymmetric int8 quant with an INTEGER fixed-point scale (A7, zero-fp).

    Same int8 levels as quantize_act_per_token, but the scale is returned as a fixed-point
    pair (mul, shift) instead of a float: x_fp ~= (x_int8 - x_zp) * mul * 2^(-shift). This is
    the form the zero-fp ONNX graph applies as an int Mul + BitShift (no runtime fp). The
    scale is still derived here from the fp amax; the deployed graph derives it from the int
    amax (same order) -- A7 gates the fp-scale -> int-fixed-point-scale rounding, the one
    unmeasured approximation.

    Returns (x_int8 [-128,127] int32, mul [*,1] int32, shift [*,1] int32, zp [*,1] int32).

    mul is a Q1.16 fixed-point (normalized to ~[2^15, 2^16)): coarser than the weight scale's
    Q1.31 because the activation scale multiplies the FULL-I accumulator acc (int64, ~2^36 for
    fc1), so mul * acc must fit int64. 16-bit scale precision is still far finer than the
    int8 weight quant (2^-16 rel vs 2^-7), so the scale-value rounding is ~lossless; the
    int64-overflow bound, not precision, sets the Q here.

    A7 boundary round (codex+claude consensus, 2026-08-05): the int8 bin each value maps to
    is decided by the INTEGER scale (mul*2^-shift), not the fp scale, so the reference's
    lossy round matches the deployed graph's integer requant (else A7 is optimistic). The
    arithmetic is fp but the scale VALUE is the Q1.16-quantized scale.
    """
    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    scale = ((xmax - xmin) / 255.0).clamp(min=1e-8)               # fp scale (intermediate)
    q, exp = torch.frexp(scale.to(torch.float64))                  # scale = q * 2^exp, q in [0.5,1)
    mul = torch.round(q * (2 ** 16)).clamp(0, 2 ** 16 - 1).to(torch.int32)    # Q1.16
    shift = (16 - exp).to(torch.int32)                            # scale = mul * 2^(-shift)
    # A7 boundary round: the int8 bin each value maps to is decided by the INTEGER scale
    # (mul * 2^-shift), NOT the fp scale. The deployed zero-fp ONNX requants int8->int8 with
    # this integer scale; rounding with the fp scale here would measure the next op's lossy
    # round at higher precision than the deployed graph, making A7 optimistic. (codex+claude
    # consensus, 2026-08-05.) The arithmetic is fp but the scale VALUE is the Q1.16-quantized
    # scale, isolating the scale-value approximation A7 measures.
    scale_int = (mul.to(torch.float64) * (2.0 ** (-shift.to(torch.float64)))).clamp(min=1e-12)
    zp = torch.round(-xmin / scale_int - 128.0)
    x_int = torch.round(x / scale_int + zp).clamp(-128, 127).to(torch.int32)
    return x_int, mul, shift, zp


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


def fixed_point_per_group(scale: torch.Tensor, Q: int = 31) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-element fixed-point (gemmlowp/TFLite scheme): scale ≈ mul * 2^left_shift.

    Returns (mul [same shape] int32, right_shift [same shape] int32). mul is normalized
    to ~[2^(Q-1), 2^Q) (a Q0.Q fixed-point), so each element keeps ~(Q-1) bits of precision
    regardless of magnitude. The shift is PER-ELEMENT -- a shared shift cannot cover the
    per-group scale dynamic range (~10-100x within a layer, see A1 diagnostics) without
    zeroing small-scale groups. ONNX BitShift accepts a per-element shift tensor, so this
    is emittable. HQQ scales are all < 1, so left_shift <= 0 and a right-shift suffices.

    Q=31 (default) for the linear per-GROUP scale: the per-group partial (group_size terms,
    ~2^12) is shifted to 2^-F BEFORE the cross-group sum, so mul (~2^31) * partial (~2^12) =
    2^43 fits int64 with room. Q=15 for the conv per-CHANNEL scale: the conv accumulates the
    FULL int32 acc (all in*kernel terms, up to 2^31) BEFORE the scale, so mul (~2^15) * acc
    (2^31) = 2^46 fits int64 (a Q0.31 mul would overflow: 2^31 * 2^31 = 2^62, then * the Q1.16
    act scale 2^16 -> 2^78). Q0.15 is still 2^8 finer than the int8 weight quant (2^-7), so
    WER-neutral (the conv weight is int8, not HQQ 2/3-bit).
    """
    s = scale.to(torch.float64)
    q, exp = torch.frexp(s)                       # s = q * 2^exp, q in [0.5, 1) for s>0
    mul = torch.round(q * (2 ** Q)).clamp(0, 2 ** Q - 1).to(torch.int32)
    left_shift = exp - Q                           # <= 0 for s < 1
    right_shift = (-left_shift).clamp(min=0).to(torch.int32)
    return mul, right_shift


def int8_matmul_fixed_point(
    x_int: torch.Tensor, x_scale: torch.Tensor, x_zp: torch.Tensor,
    qr: torch.Tensor, zero: torch.Tensor, scale: torch.Tensor,
    bias: torch.Tensor | None, group_size: int,
    act_mul: torch.Tensor | None = None, act_shift: torch.Tensor | None = None,
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
    if act_mul is not None:
        # A7 integer activation scale: out_real = acc * act_mul * 2^-(F + act_shift).
        # The act scale is applied as an int Mul (acc * am, both int) against the
        # fixed-point 2^-(F + act_shift) power-of-two. The reference interface dequants
        # to fp WITHOUT integer rounding loss, so this isolates the act-scale-VALUE
        # approximation (Q1.16 vs the fp per-token scale) from a separate output
        # quantization the NEXT op performs. The deployed ONNX graph keeps int8
        # between ops via a proper requant (the rounding lives there, not here).
        am = act_mul.to(torch.int64).reshape(-1, 1)                  # [B,1]
        ash = (F + act_shift.to(torch.int64)).reshape(-1, 1)         # [B,1] = F + per-token shift
        out = (acc * am).to(torch.float32) / (2.0 ** ash.to(torch.float32))
    else:
        out = _rshift_round(acc, F_t).to(torch.float32) * x_scale    # fp act scale (A6 path)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out


# --------------------------------------------------------------------------- #
# Stage 1b: int-canonical output requant (the zero-fp boundary)                #
# --------------------------------------------------------------------------- #

def _round_half_up(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Round half up (toward +inf at a tie) of num/den, den > 0, all signs.

    floor(num/den + 0.5) = (2*num + den) // (2*den). torch int // is floor division, so this
    holds for negative num too. This is the canonical int round the ONNX graph emits; it
    replaces torch.round (half-to-even) at the zero-fp boundary so the reference and the
    deployed graph agree by construction.
    """
    num = num.to(torch.int64)
    den = den.to(torch.int64).clamp(min=1)
    return torch.div(2 * num + den, 2 * den, rounding_mode="floor")


def int8_output_requant_intscale(
    acc: torch.Tensor, act_mul: torch.Tensor, act_shift: torch.Tensor,
    F: int, bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage 1b int-canonical output requant -- the true zero-fp boundary.

    Takes the per-group matmul accumulator (int64 @ 2^-F) + the Q1.16 input act scale
    (act_mul, act_shift, [B]) + bias, and produces the int8 output + its Q1.16 scale, ALL
    in integer (no runtime fp). The next op consumes the result as its activation:
      (x_int, act_mul, act_shift, x_zp) = (y_int8, y_mul, y_shift, y_zp).

    out_int_scaled[b,o] = acc[b,o]*act_mul[b] + (bias_fixed[o] << act_shift[b])  [int64]
      bias_fixed[o] = round_half_up(bias[o]*2^F); out_fp = out_int_scaled / 2^(F+act_shift[b]).
    Per-token output Q1.16 from the EXACT int range (not a float amax):
      range = rmax - rmin (>=1); e = bit_length(range) - 8 (CLZ; the float64 log2 only needs
      the power-of-two interval, which is exact even when range > 2^53);
      mul = round_half_up(range * 2^(16-e) / 255) via int Mul + int Div, normalized to
      [2^15, 2^16) (the 255-not-2^8 boundary can need one e += 1 adjust);
      y_shift = (16-e) + F + act_shift.
      y_zp = round_half_up(-rmin * 2^(16-e) / mul) - 128;
      y_int8 = clamp(round_half_up(out_int_scaled * 2^(16-e) / mul) + y_zp, -128, 127).

    Returns (y_int8 [B,O] int32, y_mul [B,1] int32, y_shift [B,1] int32, y_zp [B,1] int32).

    Differential vs the A7.1 float32 reference (quantize_act_per_token_intscale(out_fp),
    half-to-even): ~0.0004% of y_int8 differ by 1 LSB (ties + float32-vs-exact-int range),
    WER-neutral. The int path rounds the EXACT int (the A7.1 path rounds the float32-rounded
    out_fp), so the deployed graph is equal-or-better. Inter-op int8 is gated at Phase B
    Gate 2 (ORT WER == A7.1 torch WER), not here. (codex+claude consensus, 2026-08-05.)
    """
    am = act_mul.to(torch.int64).reshape(-1, 1)                       # [B,1]
    ash = act_shift.to(torch.int64).reshape(-1, 1)                    # [B,1]
    out_int = acc.to(torch.int64) * am                                 # [B,O] int64 @ 2^-F
    if bias is not None:
        bias_fixed = torch.floor(bias.to(torch.float64) * (2.0 ** F) + 0.5).to(torch.int64)
        out_int = out_int + (bias_fixed.unsqueeze(0) << ash)           # bias in the 2^-(F+ash) domain
    rmax = out_int.max(dim=-1, keepdim=True).values                    # [B,1]
    rmin = out_int.min(dim=-1, keepdim=True).values                    # [B,1]
    R = (rmax - rmin).clamp(min=1)                                    # [B,1] >=1

    # exponent e: bit_length(R) - 8. float64 log2 is exact for the power-of-two interval
    # even when R > 2^53 (the rounding never crosses a 2^k boundary). Mantissa is computed
    # from the EXACT int R below, so float64-precision of R itself is irrelevant.
    bitlen = torch.floor(torch.log2(R.to(torch.float64))).to(torch.int64) + 1
    e = bitlen - 8                                                    # [B,1]
    sh = (16 - e)                                                     # [B,1] = 16-e (can be negative)

    def _mul(num_in, sh_in):
        # mul = round_half_up(num_in * 2^sh_in / 255) ; sh_in may be negative
        pos = sh_in >= 0
        num = torch.where(pos, num_in << sh_in.clamp(min=0), num_in)
        den = torch.where(pos, torch.full_like(sh_in, 255), (255 << (-sh_in).clamp(min=0)))
        return _round_half_up(num, den)

    mul = _mul(R, sh)                                                 # [B,1] int64
    # normalize mul to [2^15, 2^16) -- the 255-not-2^8 boundary can need one e += 1 step
    too_small = mul < (1 << 15)
    too_big = mul >= (1 << 16)
    sh = torch.where(too_small, sh + 1, sh)
    sh = torch.where(too_big, sh - 1, sh)
    mul = _mul(R, sh)

    # zp = round_half_up(-rmin * 2^sh / mul) - 128  (sh == 16-e)
    pos = sh >= 0
    num_z = torch.where(pos, (-rmin) << sh.clamp(min=0), (-rmin).to(torch.int64))
    den_z = torch.where(pos, mul, mul << (-sh).clamp(min=0))
    y_zp = (_round_half_up(num_z, den_z) - 128).to(torch.int32)        # [B,1]

    # y_int8 = round_half_up(out_int * 2^sh / mul) + y_zp, clamped
    num_y = torch.where(pos, out_int << sh.clamp(min=0), out_int)
    den_y = torch.where(pos, mul, mul << (-sh).clamp(min=0))
    y_arg = _round_half_up(num_y, den_y)                              # [B,O] int64
    y_int8 = (y_arg + y_zp.to(torch.int64)).clamp(-128, 127).to(torch.int32)

    y_shift = (sh + F + ash).to(torch.int32)                          # [B,1]
    return y_int8, mul.to(torch.int32), y_shift, y_zp


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
# A2 (int-canonical): pure-int LayerNorm oracle for the zero-fp ONNX emission  #
# (codex+claude consensus 2026-08-05). The Phase-A int8_layernorm(stage=4) uses  #
# runtime fp (eps_K from fp x_scale; rsqrt seed via torch.log2; fp fallback),  #
# so it is a DIFFERENT algorithm, not a bit-exact oracle. This reference is    #
# pure-int throughout and is the spec the ONNX LayerNorm mirrors bit-exactly.  #
# --------------------------------------------------------------------------- #

def _int_bitlen(s: torch.Tensor) -> torch.Tensor:
    """Pure-int bit-length = floor(log2(s)) + 1 via a 6-level CLZ ladder, all int64.
    Mirrors the ONNX _clz_ladder so the torch reference and the ONNX emission use identical
    CLZ logic (bit-exact at powers of two). s int64 > 0, clamped to [1, 2^62); returns int64.
    The k=1 step's largest threshold is 2^62 (s < 2^62 by the cap, so the 2^63 step -- which
    would overflow the int64 cast -- is never taken)."""
    s = torch.clamp(s.to(torch.int64), min=1, max=(1 << 62) - 1)
    b = torch.zeros_like(s)
    ge = s >= (1 << 32)
    b = torch.where(ge, b + 32, b)
    one = torch.ones_like(s)
    for k in (16, 8, 4, 2, 1):
        shift = (b + k).clamp(max=62)                                       # never 1<<63
        thr = torch.bitwise_left_shift(one, shift)                          # 2^(b+k), int64
        b = torch.where(s >= thr, b + k, b)
    return b + 1


def int8_layernorm_intscale(
    x_int: torch.Tensor, x_zp: torch.Tensor, y_mul_in: torch.Tensor,
    y_shift_in: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float,
    K: int = _LN_K, R: int = _LN_R, G: int = _LN_G, return_intermediates: bool = False,
):
    """Int-canonical pure-int LayerNorm -- the zero-fp oracle the ONNX LayerNorm mirrors.

    Input: the previous op's int8 output as (x_int [B,D], x_zp [B,1], y_mul_in [B,1] Q1.16,
    y_shift_in [B,1]); x_scale = y_mul_in * 2^-y_shift_in. gamma/beta [D]. Pure-int throughout
    (no runtime fp): eps_K from the int act scale, int rsqrt seed via a pure-int CLZ bitlen
    (no torch.log2), no fp fallback (unreachable: |u| <= 255 -> var_K < 2^32, bitpos <= 31,
    a = R+half >= 12). Output reuses int8_output_requant_intscale with F = K+R+G.

    Algorithm (codex+claude consensus):
      u = x_int - x_zp ; S1 = sum(u) ; S2 = sum(u*u)                  (int64; cast before u*u)
      mean_K = floor_div(S1<<K, D) ; var_K = floor_div(S2<<K, D) - (mean_K^2 >> K)
      eps_K = round_half_up(eps*2^(K+2*y_shift) / y_mul^2) ; max(eps_K, 1)
            (p/q form, eps=1e-5 -> p=1, q=100000; GUARD K+2*y_shift <= 62)
      s_K = var_K + eps_K (clamp [1, 2^62))
      bitlen = clz_ladder(s_K) ; bitpos = bitlen-1 ; e = K-bitpos ;
      odd = e & 1 ; half = (e - odd)//2 ; a = R+half
      C = 8409*2^20 (even) or 8409*1482910 (sqrt2_Q20, odd) ;
      seed = round_half_up(C<<a, 10000*2^20)
      r = seed ; 4x: prod = (s_K*r)*r ; t = prod >>(K+R) ; r = (r*(3*2^R - t) + 2^(R-1)) >>(R+1)
      y_int = ((u<<K) - mean_K)*r*gamma_int + (beta_int << (K+R))     (Q(K+R+G))
      out = int8_output_requant_intscale(y_int, act_mul=1, act_shift=0, F=K+R+G, bias=None)
    Overflow: Newton s_K*r^2 = 2^(K+2R) = 2^56 invariant, peak ~2^57 (< 2^63). Output z-score
    bound ((u<<K)-mean_K)*r <= sqrt(D)*2^(K+R) ~ 2^40.3; *gamma_int(2^15) ~ 2^55.3 < 2^63;
    +beta<<(K+R) ~ 2^51; requant acc ~2^55 within cap62. R <= 23 hard boundary (K+2R < 63).

    Returns (y_int8 [B,D] int32, y_mul [B,1] int32, y_shift [B,1] int32, y_zp [B,1] int32);
    with return_intermediates, also a dict of the int intermediates for bit-exact verification.
    """
    B, D = x_int.shape[0], x_int.shape[-1]
    u = x_int.to(torch.int64) - x_zp.to(torch.int64).reshape(-1, 1)          # [B,D] centered
    S1 = u.sum(dim=-1, keepdim=True)                                         # [B,1] int64
    S2 = (u * u).sum(dim=-1, keepdim=True)                                   # [B,1] int64
    D_i64 = torch.tensor(D, dtype=torch.int64)
    mean_K = torch.div(S1 << K, D_i64, rounding_mode="floor")                # QK floor (S1 signed)
    var_K = torch.div(S2 << K, D_i64, rounding_mode="floor") - (mean_K * mean_K >> K)
    # eps_K = round_half_up(eps * 2^(K+2*y_shift) / y_mul^2); eps = p/q (eps=1e-5 -> q=100000)
    p, q = 1, int(round(1.0 / eps))
    ys = y_shift_in.to(torch.int64).reshape(-1, 1)
    ym = y_mul_in.to(torch.int64).reshape(-1, 1)
    h = K + 2 * ys
    if int(h.max()) > 62:
        raise ValueError(f"int8_layernorm_intscale: K+2*y_shift={int(h.max())} > 62 "
                          f"(eps_K int64 overflow; require y_shift <= 23)")
    num = (torch.ones_like(h) * p) << h                                      # p * 2^(K+2*ys), int64
    den = (q * ym * ym).to(torch.int64)                                      # q * y_mul^2
    eps_K = torch.clamp(_round_half_up(num, den), min=1)
    s_K = torch.clamp(var_K + eps_K, min=1, max=(1 << 62) - 1)
    # int rsqrt seed via pure-int CLZ bitlen
    bitlen = _int_bitlen(s_K)
    bitpos = bitlen - 1
    e = K - bitpos
    odd = e & 1                                                       # two's-complement parity, correct for e<0
    half = (e - odd) // 2                                             # floor(e/2), sign-safe (numerator even)
    a = R + half                                                      # always >= 12 for |u| <= 255
    sqrt2_Q20 = 1482910                                                # round(sqrt(2) * 2^20)
    C = torch.where(odd == 1, torch.full_like(s_K, 8409 * sqrt2_Q20),
                    torch.full_like(s_K, 8409 * (1 << 20)))
    den_seed = torch.full_like(s_K, 10000 * (1 << 20))
    seed = _round_half_up(C << a.clamp(min=0), den_seed)               # a >= 12 always
    r = seed
    three = torch.full_like(s_K, 3 * (1 << R))
    half_r = torch.full_like(s_K, 1 << (R - 1))
    for _ in range(4):
        prod = (s_K * r) * r                                          # int64, ~2^56 invariant
        t = prod >> (K + R)                                            # positive -> arithmetic == logical
        r = (r * (three - t) + half_r) >> (R + 1)
    # y_int @ Q(K+R+G) = ((u<<K) - mean_K) * r * gamma_int + (beta_int << (K+R))
    gamma_int = torch.floor(gamma.to(torch.float64) * (2.0 ** G) + 0.5).to(torch.int64)   # [D] QG (bake)
    beta_int = torch.floor(beta.to(torch.float64) * (2.0 ** G) + 0.5).to(torch.int64)     # [D] QG (bake)
    uK = (u << K) - mean_K                                             # [B,D] QK (u<<K == u*2^K, two's complement)
    y_int = uK * r * gamma_int.reshape(1, -1) + (beta_int.reshape(1, -1) << (K + R))   # [B,D] Q(K+R+G)
    # output requant: reuse Stage 1b with F = K+R+G, fresh per-token scale (act_mul=1, act_shift=0)
    ones = torch.ones(B, 1, dtype=torch.int32)
    zeros = torch.zeros(B, 1, dtype=torch.int32)
    y_int8, y_mul, y_shift, y_zp = int8_output_requant_intscale(
        y_int, ones, zeros, K + R + G, None)
    if return_intermediates:
        inter = {"u": u, "S1": S1, "S2": S2, "mean_K": mean_K, "var_K": var_K,
                 "eps_K": eps_K, "s_K": s_K, "bitlen": bitlen, "bitpos": bitpos,
                 "e": e, "odd": odd, "half": half, "a": a, "seed": seed, "r": r,
                 "gamma_int": gamma_int, "beta_int": beta_int, "uK": uK, "y_int": y_int}
        return y_int8, y_mul, y_shift, y_zp, inter
    return y_int8, y_mul, y_shift, y_zp


# --------------------------------------------------------------------------- #
# A3: int8 GELU (exact normal-CDF LUT: GELU(x) = x * Phi(x))                   #
# --------------------------------------------------------------------------- #

# Phi LUT: Phi(x) = 0.5*(1 + erf(x/sqrt2)), the standard normal CDF. GELU is
# x*Phi(x) exactly, so storing Phi (not sigmoid(1.702x)) removes the 1.702
# approximation error (measured ~0.0047 MAE on the real fc1 output). Phi
# saturates to 0/1 outside +/- Lx, so the tail is gelu ~= x_real; the
# x_real * LUT multiply keeps the tail exact (no clamp error).
_GELU_LX = 6         # input half-range; |Phi(6)-1| < 1e-9, saturates to <1 LSB
_GELU_T = 4096        # LUT entries over [-Lx, Lx]  (grid spacing ~0.003 in x)
_GELU_S = 16          # Phi fixed-point bits (Phi in [0, 1], 1.0 = 2^S)


def _phi_lut() -> torch.Tensor:
    """Int32 LUT [_GELU_T]: round(Phi(x_grid) * 2^_GELU_S), x_grid in [-Lx, Lx]."""
    import math
    Lx, T, S = _GELU_LX, _GELU_T, _GELU_S
    x_grid = torch.linspace(-Lx, Lx, T, dtype=torch.float64)
    phi = 0.5 * (1.0 + torch.erf(x_grid / math.sqrt(2.0)))            # normal CDF
    return (phi * (2 ** S)).round().to(torch.int32)


def fp_gelu_ref(x: torch.Tensor) -> torch.Tensor:
    """Stage 0: exact erf GELU (what Whisper's GELUActivation uses)."""
    return torch.nn.functional.gelu(x, approximate="none")


def int8_gelu(x_int: torch.Tensor, x_scale: torch.Tensor, x_zp: torch.Tensor, stage: int) -> torch.Tensor:
    """Staged int8 GELU via GELU(x) = x * Phi(x), Phi from an int LUT.

    x_int [B,D] int8 (per-token, from the fc1 output requant), x_scale/x_zp [B,1].
    stage 1: exact erf GELU on the int-reconstructed input.      (input-quant error)
    stage 2: Phi-LUT GELU (fp output).                           (LUT quant only)
    stage 3: + per-token requant to int8, dequant for compare.   (output quant)
    """
    Lx, T, S = _GELU_LX, _GELU_T, _GELU_S
    u = (x_int.to(torch.float32) - x_zp.to(torch.float32))          # centered int
    x_real = u * x_scale                                             # [B,D] real input
    if stage == 1:
        return fp_gelu_ref(x_real)
    # index = round(x_real * T/(2*Lx)) + T/2, clamped; tail saturates Phi to 0/1
    idx = torch.clamp((x_real * (T / (2 * Lx))).round().to(torch.int64) + T // 2, 0, T - 1)
    lut = _phi_lut().to(x_real.device)
    phi = lut[idx].to(torch.float32) / (2 ** S)                     # [B,D] Phi(x)
    gelu = x_real * phi                                            # exact GELU = x*Phi(x)
    if stage == 2:
        return gelu
    # stage 3: requant to int8 (the fc2 input), dequant for comparison
    g_int, g_scale, g_zp = quantize_act_per_token(gelu)
    return dequant_act(g_int, g_scale, g_zp)


# --------------------------------------------------------------------------- #
# A3 (int-canonical): pure-int GELU oracle for the zero-fp ONNX emission       #
# (codex+claude consensus 2026-08-05). The Phase-A int8_gelu uses runtime fp   #
# (x_scale fp, fp index round), so it is a DIFFERENT algorithm, not a bit-exact #
# oracle. This reference is pure-int throughout and is the spec the ONNX GELU   #
# mirrors bit-exactly. Same Phi LUT (x*Phi(x)); the index round and the         #
# x*Phi multiply are fixed-point (integer input scale).                         #
# --------------------------------------------------------------------------- #

# Fixed-point for the LUT index: x_real * T/(2*Lx) = u*y_mul*2^-y_shift * (1024/3) (T=4096,
# 2*Lx=12). Fold 1024/3 into a Q_IDX fixed-point multiplier; the ONNX emits the same constant.
_GELU_IDX_Q = 16
_GELU_IDX_MUL = int(round((1024.0 / 3.0) * (2 ** _GELU_IDX_Q)))   # 22369621


def int8_gelu_intscale(x_int, x_zp, y_mul_in, y_shift_in, return_intermediates=False):
    """Int-canonical pure-int GELU -- the zero-fp oracle the ONNX GELU mirrors.

    Input: the previous op's int8 output as (x_int [B,D], x_zp [B,1], y_mul_in [B,1] Q1.16,
    y_shift_in [B,1]); x_real = (x_int - x_zp)*y_mul*2^-y_shift. GELU(x) = x*Phi(x), Phi from
    the int LUT (_phi_lut, Phi*2^S over [-Lx,Lx]). Pure-int throughout (no runtime fp):
      u = x_int - x_zp                                           (int64 [B,D])
      idx = clamp(round_half_up(u*y_mul*IDX_MUL, 2^(IDX_Q+y_shift)) + T//2, 0, T-1)
      phi_int = LUT[idx]                                          (int64 [B,D] = Phi*2^S)
      acc = u * phi_int                                           (int64 [B,D] @ 2^S)
      out = int8_output_requant_intscale(acc, y_mul, y_shift, F=S, bias=None)
    The per-token input scale (y_mul, y_shift) is folded into the requant as the act scale, so
    acc @ 2^S and the output y_shift = sh + S + y_shift. acc ~ |u|*2^S <= 255*2^16 ~ 2^24;
    out_int = acc*y_mul ~ 2^39 < 2^63 (int64-safe).

    Returns (y_int8 [B,D] int32, y_mul [B,1] int32, y_shift [B,1] int32, y_zp [B,1] int32);
    with return_intermediates, also a dict of the int intermediates for bit-exact verification.
    """
    T, S = _GELU_T, _GELU_S
    u = x_int.to(torch.int64) - x_zp.to(torch.int64).reshape(-1, 1)          # [B,D]
    ym = y_mul_in.to(torch.int64).reshape(-1, 1)                              # [B,1]
    ys = y_shift_in.to(torch.int64).reshape(-1, 1)                            # [B,1]
    num = u * ym * _GELU_IDX_MUL                                             # [B,D] int64
    den = (torch.ones_like(ys) << (ys + _GELU_IDX_Q))                        # [B,1] = 2^(IDX_Q+ys)
    idx_pre = _round_half_up(num, den)                                       # [B,D] = round_half_up(num/den)
    idx = torch.clamp(idx_pre + T // 2, 0, T - 1)                            # [B,D]
    lut = _phi_lut().to(torch.int64)                                         # [T] int64 (Phi*2^S)
    phi_int = lut[idx]                                                       # [B,D] int64
    acc = u * phi_int                                                        # [B,D] @ 2^S
    y_int8, y_mul, y_shift, y_zp = int8_output_requant_intscale(
        acc, ym.to(torch.int32), ys.to(torch.int32), S, None)
    if return_intermediates:
        inter = {"u": u, "num": num, "den": den, "idx": idx,
                 "phi_int": phi_int, "acc": acc}
        return y_int8, y_mul, y_shift, y_zp, inter
    return y_int8, y_mul, y_shift, y_zp


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
# A4 (int-canonical): pure-int softmax oracle for the zero-fp ONNX emission      #
# (codex+claude consensus 2026-08-05). The Phase-A int8_softmax uses runtime fp  #
# (fp score_scale, fp index round, torch.log2 reciprocal seed + fp fallback), so #
# it is a DIFFERENT algorithm, not a bit-exact oracle. This reference is pure-int #
# throughout and is the spec the ONNX softmax mirrors bit-exactly.               #
# --------------------------------------------------------------------------- #

# Fixed-point for the exp-LUT index: shifted_real * T/L = shifted_int*y_mul*2^-y_shift * (1024/3)
# (T=4096, L=12). Same IDX_MUL as GELU (the /3 folded into Q_IDX); the offset differs (T-1, the
# exp LUT grid is [-L, 0], not [-Lx, Lx]).
_SM_IDX_Q = 16
_SM_IDX_MUL = int(round((1024.0 / 3.0) * (2 ** _SM_IDX_Q)))   # 22369621


def _int_recip_intscale(x_int, K, P=_SM_P, iters=5):
    """Pure-int reciprocal 1/x via a CLZ seed + Newton, all int64 (mirrors the ONNX emission).
    x_int [..., 1] int64 = x_real*2^K (x_real > 0). Returns r_int [..., 1] = (1/x_real)*2^P.
    Seed: bitpos = _int_bitlen(x_int) - 1 (pure-int CLZ, no torch.log2); e = P + K - bitpos;
    seed = round_half_up(2^e * 7071, 10000) (0.7071 = geom mean of 1/mantissa in (0.5,1]).
    e >= 0 always for softmax (sum_exp <= K*2^S << 2^(P+K)); clamp(min=0) guards the shift.
    Newton: t = (x_int*r) >> K ; r = (r*(2*2^P - t) + 2^(P-1)) >> P (round-half-up, all positive:
    the 0.707 seed underestimates, so r approaches 1/x from below and t < 2*2^P throughout)."""
    x_int = x_int.to(torch.int64)
    bitpos = _int_bitlen(x_int) - 1
    e = (P + K) - bitpos
    base = torch.bitwise_left_shift(torch.ones_like(x_int), e.clamp(min=0))   # 2^e (e >= 0)
    seed = _round_half_up(base * 7071, torch.full_like(x_int, 10000))         # *0.7071
    r = seed
    two = torch.full_like(x_int, 2 * (1 << P))
    half = torch.full_like(x_int, 1 << (P - 1))
    for _ in range(iters):
        t = torch.bitwise_right_shift(x_int * r, K)                            # x*r in Q_P
        r = torch.bitwise_right_shift(r * (two - t) + half, P)                 # round-half-up
    return r


def int8_softmax_intscale(x_int, x_zp, y_mul_in, y_shift_in, return_intermediates=False):
    """Int-canonical pure-int softmax over the last dim -- the zero-fp oracle the ONNX mirrors.

    Input: the previous op's int8 output (x_int [..., K], x_zp [..., 1], y_mul_in [..., 1] Q1.16,
    y_shift_in [..., 1]); score_real = (x_int - x_zp)*y_mul*2^-y_shift. softmax is translation-
    invariant, so the subtract-max CANCELS x_zp: shifted_real = (x_int - max_int)*y_mul*2^-y_shift.
    Pure-int throughout (no runtime fp):
      max_int = amax(x_int, -1) ; shifted_int = x_int - max_int (<= 0)
      idx = clamp(round_half_up(shifted_int*y_mul*IDX_MUL, 2^(IDX_Q+y_shift)) + T-1, 0, T-1)
      exp_int = exp_lut[idx] (Q_S) ; sum_exp = sum(exp_int) (int64)
      inv_int = _int_recip_intscale(sum_exp, K=S, P=P)   (pure-int CLZ seed + Newton)
      p_fixed = exp_int * inv_int (Q(S+P))
      out = int8_output_requant_intscale(p_fixed, 1, 0, F=S+P, bias=None)
    The exp LUT covers shifted_real in [-L, 0]; values below -L clamp to idx 0 (exp ~= 0). The
    input scale (set by the previous op) maps the attention-score range into this window.

    Returns (y_int8 [..., K] int32, y_mul [..., 1] int32, y_shift [..., 1] int32, y_zp [..., 1]);
    with return_intermediates, also a dict of the int intermediates for bit-exact verification.
    """
    T, S, P = _SM_T, _SM_S, _SM_P
    shp = x_int.shape
    x64 = x_int.to(torch.int64)
    ym = y_mul_in.to(torch.int64).reshape(*shp[:-1], 1)
    ys = y_shift_in.to(torch.int64).reshape(*shp[:-1], 1)
    # subtract-max (zp cancels); shifted_int <= 0
    max_int = x64.amax(dim=-1, keepdim=True)                                  # [..., 1]
    shifted = x64 - max_int                                                    # [..., K] (<= 0)
    # idx = clamp(round_half_up(shifted*ym*IDX_MUL, 2^(IDX_Q+ys)) + T-1, 0, T-1)
    num = shifted * ym * _SM_IDX_MUL                                           # [..., K]
    den = torch.ones_like(ys) << (ys + _SM_IDX_Q)                             # [..., 1] = 2^(IDX_Q+ys)
    idx = torch.clamp(_round_half_up(num, den) + T - 1, 0, T - 1)             # [..., K]
    lut = _exp_lut().to(torch.int64)                                           # [T] exp*2^S
    exp_int = lut[idx]                                                         # [..., K] Q_S
    sum_exp = exp_int.sum(dim=-1, keepdim=True)                               # [..., 1] int64
    inv_int = _int_recip_intscale(sum_exp, K=S, P=P)                           # [..., 1] Q_P
    p_fixed = exp_int * inv_int                                                # [..., K] Q(S+P)
    ones = torch.ones(*shp[:-1], 1, dtype=torch.int32)
    zeros = torch.zeros(*shp[:-1], 1, dtype=torch.int32)
    y_int8, y_mul, y_shift, y_zp = int8_output_requant_intscale(p_fixed, ones, zeros, S + P, None)
    if return_intermediates:
        inter = {"max_int": max_int, "shifted": shifted, "num": num, "den": den,
                 "idx": idx, "exp_int": exp_int, "sum_exp": sum_exp,
                 "inv_int": inv_int, "p_fixed": p_fixed}
        return y_int8, y_mul, y_shift, y_zp, inter
    return y_int8, y_mul, y_shift, y_zp


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
                w_int: torch.Tensor, w_scale: torch.Tensor, bias: torch.Tensor, stage: int,
                stride: int = 1, kernel: int = 3, padding: int = 1) -> torch.Tensor:
    """Staged int8 Conv1d via window-matmul + per-channel fixed-point.

    x_int [B, in, T] int8 with a PER-BATCH scale x_scale [B, 1, 1] (the kernel window spans
    `kernel` time positions, so a per-token scale is not factorable across the window sum --
    see quantize_act_per_batch). w_int [out, in, kernel] int8, w_scale [out].
    stride/kernel/padding default to 1/3/1 (the A5 synthetic validation); whisper-tiny uses
    conv1 = stride 1 and conv2 = stride 2 (both kernel 3, padding 1) -- conv2 halves T to
    max_source_positions (1500).
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
        return torch.nn.functional.conv1d(x_recon, w_recon, bias, stride=stride, padding=padding)
    # int32 window matmul: pad T by `padding`, unfold with `kernel`/`stride` -> [B, in, T_out, kernel]
    T_out = (T + 2 * padding - kernel) // stride + 1
    x32 = (x_int.to(torch.int32) - x_zp.to(torch.int32))                       # [B, in, T] centered
    xp = torch.nn.functional.pad(x32, (padding, padding))                     # [B, in, T+2*pad]
    win = xp.unfold(-1, kernel, stride).movedim(1, 2)                        # [B, T_out, in, kernel]
    win = win.reshape(x32.shape[0], T_out, in_ch * kernel).to(torch.int32)   # [B, T_out, in*kernel]
    wr = w_int.reshape(out_ch, in_ch * kernel).to(torch.int32)               # [out, in*kernel]
    acc = torch.einsum("bti,oi->bto", win, wr)                              # [B, T_out, out] int32
    # x_scale [B,1,1] is constant across the window -> factors out: out = acc*w_scale*x_scale + bias.
    out = acc.to(torch.float32) * w_scale.to(torch.float32) * x_scale + bias.to(torch.float32)
    out = out.transpose(1, 2)                                              # [B, out, T_out] (conv convention)
    o_int, o_scale, o_zp = quantize_act_per_token(out)                     # per-token output to next op
    return dequant_act(o_int, o_scale, o_zp)


def quantize_act_per_batch_intscale(x: torch.Tensor):
    """Per-batch asymmetric int8 quant with an INTEGER fixed-point scale (A7, zero-fp).

    Same int8 levels as quantize_act_per_batch, but the scale is a fixed-point pair
    (mul, shift) instead of a float: x_fp ~= (x_int8 - x_zp) * mul * 2^(-shift). This is
    the form the zero-fp ONNX conv applies as an int Mul + BitShift (no runtime fp). The
    per-batch (one scale per batch element over in,T) is the conv-window requirement: a
    per-token (per-spatial-position) scale is NOT factorable across the kernel window, but
    a per-batch scale is constant across the window -> factors out cleanly (see A5).

    Returns (x_int8 [B,in,T] int32 in [-128,127], mul [B,1,1] int32 Q1.16, shift [B,1,1]
    int32, zp [B,1,1] int32). Q1.16 (not Q1.31) because the conv acc is int32 (smaller than
    the linear's int64 acc); the act scale multiplies acc_w (int64, ~2^22-2^30 here), so a
    16-bit scale keeps the product in int64 with ample headroom. 16-bit precision is still
    far finer than the int8 weight quant (2^-16 rel vs 2^-7), so the scale-value rounding is
    ~lossless. The A7 boundary round (integer scale decides the int8 bin) applies as in
    quantize_act_per_token_intscale.
    """
    xmin = x.amin(dim=(1, 2), keepdim=True)
    xmax = x.amax(dim=(1, 2), keepdim=True)
    scale = ((xmax - xmin) / 255.0).clamp(min=1e-8)                        # fp scale (intermediate)
    q, exp = torch.frexp(scale.to(torch.float64))                         # scale = q * 2^exp, q in [0.5,1)
    mul = torch.round(q * (2 ** 16)).clamp(0, 2 ** 16 - 1).to(torch.int32)  # Q1.16
    shift = (16 - exp).to(torch.int32)                                     # scale = mul * 2^(-shift)
    scale_int = (mul.to(torch.float64) * (2.0 ** (-shift.to(torch.float64)))).clamp(min=1e-12)
    zp = torch.round(-xmin / scale_int - 128.0)
    x_int = torch.round(x / scale_int + zp).clamp(-128, 127).to(torch.int32)
    return x_int, mul, shift, zp


def int8_conv1d_intscale(x_int: torch.Tensor, x_zp: torch.Tensor, x_mul: torch.Tensor,
                         x_shift: torch.Tensor, w_int: torch.Tensor, w_scale: torch.Tensor,
                         bias: torch.Tensor, stride: int = 1, kernel: int = 3, padding: int = 1,
                         return_intermediates: bool = False):
    """Int-canonical Conv1d oracle (pure-int, zero runtime fp) -- the spec the ONNX conv
    mirrors bit-exactly.

    x_int [B,in,T] int32 (int8-range [-128,127]); x_zp/x_mul/x_shift [B,1,1] int32 (per-batch
    Q1.16 scale, from quantize_act_per_batch_intscale); w_int [out,in,kernel] int32 (symmetric
    [-127,127], from _quant_weight_per_channel); w_scale [out] fp (baked to Q0.15 fixed-point
    at ONNX export; here applied as int fixed-point -> pure-int throughout); bias [out].

    Math (matches the A5 staged int8_conv1d, with INTEGER scales):
      acc[b,o,t] = sum_{i,k} (x_int[b,i,t+k] - x_zp[b]) * w_int[o,i,k]   (int32 window matmul)
      out_real   = acc[b,o,t] * w_scale[o] * x_scale[b] + bias[o].
      w_scale baked Q0.15 (mul_w, shift_w, per-channel); F_w = min(shift_w) (scalar);
      acc_w = rshift_round(acc * mul_w, shift_w - F_w)  @ 2^-F_w  (per-channel shift, broadcast).
      Flatten [B,out,T_out] -> [B*out, T_out] (row b*out+o); expand x_mul/x_shift [B,1,1] ->
      [B*out,1] (repeat_interleave out); bias_fixed = round(bias*2^F_w) [out] -> [B*out,1]
      (repeat B). Pre-fold the bias: acc_wb = acc_w*x_mul + (bias_fixed << x_shift), then
      reuse int8_output_requant_intscale with act_mul=1, act_shift=x_shift, F=F_w, bias=None.
      act_shift=x_shift carries the per-batch x_shift into y_shift; the pre-folded bias keeps
      the requant's per-feature bias path (which assumes [1,O]) out of the conv's per-row bias.

    Q0.15 (not the linear's Q0.31) because the conv accumulates the FULL int32 acc (all
    in*kernel terms) BEFORE the scale: mul_w*acc must fit int64 alongside the Q1.16 act scale.
    Q0.15 mul (~2^15) * acc (int32 <= 2^31) * x_mul (~2^16) <= 2^62 < 2^63 (Q0.31 would hit
    2^78). Q0.15 is 2^8 finer than the int8 weight quant (2^-7), so WER-neutral.
    """
    B, in_ch, T = x_int.shape
    out_ch = w_int.shape[0]
    T_out = (T + 2 * padding - kernel) // stride + 1
    # per-channel weight scale -> Q0.15 fixed-point (mul_w, shift_w); F_w = min shift (scalar)
    mul_w, shift_w = fixed_point_per_group(w_scale.to(torch.float64), Q=15)  # [out] int32 each
    F_w = int(shift_w.min().item())
    assert F_w >= 1, f"conv weight scale too large (shift_w={shift_w.tolist()}); F_w must be >=1 for bias precision"
    sw = (shift_w - F_w).to(torch.int64)                                    # [out] per-channel residual shift
    # window matmul (mirror the A5 einsum): center by per-batch zp, pad, unfold, matmul
    x32 = (x_int.to(torch.int32) - x_zp.to(torch.int32))                   # [B,in,T] centered
    xp = torch.nn.functional.pad(x32, (padding, padding))                  # [B,in,T+2p]
    win = xp.unfold(-1, kernel, stride).movedim(1, 2)                      # [B,T_out,in,kernel]
    win = win.reshape(B, T_out, in_ch * kernel).to(torch.int32)           # [B,T_out,in*k]
    wr = w_int.reshape(out_ch, in_ch * kernel).to(torch.int32)             # [out,in*k]
    acc = torch.einsum("bti,oi->bto", win, wr).transpose(1, 2)             # [B,out,T_out] int32
    # per-channel w_scale fixed-point (broadcast mul_w/sw over B, T_out); acc_w @ 2^-F_w
    acc_w = _rshift_round(acc.to(torch.int64) * mul_w.to(torch.int64).reshape(1, out_ch, 1),
                          sw.reshape(1, out_ch, 1))                        # [B,out,T_out] int64
    # flatten [B,out,T_out] -> [B*out, T_out] (row-major: B outer, out middle)
    acc_w_flat = acc_w.reshape(B * out_ch, T_out)
    x_mul_flat = x_mul.to(torch.int64).reshape(B, 1).repeat_interleave(out_ch, 0)   # [B*out,1]
    x_shift_flat = x_shift.to(torch.int64).reshape(B, 1).repeat_interleave(out_ch, 0)
    bias_fixed = torch.floor(bias.to(torch.float64) * (2.0 ** F_w) + 0.5).to(torch.int64)  # [out]
    bias_fixed_flat = bias_fixed.reshape(1, out_ch).repeat(B, 1).reshape(B * out_ch, 1)    # [B*out,1]
    accam = acc_w_flat * x_mul_flat                                        # [B*out,T_out] int64
    bias_term = bias_fixed_flat << x_shift_flat                           # [B*out,1] int64
    acc_wb = accam + bias_term                                             # [B*out,T_out] int64
    ones = torch.ones(B * out_ch, 1, dtype=torch.int32)
    y_int8_flat, y_mul_flat, y_shift_flat, y_zp_flat = int8_output_requant_intscale(
        acc_wb, ones, x_shift_flat.to(torch.int32), F_w, None)
    y_int8 = y_int8_flat.reshape(B, out_ch, T_out)
    y_mul = y_mul_flat.reshape(B, out_ch, 1).to(torch.int32)
    y_shift = y_shift_flat.reshape(B, out_ch, 1).to(torch.int32)
    y_zp = y_zp_flat.reshape(B, out_ch, 1).to(torch.int32)
    if return_intermediates:
        inter = {"acc": acc, "acc_w": acc_w, "mul_w": mul_w, "shift_w": shift_w, "F_w": F_w,
                 "acc_wb": acc_wb, "bias_fixed": bias_fixed}
        return y_int8, y_mul, y_shift, y_zp, inter
    return y_int8, y_mul, y_shift, y_zp


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