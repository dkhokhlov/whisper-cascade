#!/usr/bin/env python3
"""Zero-fp int8-compute ONNX export for the HQQ Whisper models (Phase B).

The deployment HW has no fp16 and no fp32 unit, so the emitted graph must do compute in int8
(int8 weights + int8 activations, int32/int64 accumulators, zero fp16/fp32). This module emits
the per-group int8 matmul as STANDARD ONNX int ops (opset 18, ORT-runnable), mirroring the
torch int8-compute reference in int8_compute.py / int8_forward.py. See docs/onnx.md
"Zero-fp int8-compute ONNX" and the plan.

Stage 1a (this module): one int8-compute LINEAR -- int inputs (x_int8, x_zp, act_mul,
act_shift) -> fp output (the A7.1 reference dequant). Every int intermediate (unpack, the four
MatMulInteger terms A/Bt/C/D, the zp corrections T1/T2, the signed fixed-point rshift_round
p1/p2, the accumulator) is BIT-EXACT vs int8_compute.int8_matmul_fixed_point (int-act path),
verified in tests/test_int8_onnx.py on a real 3-bit k_proj.

Stage 1b (this module, emit_output_requant=True): the same linear emits the int-canonical
output requant (the true zero-fp boundary) -> (y_int8, y_mul, y_shift, y_zp), all integer.
Mirrors int8_compute.int8_output_requant_intscale bit-exactly: out_int = acc*act_mul +
(bias_fixed << act_shift); R = rmax-rmin clamped to [1, 2^62); e = bit_length(R)-8 via a CLZ
ladder (no CLZ/log2-int in ONNX); mul = round_half_up(R*2^sh/255) normalized to [2^15,2^16);
zp = round_half_up(-rmin*2^sh/mul)-128; y_int8 = clamp(round_half_up(out_int*2^sh/mul)+zp).
Round-half-up = floor((2*num+den)/(2*den)) with a signed-floor correction (ONNX Div truncates
toward 0). Bit-exact vs the reference across 3-bit + 8-bit layers, batch 1/2/3, extremes
(all-positive, all-negative, large magnitude). The remaining Q6 sub-modules (LN, exact-Phi
GELU LUT, softmax, ConvInteger, Loop) and the full recursive zero-fp audit are next.

Key ONNX facts (opset 18, ORT 1.19.2) learned and encoded here:
  * 3-bit/2-bit unpack via Div/Mod (W_q >= 0 so floor == trunc); BitShift is unsigned-only.
  * Batched-over-G MatMulInteger: A=[G,B,gs] x B=[G,gs,O] -> [G,B,O] -> Transpose [B,O,G].
    The batch dim must match on dim 0; [B,G,gs]@[G,gs,O] is rejected (B != G broadcast).
  * MatMulInteger zp is scalar-only in ORT 1.19.2; HQQ zero is per-(o,g). Pass zp=0 and emit the
    -x_zp*B / zero*(C-x_zp*D) corrections explicitly as int64 Add/Sub/Mul.
  * Signed rshift_round (round-half-up) with NO signed BitShift: precompute int64 half=2^max
    (s-1,0) and p=2^s (s >= 0 always since F = min(sh_s, sh_z)), then
    vh=v+half; q=Div(vh,p); r=Mod(vh,p); out = q - Cast(neg & (r != 0)). Div truncates toward 0
    (ceil for negatives), so subtract 1 on a nonzero negative remainder -> floor = round-half-up.
  * Transpose perm is an attribute (not an input); ReduceSum axes is an INPUT in opset >= 13 and
    rejects int8 (cast to int32); MatMulInteger needs 8-bit inputs; BitShift inputs must match
    type (both uint64); output value_info dtype must match the producing op.
"""

import numpy as np
import torch
import onnx
from onnx import helper, TensorProto as T

INT8 = T.INT8
INT32 = T.INT32
INT64 = T.INT64
UINT8 = T.UINT8
UINT64 = T.UINT64
FLOAT = T.FLOAT
BOOL = T.BOOL

# 3-bit packing: 10 levels per int32, shifts [27,24,...,0], mask 0b111.
_PACK_3BIT = {"shifts": [27, 24, 21, 18, 15, 12, 9, 6, 3, 0], "mod": 8, "levels_per_int": 10}
# mod=8 for ONNX Mod (the modulus); BitwiseAnd would use 0b111=7. We use Mod since BitShift
# is unsigned-only and W_q >= 0 so Div/Mod floor == trunc.
# 2-bit packing: 4 levels per uint8, but hqq stores 2bit_u8 as contiguous row slabs; not used
# (2-bit is broken/dropped). 8-bit has no bit-packing.
_PACK_8BIT = None


def _meta(hqq):
    m = hqq.meta
    O, I = (int(s) for s in m["shape"])
    Wq = m.get("_W_q", None)
    if Wq is None:
        Wq = getattr(hqq, "W_q", None)
    return {
        "nbits": int(m["nbits"]), "packing": m["packing"], "O": O, "I": I,
        "g": int(m["group_size"]), "axis": int(m["axis"]),
        "W_q": Wq,
        "zero": m["zero"].to(torch.float32).numpy().reshape(-1),
        "scale": m["scale"].to(torch.float32).numpy().reshape(-1),
        "bias": (hqq.bias.to(torch.float32).numpy() if hqq.bias is not None else None),
    }


class _Builder:
    """Accumulates initializers + nodes for one int8-compute linear ONNX graph."""

    def __init__(self):
        self.inits = []
        self.nodes = []
        self._n = 0
        self._seen = set()

    def init(self, name, arr, dtype):
        if name in self._seen:
            return name
        self._seen.add(name)
        arr = np.asarray(arr)
        self.inits.append(helper.make_tensor(name, dtype, arr.shape, arr.flatten().tolist()))
        return name

    def node(self, op, inputs, outputs, name=None, **attrs):
        self._n += 1
        if name is None:
            name = f"{op.lower()}_{self._n}"
        self.nodes.append(helper.make_node(op, inputs, outputs, name=name, **attrs))
        return outputs[-1]

    def const(self, name, arr, dtype):
        return self.init(name, arr, dtype)


def _unpack(builder, meta, O, I, g, G):
    """packed W_q -> qr_g [O,G,gs] int32 RAW levels + qr_8 int8-baked [G,gs,O] for MatMulInteger.

    Returns (qr_g, qr_8, corr). qr_g is the raw int32 levels (0..7 for 3-bit, 0..255 for
    8-bit) used for Bt = ReduceSum and emitted as the bit-exact intermediate. qr_8 is the
    int8 weight tensor (transposed to [G,gs,O]) fed to MatMulInteger. corr is the level
    offset baked into qr_8 (0 for 3-bit, 128 for 8-bit): MatMulInteger gives
    A_s = sum x*(qr-corr) = A_ref - corr*C, so the caller reconstructs A = A_s + corr*C.
    3-bit levels 0..7 fit int8 losslessly (corr=0). 8-bit levels 0..255 need the -128 bake
    because ORT 1.19.2 implements batched MatMulInteger only when BOTH inputs share a dtype
    (int8/uint8 is NOT_IMPLEMENTED), and activations are int8.
    """
    nbits, packing = meta["nbits"], meta["packing"]
    Wq = meta["W_q"]
    if packing == "8bit_u8":
        # 1:1, no bit-packing. W_q is uint8 levels row-major; reshape to [O,G,gs] directly.
        builder.init("W_q", Wq.to(torch.int32).numpy(), INT32)  # store as int32 for reshape
        builder.init("shp_qrg", np.array([O, G, g], dtype=np.int64), INT64)
        builder.node("Reshape", ["W_q", "shp_qrg"], ["qr_g"], "unpack_reshape8")
        corr = 128
    elif packing == "3bit_32":
        cfg = _PACK_3BIT
        builder.init("W_q", Wq.to(torch.int32).numpy(), INT32)
        for k, sh in enumerate(cfg["shifts"]):
            builder.init(f"pow{k}", np.array([1 << sh], dtype=np.int32), INT32)
        builder.init("mask", np.array([cfg["mod"], ], dtype=np.int32), INT32)  # Mod modulus (8)
        lvls = []
        for k in range(len(cfg["shifts"])):
            dv, lv = f"dv{k}", f"lv{k}"
            builder.node("Div", ["W_q", f"pow{k}"], [dv], f"unpack_div{k}")
            builder.node("Mod", [dv, "mask"], [lv], f"unpack_mod{k}")
            lvls.append(lv)
        builder.node("Concat", lvls, ["tmp_all"], "unpack_concat", axis=0)
        N = O * G
        builder.init("N_end", np.array([N], dtype=np.int64), INT64)
        builder.init("ax0", np.array([0], dtype=np.int64), INT64)
        builder.init("shp_qrg", np.array([O, G, g], dtype=np.int64), INT64)
        builder.node("Slice", ["tmp_all", "zero_i64", "N_end", "ax0"], ["tmp_N"], "unpack_slice")
        builder.node("Reshape", ["tmp_N", "shp_qrg"], ["qr_g"], "unpack_reshape3")
        corr = 0  # 3-bit levels 0..7 fit int8 losslessly
    else:
        raise NotImplementedError(f"unpack: packing={packing} not supported (3-bit/8-bit only)")
    # bake the level offset, cast to int8, transpose [O,G,gs] -> [G,gs,O] for MatMulInteger
    builder.init("corr_i32", np.array([corr], dtype=np.int32), INT32)
    builder.node("Sub", ["qr_g", "corr_i32"], ["qr_off"], "unpack_subcorr")
    builder.node("Cast", ["qr_off"], ["qr_off_i8"], "unpack_qcast", to=INT8)
    builder.node("Transpose", ["qr_off_i8"], ["qr_8"], "unpack_qtranspose", perm=[1, 2, 0])
    return "qr_g", "qr_8", corr


def _rshift_round(b, v, mul, half, p, tag):
    """Signed round-half-up right shift of int64 v by per-element s, via precomputed int64
    initializers half=2^max(s-1,0), p=2^s. out = floor((v+half)/p) = round-half-up(v/2^s)."""
    b.node("Cast", [mul], [f"{tag}_m64"], f"{tag}_mcast", to=INT64)
    b.node("Mul", [v, f"{tag}_m64"], [f"{tag}_prod"], f"{tag}_mul")
    b.node("Add", [f"{tag}_prod", half], [f"{tag}_vh"], f"{tag}_add")
    b.node("Div", [f"{tag}_vh", p], [f"{tag}_q"], f"{tag}_div")
    b.node("Mod", [f"{tag}_vh", p], [f"{tag}_r"], f"{tag}_mod")
    b.node("Less", [f"{tag}_vh", "zero_i64"], [f"{tag}_neg"], f"{tag}_less")
    b.node("Equal", [f"{tag}_r", "zero_i64"], [f"{tag}_eq"], f"{tag}_eq")
    b.node("Not", [f"{tag}_eq"], [f"{tag}_nz"], f"{tag}_nz")
    b.node("And", [f"{tag}_neg", f"{tag}_nz"], [f"{tag}_cb"], f"{tag}_and")
    b.node("Cast", [f"{tag}_cb"], [f"{tag}_corr"], f"{tag}_cast", to=INT64)
    b.node("Sub", [f"{tag}_q", f"{tag}_corr"], [f"{tag}_out"], f"{tag}_sub")
    return f"{tag}_out"


def _rhu(b, num, den, tag):
    """Integer round-half-up of num/den = floor((2*num+den)/(2*den)) with signed-floor
    correction. ONNX Div truncates toward 0 (ceil for negatives); subtract 1 when the
    numerator is negative and the remainder is nonzero -> floor = round-half-up. Mirrors
    int8_compute._round_half_up exactly. All arithmetic in int64."""
    b.init("c2_i64", np.array([2], dtype=np.int64), INT64)
    b.init("zero_i64", np.array([0], dtype=np.int64), INT64)
    a = b.node("Mul", [num, "c2_i64"], [f"{tag}_2n"], f"{tag}_2n")           # 2*num
    a = b.node("Add", [a, den], [f"{tag}_a"], f"{tag}_a")                    # 2*num+den
    p = b.node("Mul", [den, "c2_i64"], [f"{tag}_p"], f"{tag}_p")             # 2*den
    q = b.node("Div", [a, p], [f"{tag}_q"], f"{tag}_div")
    r = b.node("Mod", [a, p], [f"{tag}_r"], f"{tag}_mod")
    neg = b.node("Less", [a, "zero_i64"], [f"{tag}_neg"], f"{tag}_less")
    eq = b.node("Equal", [r, "zero_i64"], [f"{tag}_eq"], f"{tag}_eq")
    nz = b.node("Not", [eq], [f"{tag}_nz"], f"{tag}_nz")
    cb = b.node("And", [neg, nz], [f"{tag}_cb"], f"{tag}_and")
    corr = b.node("Cast", [cb], [f"{tag}_c"], f"{tag}_cast", to=INT64)
    return b.node("Sub", [q, corr], [f"{tag}_out"], f"{tag}_sub")


def _clz_ladder(b, R):
    """floor(log2(R)) + 1 = bit-length, via a binary-search CLZ ladder (no CLZ/log2-int in
    ONNX). R is int64 [*,1], 1 <= R < 2^62 (the cap62 guard). Returns the int64 bit-length.
    Thresholds are built as uint64 BitShift(1, k) then cast to int64 for the < comparison;
    the largest threshold reached is 2^62 (R < 2^62 by cap, so the 2^63 step is never taken,
    avoiding the int64-overflow cast of 2^63)."""
    b.init("zero_i64", np.array([0], dtype=np.int64), INT64)
    b.init("one_u64", np.array([1], dtype=np.uint64), UINT64)
    b.init("one_i64", np.array([1], dtype=np.int64), INT64)
    b.init("k32_i64", np.array([32], dtype=np.int64), INT64)
    b.init("k32_u64", np.array([32], dtype=np.uint64), UINT64)
    for k in (16, 8, 4, 2, 1):
        b.init(f"k{k}_i64", np.array([k], dtype=np.int64), INT64)
        b.init(f"k{k}_u64", np.array([k], dtype=np.uint64), UINT64)
    # level 1: 2^32. R < 2^32 -> bitlen < 33 (base 0); else base 32.
    t = b.node("BitShift", ["one_u64", "k32_u64"], ["clz_t1u"], "clz_t1u", direction="LEFT")
    t = b.node("Cast", [t], ["clz_tA"], "clz_tA", to=INT64)
    lt = b.node("Less", [R, t], ["clz_ltA"], "clz_ltA")
    ge = b.node("Not", [lt], ["clz_geA"], "clz_geA")
    b0 = b.node("Where", [ge, "k32_i64", "zero_i64"], ["clz_bbase"], "clz_bbase")
    bcur = "clz_bbase"
    for lvl, k in enumerate([16, 8, 4, 2, 1]):
        bk = b.node("Add", [bcur, f"k{k}_i64"], [f"clz_bk{lvl}"], f"clz_bk{lvl}")
        bku = b.node("Cast", [bk], [f"clz_bku{lvl}"], f"clz_bku{lvl}", to=UINT64)
        tu = b.node("BitShift", ["one_u64", bku], [f"clz_tu{lvl}"], f"clz_tu{lvl}", direction="LEFT")
        ti = b.node("Cast", [tu], [f"clz_ti{lvl}"], f"clz_ti{lvl}", to=INT64)
        lt = b.node("Less", [R, ti], [f"clz_lt{lvl}"], f"clz_lt{lvl}")
        ge = b.node("Not", [lt], [f"clz_ge{lvl}"], f"clz_ge{lvl}")
        w = b.node("Where", [ge, f"k{k}_i64", "zero_i64"], [f"clz_w{lvl}"], f"clz_w{lvl}")
        bcur = b.node("Add", [bcur, w], [f"clz_b{lvl}"], f"clz_b{lvl}")
    return b.node("Add", [bcur, "one_i64"], ["clz_bitlen"], "clz_bitlen")   # bitlen = b+1


def _emit_output_requant(b, acc, act_mul, act_shift, F, bias, O):
    """Stage 1b int-canonical output requant (the true zero-fp boundary). Consumes the
    int64 accumulator `acc` [B,O] @ 2^-F, the per-token int act scale (act_mul, act_shift),
    and the export-time-baked int bias; produces (y_int8 [B,O], y_mul [B,1], y_shift [B,1],
    y_zp [B,1]) all integer. Mirrors int8_compute.int8_output_requant_intscale bit-exactly:
      out_int = acc*act_mul + (bias_fixed << act_shift)          (int64)
      R = max(out_int)-min(out_int), clamped to [1, 2^62)
      e = bit_length(R) - 8 ;  sh = 16 - e
      mul = round_half_up(R * 2^sh / 255) ;  normalize mul to [2^15, 2^16) (adjust sh by +-1)
      zp = round_half_up(-rmin * 2^sh / mul) - 128
      y_int8 = clamp(round_half_up(out_int * 2^sh / mul) + zp, -128, 127)
      y_shift = sh + F + act_shift
    All runtime ops are integer (int64 mul/shift, uint64 BitShift for the left shifts,
    round-half-up via int Div/Mod + signed-floor correction). F is baked as an int64
    initializer (export-time fp->int, not runtime fp). Returns the 4 output tensor names.
    """
    b.init("zero_i64", np.array([0], dtype=np.int64), INT64)
    b.init("one_i64", np.array([1], dtype=np.int64), INT64)
    b.init("F_i64", np.array([F], dtype=np.int64), INT64)
    b.init("c255_i64", np.array([255], dtype=np.int64), INT64)
    b.init("c255_u", np.array([255], dtype=np.uint64), UINT64)
    b.init("c128_i64", np.array([128], dtype=np.int64), INT64)
    b.init("lo15_i64", np.array([1 << 15], dtype=np.int64), INT64)
    b.init("hi16_i64", np.array([1 << 16], dtype=np.int64), INT64)
    b.init("lo128_i64", np.array([-128], dtype=np.int64), INT64)
    b.init("hi127_i64", np.array([127], dtype=np.int64), INT64)
    b.init("cap62_i64", np.array([(1 << 62) - 1], dtype=np.int64), INT64)
    b.init("axes_neg1", np.array([-1], dtype=np.int64), INT64)
    # out_int = acc*am + (bias_fixed << act_shift)
    am64 = b.node("Cast", [act_mul], ["am64"], "am64", to=INT64)
    ash64 = b.node("Cast", [act_shift], ["ash64"], "ash64", to=INT64)
    accam = b.node("Mul", [acc, am64], ["accam"], "accam")                  # [B,O] int64
    if bias is not None:
        bf = np.floor(bias.astype(np.float64) * (2.0 ** F) + 0.5).astype(np.int64)
        b.init("bf", bf.reshape(1, O), INT64)                               # [1,O]
        ash64u = b.node("Cast", [ash64], ["ash64u_bf"], "ash64u_bf", to=UINT64)
        bf_u = b.node("Cast", ["bf"], ["bf_u"], "bf_u", to=UINT64)
        bfterm = b.node("BitShift", [bf_u, ash64u], ["bfterm"], "bfterm", direction="LEFT")
        bfterm_i = b.node("Cast", [bfterm], ["bfterm_i"], "bfterm_i", to=INT64)
        out_int = b.node("Add", [accam, bfterm_i], ["out_int"], "out_int")
    else:
        out_int = b.node("Identity", [accam], ["out_int"], "out_int")
    # rmax/rmin/R
    rmax = b.node("ReduceMax", [out_int, "axes_neg1"], ["rmax"], "rmax", keepdims=1)
    rmin = b.node("ReduceMin", [out_int, "axes_neg1"], ["rmin"], "rmin", keepdims=1)
    Rraw = b.node("Sub", [rmax, rmin], ["Rraw"], "Rraw")
    R = b.node("Max", [Rraw, "one_i64"], ["R"], "R")
    R = b.node("Min", [R, "cap62_i64"], ["Rc"], "Rc")                       # [B,1] < 2^62
    # e = bitlen(R) - 8 ; sh = 16 - e
    bitlen = _clz_ladder(b, R)
    e = b.node("Sub", [bitlen, "k8_i64"], ["e"], "e")
    c16_i64 = b.init("c16_i64", np.array([16], dtype=np.int64), INT64)
    sh = b.node("Sub", [c16_i64, e], ["sh"], "sh")

    def _mul_at(sh_name, tag):
        pos = b.node("Not", [b.node("Less", [sh_name, "zero_i64"], [f"{tag}_plt"], f"{tag}_plt")],
                     [f"{tag}_pos"], f"{tag}_pos")
        shp = b.node("Max", [sh_name, "zero_i64"], [f"{tag}_shp"], f"{tag}_shp")
        neg_sh = b.node("Sub", ["zero_i64", sh_name], [f"{tag}_nsh"], f"{tag}_nsh")
        shn = b.node("Max", ["zero_i64", neg_sh], [f"{tag}_shn"], f"{tag}_shn")
        Ru = b.node("Cast", [R], [f"{tag}_Ru"], f"{tag}_Ru", to=UINT64)
        shp_u = b.node("Cast", [shp], [f"{tag}_shpu"], f"{tag}_shpu", to=UINT64)
        shn_u = b.node("Cast", [shn], [f"{tag}_shnu"], f"{tag}_shnu", to=UINT64)
        rs = b.node("BitShift", [Ru, shp_u], [f"{tag}_rs"], f"{tag}_rs", direction="LEFT")
        ds = b.node("BitShift", ["c255_u", shn_u], [f"{tag}_ds"], f"{tag}_ds", direction="LEFT")
        rs_i = b.node("Cast", [rs], [f"{tag}_rsi"], f"{tag}_rsi", to=INT64)
        ds_i = b.node("Cast", [ds], [f"{tag}_dsi"], f"{tag}_dsi", to=INT64)
        num_m = b.node("Where", [pos, rs_i, R], [f"{tag}_num"], f"{tag}_num")
        den_m = b.node("Where", [pos, "c255_i64", ds_i], [f"{tag}_den"], f"{tag}_den")
        return _rhu(b, num_m, den_m, f"{tag}_m"), pos, shp_u, shn_u

    mul, _pos, _shp_u, _shn_u = _mul_at(sh, "mul0")
    # normalize mul to [2^15, 2^16): if mul<2^15 sh+=1 ; if mul>=2^16 sh-=1 ; recompute
    ts = b.node("Less", [mul, "lo15_i64"], ["ts"], "ts")
    tb = b.node("Not", [b.node("Less", [mul, "hi16_i64"], ["tb_lt"], "tb_lt")], ["tb"], "tb")
    sh = b.node("Where", [ts, b.node("Add", [sh, "one_i64"], ["sh_p1"], "sh_p1"), sh], ["sh_a"], "sh_a")
    sh = b.node("Where", [tb, b.node("Sub", [sh, "one_i64"], ["sh_m1"], "sh_m1"), sh], ["sh_n"], "sh_n")
    mul, _pos3, shp3_u, shn3_u = _mul_at(sh, "mul1")
    # zp = rhu(-rmin * 2^sh / mul) - 128
    neg_rmin = b.node("Sub", ["zero_i64", rmin], ["neg_rmin"], "neg_rmin")
    neg_rmin_u = b.node("Cast", [neg_rmin], ["neg_rmin_u"], "neg_rmin_u", to=UINT64)
    nz_shift = b.node("BitShift", [neg_rmin_u, shp3_u], ["nz_shift"], "nz_shift", direction="LEFT")
    nz_shift_i = b.node("Cast", [nz_shift], ["nz_shift_i"], "nz_shift_i", to=INT64)
    num_z = b.node("Where", [_pos3, nz_shift_i, neg_rmin], ["num_z"], "num_z")
    mul_u = b.node("Cast", [mul], ["mul_u"], "mul_u", to=UINT64)
    mul_shn3 = b.node("BitShift", [mul_u, shn3_u], ["mul_shn3"], "mul_shn3", direction="LEFT")
    mul_shn3_i = b.node("Cast", [mul_shn3], ["mul_shn3_i"], "mul_shn3_i", to=INT64)
    den_z = b.node("Where", [_pos3, mul, mul_shn3_i], ["den_z"], "den_z")
    zp = b.node("Sub", [_rhu(b, num_z, den_z, "zp"), "c128_i64"], ["zp"], "zp_final")
    # y_int8 = clamp(rhu(out_int * 2^sh / mul) + zp, -128, 127)
    out_int_u = b.node("Cast", [out_int], ["out_int_u"], "out_int_u", to=UINT64)
    oy_shift = b.node("BitShift", [out_int_u, shp3_u], ["oy_shift"], "oy_shift", direction="LEFT")
    oy_shift_i = b.node("Cast", [oy_shift], ["oy_shift_i"], "oy_shift_i", to=INT64)
    num_y = b.node("Where", [_pos3, oy_shift_i, out_int], ["num_y"], "num_y")
    den_y = b.node("Where", [_pos3, mul, mul_shn3_i], ["den_y"], "den_y")
    y_arg = _rhu(b, num_y, den_y, "y")
    y_plus = b.node("Add", [y_arg, zp], ["y_plus"], "y_plus")
    y_lo = b.node("Max", [y_plus, "lo128_i64"], ["y_lo"], "y_lo")
    y_hi = b.node("Min", [y_lo, "hi127_i64"], ["y_hi"], "y_hi")
    y_int8 = b.node("Cast", [y_hi], ["y_int8"], "y_int8", to=INT8)
    # y_shift = sh + F + act_shift ; y_mul = mul ; y_zp = zp  (all to int32)
    y_sh_f = b.node("Add", [sh, "F_i64"], ["y_sh_F"], "y_sh_F")
    y_sh_pre = b.node("Add", [y_sh_f, ash64], ["y_shift_pre"], "y_shift_pre")
    y_mul = b.node("Cast", [mul], ["y_mul"], "y_mul", to=INT32)
    y_zp = b.node("Cast", [zp], ["y_zp"], "y_zp", to=INT32)
    y_shift = b.node("Cast", [y_sh_pre], ["y_shift"], "y_shift", to=INT32)
    return "y_int8", "y_mul", "y_shift", "y_zp"


def build_int8_linear_onnx(hqq_linear, model_name="int8_linear", emit_intermediates=True,
                           emit_output_requant=False):
    """Build the Stage 1a int8-compute linear ONNX (int inputs -> fp output).

    Inputs (int, zero runtime fp):
      x_int [B, I] int8, x_zp [B,1] int32, act_mul [B,1] int32 (Q1.16), act_shift [B,1] int32.
    Output: out [B, O] fp32 (the A7.1 reference dequant). With emit_intermediates=True the
    graph also outputs qr_g, A, Bt, C, T1, T2, p1_out, p2_out, acc for bit-exact verification.
    """
    import torch
    b = _Builder()
    meta = _meta(hqq_linear)
    O, I, g, G = meta["O"], meta["I"], meta["g"], meta["I"] // meta["g"]

    # ---- per-group fixed-point initializers (export-time fp -> int bake) ----
    zero_g = meta["zero"].reshape(O, G)
    scale_g = meta["scale"].reshape(O, G)
    zscale_g = zero_g * scale_g
    # mirror int8_compute.fixed_point_per_group: frexp -> Q0.31 mul + right_shift
    def fp(s):
        s = np.asarray(s, dtype=np.float64)
        q = np.zeros_like(s, dtype=np.float64)
        exp = np.zeros_like(s, dtype=np.int32)
        for idx in np.ndindex(s.shape):
            qi, ei = np.frexp(s[idx])
            q[idx] = qi
            exp[idx] = ei
        mul = np.round(q * (2 ** 31)).clip(0, 2 ** 31 - 1).astype(np.int32)
        right_shift = (-(np.int64(exp) - 31)).clip(min=0).astype(np.int32)
        return mul, right_shift
    mul_s, sh_s = fp(scale_g)
    mul_z, sh_z = fp(zscale_g)
    F = int(min(sh_s.min(), sh_z.min()))
    s_s = (sh_s - F).astype(np.int64)
    s_z = (sh_z - F).astype(np.int64)
    half_s = (2 ** np.clip(s_s - 1, 0, None)).astype(np.int64)
    p_s = (2 ** s_s).astype(np.int64)
    half_z = (2 ** np.clip(s_z - 1, 0, None)).astype(np.int64)
    p_z = (2 ** s_z).astype(np.int64)
    b.init("mul_s", mul_s.reshape(1, O, G), INT64)
    b.init("mul_z", mul_z.reshape(1, O, G), INT64)
    b.init("half_s", half_s.reshape(1, O, G), INT64)
    b.init("p_s", p_s.reshape(1, O, G), INT64)
    b.init("half_z", half_z.reshape(1, O, G), INT64)
    b.init("p_z", p_z.reshape(1, O, G), INT64)
    b.init("D", np.array([g], dtype=np.int64), INT64)
    b.init("zero_i64", np.array([0], dtype=np.int64), INT64)
    b.init("axes0", np.array([0], dtype=np.int64), INT64)
    b.init("axes2", np.array([2], dtype=np.int64), INT64)
    b.init("Fconst", np.array([F], dtype=np.int32), INT32)
    b.init("one_u64", np.array([1], dtype=np.uint64), UINT64)
    b.init("shp_xg", np.array([-1, G, g], dtype=np.int64), INT64)
    b.init("shp_xz3", np.array([-1, 1, 1], dtype=np.int64), INT64)
    b.init("shp_c1g", np.array([-1, 1, G], dtype=np.int64), INT64)

    # ---- 1) unpack W_q -> qr_g [O,G,gs] int32 (raw) + qr_8 int8-baked [G,gs,O] ----
    qr_g, qr_8, corr = _unpack(b, meta, O, I, g, G)

    # ---- 2) x_int [B,I] -> [B,G,gs] -> [G,B,gs] ----
    b.node("Reshape", ["x_int", "shp_xg"], ["x_bgg"], "x_reshape")
    b.node("Transpose", ["x_bgg"], ["x_gbg"], "x_transpose", perm=[1, 0, 2])  # [G,B,gs]

    # ---- 3) A_s = MatMulInteger([G,B,gs], [G,gs,O]) -> [G,B,O] -> [B,O,G] int32 ----
    #       A = A_s + corr*C  (reconstruct raw-level A from the int8-baked weight)
    b.node("MatMulInteger", ["x_gbg", qr_8], ["A_gbo"], "mm_integer")
    b.node("Transpose", ["A_gbo"], ["A_s"], "A_transpose", perm=[1, 2, 0])  # [B,O,G] int32

    # ---- 4) Bt = sum_i qr[o,i] -> [O,G] ; C = sum_i x[b,i] -> [B,G] ----
    b.node("ReduceSum", [qr_g, "axes2"], ["Bt"], "Bt_sum", keepdims=0)        # int32
    b.node("Cast", ["x_bgg"], ["x_bgg_i32"], "xg_cast", to=INT32)
    b.node("ReduceSum", ["x_bgg_i32", "axes2"], ["C"], "C_sum", keepdims=0)  # int32

    # ---- 4b) A = A_s + corr*C  (skip when corr=0, i.e. 3-bit) ----
    if corr:
        b.init("corr_mul", np.array([corr], dtype=np.int32), INT32)
        b.node("Reshape", ["C", "shp_c1g"], ["C_b1g_a"], "C_reshapeA")        # [B,1,G]
        b.node("Mul", ["C_b1g_a", "corr_mul"], ["corrC"], "A_corr_mul")     # [B,1,G] int32
        b.node("Add", ["A_s", "corrC"], ["A"], "A_corr_add")                # [B,O,G] int32
    else:
        b.node("Identity", ["A_s"], ["A"], "A_id")

    # ---- 5) zp corrections: T1 = A - x_zp*Bt ; T2 = C - x_zp*D ----
    b.node("Reshape", ["x_zp", "shp_xz3"], ["xz3"], "xz3_reshape")           # [B,1,1]
    b.node("Cast", ["A"], ["A_i64"], "A_cast", to=INT64)
    b.node("Cast", ["Bt"], ["Bt_i64"], "Bt_cast", to=INT64)
    b.node("Cast", ["xz3"], ["xz3_i64"], "xz3_cast", to=INT64)
    b.node("Unsqueeze", ["Bt_i64", "axes0"], ["Bt_1og"], "Bt_unsq")           # [1,O,G]
    b.node("Mul", ["xz3_i64", "Bt_1og"], ["xzBt"], "T1_mul")                 # [B,O,G]
    b.node("Sub", ["A_i64", "xzBt"], ["T1"], "T1_sub")                        # [B,O,G] int64
    b.node("Cast", ["C"], ["C_i64"], "C_cast", to=INT64)
    b.node("Reshape", ["C_i64", "shp_c1g"], ["C_b1g"], "C_reshape")          # [B,1,G]
    b.node("Mul", ["xz3_i64", "D"], ["xzD"], "T2_mul")                       # [B,1,1]
    b.node("Sub", ["C_b1g", "xzD"], ["T2"], "T2_sub")                        # [B,1,G] int64

    # ---- 6) p1 = rshift_round(T1*mul_s) ; p2 = rshift_round(T2*mul_z) ----
    p1 = _rshift_round(b, "T1", "mul_s", "half_s", "p_s", "p1")             # [B,O,G] int64
    p2 = _rshift_round(b, "T2", "mul_z", "half_z", "p_z", "p2")             # [B,O,G] int64

    # ---- 7) acc = sum_g (p1 - p2) -> [B,O] int64 ----
    b.node("Sub", [p1, p2], ["pdiff"], "pdiff")
    b.node("ReduceSum", ["pdiff", "axes2"], ["acc"], "acc_sum", keepdims=0)   # [B,O] int64

    # ---- 8) output: Stage 1a fp dequant OR Stage 1b int output requant ----
    if emit_output_requant:
        _emit_output_requant(b, "acc", "act_mul", "act_shift", F, meta["bias"], O)
    else:
        b.node("Cast", ["acc"], ["acc_f"], "acc_castf", to=FLOAT)
        b.node("Cast", ["act_mul"], ["am_f"], "am_castf", to=FLOAT)
        b.node("Mul", ["acc_f", "am_f"], ["accam_f"], "out_mul")
        b.node("Add", ["Fconst", "act_shift"], ["tot_shift"], "ts_add")          # [B,1] int32
        b.node("Cast", ["tot_shift"], ["ts_u64"], "ts_u64", to=UINT64)
        b.node("BitShift", ["one_u64", "ts_u64"], ["denom_u64"], "denom_shift", direction="LEFT")
        b.node("Cast", ["denom_u64"], ["denom_f"], "denom_castf", to=FLOAT)
        b.node("Div", ["accam_f", "denom_f"], ["out_pre"], "out_div")
        if meta["bias"] is not None:
            b.init("bias", meta["bias"].astype(np.float32), FLOAT)
            b.node("Add", ["out_pre", "bias"], ["out"], "out_bias")
        else:
            b.node("Identity", ["out_pre"], ["out"], "out_id")

    # ---- graph ----
    inputs = [
        helper.make_tensor_value_info("x_int", INT8, ["B", I]),
        helper.make_tensor_value_info("x_zp", INT32, ["B", 1]),
        helper.make_tensor_value_info("act_mul", INT32, ["B", 1]),
        helper.make_tensor_value_info("act_shift", INT32, ["B", 1]),
    ]
    vi = helper.make_tensor_value_info
    if emit_output_requant:
        outputs = [
            vi("y_int8", INT8, ["B", O]),
            vi("y_mul", INT32, ["B", 1]),
            vi("y_shift", INT32, ["B", 1]),
            vi("y_zp", INT32, ["B", 1]),
        ]
    else:
        outputs = [vi("out", FLOAT, ["B", O])]
    if emit_intermediates:
        outputs += [
            vi("qr_g", INT32, [O, G, g]),
            vi("A", INT32, ["B", O, G]),
            vi("Bt", INT32, [O, G]),
            vi("C", INT32, ["B", G]),
            vi("T1", INT64, ["B", O, G]),
            vi("T2", INT64, ["B", 1, G]),
            vi("p1_out", INT64, ["B", O, G]),
            vi("p2_out", INT64, ["B", O, G]),
            vi("acc", INT64, ["B", O]),
        ]
    g_ = helper.make_graph(b.nodes, model_name, inputs, outputs, initializer=b.inits)
    m = helper.make_model(g_, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


def zero_fp_audit(model):
    """Recursive zero-fp audit: assert no fp16/fp32/bfloat16/float8 anywhere in the graph
    (inputs/outputs/value_info/initializers/Constant attrs/nested graph attrs). FAIL CLOSED on
    unknown types. Stage 1b will also audit the ORT-optimized artifact. (Stub for the full
    recursive impl; the linear Stage 1a intentionally has an fp dequant output.)"""
    raise NotImplementedError("full recursive zero-fp audit is Stage 1b")