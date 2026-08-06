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


def _clz_ladder(b, R, tag="clz"):
    """floor(log2(R)) + 1 = bit-length, via a binary-search CLZ ladder (no CLZ/log2-int in
    ONNX). R is int64 [*,1], 1 <= R < 2^62 (the cap62 guard). Returns the int64 bit-length.
    Thresholds are built as uint64 BitShift(1, k) then cast to int64 for the < comparison;
    the largest threshold reached is 2^62 (R < 2^62 by cap, so the 2^63 step is never taken,
    avoiding the int64-overflow cast of 2^63). `tag` namespaces the node names so the ladder
    can be emitted more than once in one graph (e.g. the LayerNorm emits it for s_K AND the
    output requant emits it again for R) without SSA name collisions."""
    b.init("zero_i64", np.array([0], dtype=np.int64), INT64)
    b.init("one_u64", np.array([1], dtype=np.uint64), UINT64)
    b.init("one_i64", np.array([1], dtype=np.int64), INT64)
    b.init("k32_i64", np.array([32], dtype=np.int64), INT64)
    b.init("k32_u64", np.array([32], dtype=np.uint64), UINT64)
    for k in (16, 8, 4, 2, 1):
        b.init(f"k{k}_i64", np.array([k], dtype=np.int64), INT64)
        b.init(f"k{k}_u64", np.array([k], dtype=np.uint64), UINT64)
    p = f"{tag}_"
    # level 1: 2^32. R < 2^32 -> bitlen < 33 (base 0); else base 32.
    t = b.node("BitShift", ["one_u64", "k32_u64"], [f"{p}t1u"], f"{p}t1u", direction="LEFT")
    t = b.node("Cast", [t], [f"{p}tA"], f"{p}tA", to=INT64)
    lt = b.node("Less", [R, t], [f"{p}ltA"], f"{p}ltA")
    ge = b.node("Not", [lt], [f"{p}geA"], f"{p}geA")
    b.node("Where", [ge, "k32_i64", "zero_i64"], [f"{p}bbase"], f"{p}bbase")
    bcur = f"{p}bbase"
    for lvl, k in enumerate([16, 8, 4, 2, 1]):
        bk = b.node("Add", [bcur, f"k{k}_i64"], [f"{p}bk{lvl}"], f"{p}bk{lvl}")
        bku = b.node("Cast", [bk], [f"{p}bku{lvl}"], f"{p}bku{lvl}", to=UINT64)
        tu = b.node("BitShift", ["one_u64", bku], [f"{p}tu{lvl}"], f"{p}tu{lvl}", direction="LEFT")
        ti = b.node("Cast", [tu], [f"{p}ti{lvl}"], f"{p}ti{lvl}", to=INT64)
        lt = b.node("Less", [R, ti], [f"{p}lt{lvl}"], f"{p}lt{lvl}")
        ge = b.node("Not", [lt], [f"{p}ge{lvl}"], f"{p}ge{lvl}")
        w = b.node("Where", [ge, f"k{k}_i64", "zero_i64"], [f"{p}w{lvl}"], f"{p}w{lvl}")
        bcur = b.node("Add", [bcur, w], [f"{p}b{lvl}"], f"{p}b{lvl}")
    return b.node("Add", [bcur, "one_i64"], [f"{p}bitlen"], f"{p}bitlen")   # bitlen = b+1


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


# ---- LayerNorm (Q6): int-canonical, pure-int, mirrors int8_layernorm_intscale ----

_LN_K, _LN_R, _LN_G = 16, 20, 15   # mean/var Q16, rsqrt Q20, gamma/beta Q15 (match int8_compute)
_SQRT2_Q20 = 1482910               # round(sqrt(2) * 2^20)


def _floor_div_pos(b, a, d_pos, tag):
    """floor(a / d) for d > 0. ONNX Div truncates toward 0 (ceil for a < 0); ONNX Mod (default
    fmod=0) returns the EUCLIDEAN remainder (sign of the divisor, >= 0 for d > 0), so r < 0 is
    never true -- the correction must key off the NUMERATOR sign, not the remainder sign. Subtract
    1 when a < 0 and a is not an exact multiple of d (Euclidean r == 0 iff d | a). Mirrors _rhu."""
    b.init("zero_i64", np.array([0], dtype=np.int64), INT64)
    q = b.node("Div", [a, d_pos], [f"{tag}_q"], f"{tag}_div")
    r = b.node("Mod", [a, d_pos], [f"{tag}_r"], f"{tag}_mod")
    neg = b.node("Less", [a, "zero_i64"], [f"{tag}_neg"], f"{tag}_less")
    eq = b.node("Equal", [r, "zero_i64"], [f"{tag}_eq"], f"{tag}_eq")
    nz = b.node("Not", [eq], [f"{tag}_nz"], f"{tag}_nz")
    cb = b.node("And", [neg, nz], [f"{tag}_cb"], f"{tag}_and")
    corr = b.node("Cast", [cb], [f"{tag}_c"], f"{tag}_cast", to=INT64)
    return b.node("Sub", [q, corr], [f"{tag}_out"], f"{tag}_sub")


def _emit_layernorm_int8(b, x_int, x_zp, y_mul_in, y_shift_in, gamma_int, beta_int, D,
                         K=_LN_K, R=_LN_R, G=_LN_G, eps_q=100000):
    """Append the int-canonical LayerNorm (mirrors int8_compute.int8_layernorm_intscale) to
    builder `b`. Consumes the previous op's int8 output (x_int [B,D], x_zp [B,1], y_mul_in
    [B,1], y_shift_in [B,1]); emits (y_int8 [B,D], y_mul [B,1], y_shift [B,1], y_zp [B,1]).
    eps = 1/eps_q (whisper eps=1e-5 -> eps_q=100000). Returns the 4 output tensor names.

    Pure-int throughout: eps_K from the int act scale, pure-int CLZ rsqrt seed (no fp
    fallback), 4 Newton iters; the output reuses _emit_output_requant with F = K+R+G. All
    LN-specific node names are 'ln_'-prefixed to avoid SSA collisions with the requant's
    internal names; _clz_ladder is emitted with tag='ln_clz' (the requant emits it again with
    tag='clz'). See int8_compute.int8_layernorm_intscale for the bit-exact spec."""
    P = "ln_"
    b.init("zero_i64", np.array([0], dtype=np.int64), INT64)
    b.init("one_i64", np.array([1], dtype=np.int64), INT64)
    b.init("c2_i64", np.array([2], dtype=np.int64), INT64)
    b.init(f"{P}K_i64", np.array([K], dtype=np.int64), INT64)
    b.init(f"{P}R_i64", np.array([R], dtype=np.int64), INT64)
    b.init(f"{P}D_i64", np.array([D], dtype=np.int64), INT64)
    b.init("axes_neg1", np.array([-1], dtype=np.int64), INT64)
    for s in (K, K + R, R + 1):
        b.init(f"{P}sh{s}_u64", np.array([s], dtype=np.uint64), UINT64)
    # u = x_int - x_zp  [B,D] int64
    x64 = b.node("Cast", [x_int], [f"{P}x64"], f"{P}x_cast", to=INT64)
    xz64 = b.node("Cast", [x_zp], [f"{P}xz64"], f"{P}xz_cast", to=INT64)
    u = b.node("Sub", [x64, xz64], [f"{P}u"], f"{P}u_sub")                       # [B,D]
    # S1 = sum(u) ; S2 = sum(u*u)
    S1 = b.node("ReduceSum", [u, "axes_neg1"], [f"{P}S1"], f"{P}S1", keepdims=1)  # [B,1]
    u2 = b.node("Mul", [u, u], [f"{P}u2"], f"{P}u2_mul")
    S2 = b.node("ReduceSum", [u2, "axes_neg1"], [f"{P}S2"], f"{P}S2", keepdims=1)  # [B,1]
    # mean_K = floor_div(S1<<K, D) ; var_K = floor_div(S2<<K, D) - (mean_K^2 >> K)
    S1u = b.node("Cast", [S1], [f"{P}S1u"], f"{P}S1u_cast", to=UINT64)
    S1s = b.node("BitShift", [S1u, f"{P}sh{K}_u64"], [f"{P}S1s"], f"{P}S1s", direction="LEFT")
    S1s_i = b.node("Cast", [S1s], [f"{P}S1s_i"], f"{P}S1s_i", to=INT64)
    mean_K = _floor_div_pos(b, S1s_i, f"{P}D_i64", f"{P}mean")                  # floor((S1<<K)/D)
    S2u = b.node("Cast", [S2], [f"{P}S2u"], f"{P}S2u_cast", to=UINT64)
    S2s = b.node("BitShift", [S2u, f"{P}sh{K}_u64"], [f"{P}S2s"], f"{P}S2s", direction="LEFT")
    S2s_i = b.node("Cast", [S2s], [f"{P}S2s_i"], f"{P}S2s_i", to=INT64)
    var_hi = _floor_div_pos(b, S2s_i, f"{P}D_i64", f"{P}varhi")                 # floor((S2<<K)/D)
    mK2 = b.node("Mul", [mean_K, mean_K], [f"{P}mK2"], f"{P}mK2_mul")
    mK2u = b.node("Cast", [mK2], [f"{P}mK2u"], f"{P}mK2u_cast", to=UINT64)
    mK2r = b.node("BitShift", [mK2u, f"{P}sh{K}_u64"], [f"{P}mK2r"], f"{P}mK2r", direction="RIGHT")
    mK2r_i = b.node("Cast", [mK2r], [f"{P}mK2r_i"], f"{P}mK2r_i", to=INT64)
    var_K = b.node("Sub", [var_hi, mK2r_i], [f"{P}var_K"], f"{P}var_sub")        # [B,1]
    # eps_K = round_half_up(p*2^(K+2*y_shift) / (q*y_mul^2)); max(eps_K, 1)
    ys = b.node("Cast", [y_shift_in], [f"{P}ys"], f"{P}ys_cast", to=INT64)
    ym = b.node("Cast", [y_mul_in], [f"{P}ym"], f"{P}ym_cast", to=INT64)
    two_ys = b.node("Mul", [ys, "c2_i64"], [f"{P}two_ys"], f"{P}two_ys")           # 2*y_shift
    h = b.node("Add", [two_ys, f"{P}K_i64"], [f"{P}h"], f"{P}h_add")             # K + 2*y_shift [B,1]
    h_u = b.node("Cast", [h], [f"{P}h_u"], f"{P}h_cast", to=UINT64)
    num = b.node("BitShift", ["one_u64", h_u], [f"{P}eps_num"], f"{P}eps_num", direction="LEFT")
    num_i = b.node("Cast", [num], [f"{P}eps_num_i"], f"{P}eps_num_i", to=INT64)
    ym2 = b.node("Mul", [ym, ym], [f"{P}ym2"], f"{P}ym2_mul")
    b.init(f"{P}q_i64", np.array([eps_q], dtype=np.int64), INT64)
    den = b.node("Mul", [ym2, f"{P}q_i64"], [f"{P}eps_den"], f"{P}eps_den")     # q * y_mul^2
    eps_K = _rhu(b, num_i, den, f"{P}eps")                                       # round_half_up
    eps_K = b.node("Max", [eps_K, "one_i64"], [f"{P}eps_K"], f"{P}eps_max")     # >= 1
    # s_K = var_K + eps_K (clamp [1, 2^62))
    s_K = b.node("Add", [var_K, eps_K], [f"{P}s_K_raw"], f"{P}sK_add")
    s_K = b.node("Max", [s_K, "one_i64"], [f"{P}s_K1"], f"{P}sK_max1")
    b.init("cap62_i64", np.array([(1 << 62) - 1], dtype=np.int64), INT64)
    s_K = b.node("Min", [s_K, "cap62_i64"], [f"{P}s_K"], f"{P}sK_min")          # [B,1]
    # rsqrt seed via pure-int CLZ bitlen
    bitlen = _clz_ladder(b, s_K, tag=f"{P}clz")                                  # ln_clz_bitlen
    bitpos = b.node("Sub", [bitlen, "one_i64"], [f"{P}bitpos"], f"{P}bitpos")
    e = b.node("Sub", [f"{P}K_i64", bitpos], [f"{P}e"], f"{P}e")                 # K - bitpos
    odd = b.node("BitwiseAnd", [e, "one_i64"], [f"{P}odd"], f"{P}odd")          # e & 1
    e_minus_odd = b.node("Sub", [e, odd], [f"{P}e_mo"], f"{P}e_mo")
    half = b.node("Div", [e_minus_odd, "c2_i64"], [f"{P}half"], f"{P}half")     # (e-odd)//2 (even -> trunc==floor)
    a = b.node("Add", [f"{P}R_i64", half], [f"{P}a"], f"{P}a")                  # R + half [B,1]
    # C = where(odd, 8409*sqrt2_Q20, 8409*2^20)
    b.init(f"{P}Ceven", np.array([8409 * (1 << 20)], dtype=np.int64), INT64)
    b.init(f"{P}Codd", np.array([8409 * _SQRT2_Q20], dtype=np.int64), INT64)
    is_odd = b.node("Equal", [odd, "one_i64"], [f"{P}is_odd"], f"{P}is_odd")
    C = b.node("Where", [is_odd, f"{P}Codd", f"{P}Ceven"], [f"{P}C"], f"{P}C")   # [B,1]
    # seed = round_half_up(C<<a, 10000*2^20); clamp a>=0 (matches the reference's a.clamp(min=0);
    # a >= 12 always for |u| <= 255, but the clamp also guards the uint64 cast on negative a)
    a_c = b.node("Max", [a, "zero_i64"], [f"{P}a_c"], f"{P}a_clamp")
    a_u = b.node("Cast", [a_c], [f"{P}a_u"], f"{P}a_cast", to=UINT64)
    Cu = b.node("Cast", [C], [f"{P}Cu"], f"{P}C_cast", to=UINT64)
    Csh = b.node("BitShift", [Cu, a_u], [f"{P}Csh"], f"{P}Csh", direction="LEFT")
    Csh_i = b.node("Cast", [Csh], [f"{P}Csh_i"], f"{P}Csh_i", to=INT64)
    b.init(f"{P}den_seed", np.array([10000 * (1 << 20)], dtype=np.int64), INT64)
    r = _rhu(b, Csh_i, f"{P}den_seed", f"{P}seed")                              # seed [B,1]
    # 4 Newton iters: prod = (s_K*r)*r ; t = prod>>(K+R) ; r = (r*(3*2^R - t) + 2^(R-1))>>(R+1)
    b.init(f"{P}three", np.array([3 * (1 << R)], dtype=np.int64), INT64)
    b.init(f"{P}half_r", np.array([1 << (R - 1)], dtype=np.int64), INT64)
    for i in range(4):
        sr = b.node("Mul", [s_K, r], [f"{P}sr{i}"], f"{P}sr{i}")
        prod = b.node("Mul", [sr, r], [f"{P}prod{i}"], f"{P}prod{i}")            # (s_K*r)*r
        produ = b.node("Cast", [prod], [f"{P}produ{i}"], f"{P}produ{i}", to=UINT64)
        tu = b.node("BitShift", [produ, f"{P}sh{K+R}_u64"], [f"{P}t{i}u"], f"{P}t{i}u", direction="RIGHT")
        t = b.node("Cast", [tu], [f"{P}t{i}"], f"{P}t{i}", to=INT64)
        tmt = b.node("Sub", [f"{P}three", t], [f"{P}3mt{i}"], f"{P}3mt{i}")     # 3*2^R - t
        rt = b.node("Mul", [r, tmt], [f"{P}rt{i}"], f"{P}rt{i}")
        rth = b.node("Add", [rt, f"{P}half_r"], [f"{P}rth{i}"], f"{P}rth{i}")
        rthu = b.node("Cast", [rth], [f"{P}rth{i}u"], f"{P}rth{i}u", to=UINT64)
        rsh = b.node("BitShift", [rthu, f"{P}sh{R+1}_u64"], [f"{P}r{i}"], f"{P}r{i}", direction="RIGHT")
        r = b.node("Cast", [rsh], [f"{P}r{i}_i"], f"{P}r{i}_i", to=INT64)
    r_final = r
    # uK = (u<<K) - mean_K ; y_int = uK*r*gamma_int + (beta_int << (K+R))
    uu = b.node("Cast", [u], [f"{P}uu"], f"{P}uu_cast", to=UINT64)
    ush = b.node("BitShift", [uu, f"{P}sh{K}_u64"], [f"{P}ush"], f"{P}ush", direction="LEFT")
    ush_i = b.node("Cast", [ush], [f"{P}ush_i"], f"{P}ush_i", to=INT64)
    uK = b.node("Sub", [ush_i, mean_K], [f"{P}uK"], f"{P}uK")                   # [B,D] QK
    uKr = b.node("Mul", [uK, r_final], [f"{P}uKr"], f"{P}uKr")                  # [B,D] * [B,1]
    b.init(f"{P}gamma_int", gamma_int.reshape(1, D), INT64)                      # [1,D] QG (baked)
    uKrg = b.node("Mul", [uKr, f"{P}gamma_int"], [f"{P}uKrg"], f"{P}uKrg")      # [B,D] Q(K+R+G)
    b.init(f"{P}beta_int", beta_int.reshape(1, D), INT64)                        # [1,D] QG (baked)
    bu = b.node("Cast", [f"{P}beta_int"], [f"{P}bu"], f"{P}bu_cast", to=UINT64)
    bsh = b.node("BitShift", [bu, f"{P}sh{K+R}_u64"], [f"{P}bsh"], f"{P}bsh", direction="LEFT")
    bsh_i = b.node("Cast", [bsh], [f"{P}bsh_i"], f"{P}bsh_i", to=INT64)          # [1,D] Q(K+R+G)
    y_int_ln = b.node("Add", [uKrg, bsh_i], [f"{P}y_int"], f"{P}y_int")         # [B,D] Q(K+R+G)
    # output requant: reuse Stage 1b with F = K+R+G, fresh per-token scale (act_mul=1, act_shift=0)
    b.init(f"{P}am_one", np.array([1], dtype=np.int32), INT32)
    b.init(f"{P}ash_zero", np.array([0], dtype=np.int32), INT32)
    y_out, y_mul, y_shift, y_zp = _emit_output_requant(
        b, y_int_ln, f"{P}am_one", f"{P}ash_zero", K + R + G, None, D)
    return y_out, y_mul, y_shift, y_zp


def build_int8_layernorm_onnx(ln_module, model_name="int8_layernorm", emit_intermediates=False,
                               eps_q=100000):
    """Build the int-canonical LayerNorm ONNX (mirrors int8_compute.int8_layernorm_intscale).

    Inputs (the previous op's int8 output, zero runtime fp):
      x_int [B, D] int8, x_zp [B, 1] int32, y_mul [B, 1] int32 (Q1.16 input scale),
      y_shift [B, 1] int32.
    Outputs: y_int8 [B, D] int8, y_mul [B, 1] int32, y_shift [B, 1] int32, y_zp [B, 1] int32.
    `ln_module` is a torch nn.LayerNorm (gamma = .weight, beta = .bias, eps = .eps,
    D = .normalized_shape[0]); gamma_int/beta_int are baked as int Q15 initializers.
    `eps_q` is 1/eps (whisper eps=1e-5 -> 100000). With emit_intermediates the graph also
    outputs the int LN intermediates (u, S1, S2, mean_K, var_K, eps_K, s_K, r, y_int) for
    bit-exact verification vs the torch reference.
    """
    D = int(ln_module.normalized_shape[0])
    gamma = ln_module.weight.detach().to(torch.float64).cpu()
    beta = ln_module.bias.detach().to(torch.float64).cpu()
    gamma_int = np.floor(gamma.numpy() * (2.0 ** _LN_G) + 0.5).astype(np.int64)
    beta_int = np.floor(beta.numpy() * (2.0 ** _LN_G) + 0.5).astype(np.int64)
    b = _Builder()
    # inputs are named x_mul/x_shift (the previous op's OUTPUT scale = this LN's input scale);
    # NOT y_mul/y_shift, which collide with the requant's same-named outputs (SSA).
    out_names = _emit_layernorm_int8(b, "x_int", "x_zp", "x_mul", "x_shift", gamma_int, beta_int, D,
                                      eps_q=eps_q)
    y_out, y_mul, y_shift, y_zp = out_names
    inputs = [
        helper.make_tensor_value_info("x_int", INT8, ["B", D]),
        helper.make_tensor_value_info("x_zp", INT32, ["B", 1]),
        helper.make_tensor_value_info("x_mul", INT32, ["B", 1]),
        helper.make_tensor_value_info("x_shift", INT32, ["B", 1]),
    ]
    vi = helper.make_tensor_value_info
    outputs = [vi("y_int8", INT8, ["B", D]), vi("y_mul", INT32, ["B", 1]),
               vi("y_shift", INT32, ["B", 1]), vi("y_zp", INT32, ["B", 1])]
    if emit_intermediates:
        outputs += [vi("ln_u", INT64, ["B", D]), vi("ln_S1", INT64, ["B", 1]),
                    vi("ln_S2", INT64, ["B", 1]), vi("ln_mean_out", INT64, ["B", 1]),
                    vi("ln_var_K", INT64, ["B", 1]), vi("ln_eps_K", INT64, ["B", 1]),
                    vi("ln_s_K", INT64, ["B", 1]), vi("ln_r3_i", INT64, ["B", 1]),
                    vi("ln_y_int", INT64, ["B", D])]
    g_ = helper.make_graph(b.nodes, model_name, inputs, outputs, initializer=b.inits)
    m = helper.make_model(g_, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


# ---- recursive zero-fp audit (Q4) ----
# Tensor elem types that are legal in a zero-fp graph: integers, bool, string. Any float
# (fp32/fp16/bfloat16/float8*) is a violation. Unknown/UNDEFINED types fail closed.
_ALLOWED_TYPES = {
    T.INT8, T.INT16, T.INT32, T.INT64, T.UINT8, T.UINT16, T.UINT32, T.UINT64, T.BOOL, T.STRING,
}
_FP_TYPES = {
    T.FLOAT, T.FLOAT16, T.BFLOAT16, T.FLOAT8E4M3FN, T.FLOAT8E5M2,
    T.FLOAT8E4M3FNUZ, T.FLOAT8E5M2FNUZ,
}
# Ops that are int-typed on the boundary but compute in fp internally, or are fp-only. The
# tensor-type walk catches most fp (fp initializers, fp Casts, fp-typed ops); this denylist is
# belt-and-suspenders for ops whose fp is hidden (DynamicQuantizeLinear's fp scale, Round's
# fp-only domain, the com.microsoft QLinear* ops) and the standard fp nonlinear ops the
# consensus said to avoid (replaced by int LUT decompositions).
_FP_DENYLIST = {
    "DynamicQuantizeLinear", "QLinearMatMul", "QLinearConv", "QLinearSoftmax",
    "QLinearAdd", "QLinearMul", "Round",
    "Erf", "Sqrt", "Pow", "Exp", "Log", "Reciprocal",
    "Softmax", "SoftmaxGrad", "LayerNormalization", "Gelu", "Gemm",
}
_TYPE_NAMES = {
    getattr(T, n): n for n in
    ("UNDEFINED", "FLOAT", "UINT8", "INT8", "UINT16", "INT16", "INT32", "INT64", "STRING",
     "BOOL", "FLOAT16", "DOUBLE", "UINT32", "UINT64", "COMPLEX64", "COMPLEX128", "BFLOAT16",
     "FLOAT8E4M3FN", "FLOAT8E5M2", "FLOAT8E4M3FNUZ", "FLOAT8E5M2FNUZ")
}


def _type_name(t):
    return _TYPE_NAMES.get(t, f"UNKNOWN({t})")


def _check_elem_type(t, where, violations):
    if t in _FP_TYPES:
        violations.append(f"{where}: float type {_type_name(t)}")
    elif t not in _ALLOWED_TYPES:
        violations.append(f"{where}: non-allowed type {_type_name(t)} (fail closed)")


def _audit_tensor_proto(t, where, violations):
    _check_elem_type(t.data_type, where, violations)


def _audit_value_info(vi, where, violations):
    tt = vi.type.tensor_type
    if tt.elem_type == 0 and (tt.HasField("shape") or vi.type.HasField("sequence_type")
                              or vi.type.HasField("map_type") or vi.type.HasField("optional_type")):
        # non-tensor (sequence/map/optional) -- recurse into the leaf elem type if present.
        violations.append(f"{where}: non-tensor value-info (sequence/map/optional) not supported "
                          f"(fail closed)")
        return
    _check_elem_type(tt.elem_type, where, violations)


def _audit_attrs(node, node_loc, violations):
    """Recurse into a node's attributes: flag fp scalar attrs (FLOAT/FLOATS), fp tensor attrs
    (value/sparse_value of a Constant), and nested graph-valued attrs (then_branch/else_branch/
    body/sub_graph/...)."""
    from onnx import AttributeProto as A
    for attr in node.attribute:
        an = attr.name
        if attr.type == A.FLOAT:
            violations.append(f"{node_loc}: attribute {an} is FLOAT (scalar)")
        elif attr.type == A.FLOATS:
            violations.append(f"{node_loc}: attribute {an} is FLOATS (list)")
        elif attr.type == A.TENSOR:
            _audit_tensor_proto(attr.t, f"{node_loc}: attribute {an} (Constant value)", violations)
        elif attr.type == A.SPARSE_TENSOR:
            _audit_tensor_proto(attr.sparse_tensor.values,
                                f"{node_loc}: attribute {an} (sparse values)", violations)
        elif attr.type == A.GRAPH:
            _audit_graph(attr.g, f"{node_loc}:{an}", violations)
        elif attr.type == A.GRAPHS:
            for i, sg in enumerate(attr.graphs):
                _audit_graph(sg, f"{node_loc}:{an}[{i}]", violations)


def _audit_graph(graph, where, violations):
    for vi in graph.input:
        _audit_value_info(vi, f"{where}:input '{vi.name}'", violations)
    for vi in graph.output:
        _audit_value_info(vi, f"{where}:output '{vi.name}'", violations)
    for vi in graph.value_info:
        _audit_value_info(vi, f"{where}:value_info '{vi.name}'", violations)
    for init in graph.initializer:
        _audit_tensor_proto(init, f"{where}:initializer '{init.name}'", violations)
    for node in graph.node:
        nloc = f"{where}:node '{node.name or node.op_type}'({node.op_type})"
        if node.op_type in _FP_DENYLIST:
            violations.append(f"{nloc}: forbidden op '{node.op_type}' (fp-computing or fp-only)")
        if node.op_type == "Cast":
            to = None
            for attr in node.attribute:
                if attr.name == "to":
                    to = attr.i
            if to is not None and to in _FP_TYPES:
                violations.append(f"{nloc}: Cast to {_type_name(to)} (fp)")
            elif to is not None and to not in _ALLOWED_TYPES:
                violations.append(f"{nloc}: Cast to {_type_name(to)} (non-allowed, fail closed)")
        _audit_attrs(node, nloc, violations)


def zero_fp_audit(model, check_optimized=False, optimized_path=None):
    """Recursive zero-fp audit (Q4). Assert the graph (and every nested graph-valued attr:
    then_branch/else_branch/body/sub_graph/...) has NO float (fp32/fp16/bfloat16/float8*) in any
    input/output/value_info/initializer/Constant-attr, no fp Cast, and no fp-computing op from
    the denylist. FAIL CLOSED on unknown/UNDEFINED types. With check_optimized=True, ALSO audit
    the ORT-optimized artifact (ORT_ENABLE_ALL) -- ORT can inject fp Casts/fuse LN at session
    load. Returns None on success; raises AssertionError listing every violation on failure.

    The raw graph audit catches the export's own fp. The optimized audit catches ORT-injected
    fp; ORT 1.19.2 CPU adds none for the int-only linear (verified), but the decoder's
    LayerNormalization/GELU/Softmax may trigger fp fusion at load, so Phase C validates the
    merged-decoder under ORT_DISABLE_ALL as well."""
    if isinstance(model, (str, bytes)):
        m = onnx.load(model)
        path = model if isinstance(model, str) else None
    else:
        m = model
        path = None
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception:
        pass  # shape inference is best-effort; the type walk does not depend on it
    violations = []
    _audit_graph(m.graph, "graph", violations)
    if violations:
        raise AssertionError("zero-fp audit FAILED:\n  " + "\n  ".join(violations))
    if check_optimized:
        _audit_optimized(m if path is None else path, optimized_path)
    return None


def _audit_optimized(model_or_path, optimized_path=None):
    """Audit the ORT-optimized graph. ORT_ENABLE_ALL writes the optimized model to disk; audit
    that artifact recursively. ORT may inject fp Casts or fuse LayerNormalization/GELU into fp
    subgraphs at load, so the optimized artifact must be audited separately from the raw export."""
    import os
    import tempfile
    if optimized_path is None:
        fd, optimized_path = tempfile.mkstemp(suffix="_ort_optimized.onnx")
        os.close(fd)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = optimized_path
    path = model_or_path if isinstance(model_or_path, str) else _save_tmp(model_or_path)
    ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    mopt = onnx.load(optimized_path)
    try:
        mopt = onnx.shape_inference.infer_shapes(mopt)
    except Exception:
        pass
    violations = []
    _audit_graph(mopt.graph, "optimized", violations)
    if violations:
        raise AssertionError("zero-fp audit FAILED on the ORT-optimized artifact:\n  "
                             + "\n  ".join(violations))
    return None


def _save_tmp(model):
    import os
    import tempfile
    fd, p = tempfile.mkstemp(suffix="_audit.onnx")
    os.close(fd)
    onnx.save(model, p)
    return p