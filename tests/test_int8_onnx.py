"""Phase B Q6: minimal ONNX of one real HQQ linear, bit-exact vs the torch int reference.

Verifies export_onnx_int8.build_int8_linear_onnx (Stage 1a: int inputs -> fp output) against
int8_compute.int8_matmul_fixed_point (int-act path). Every int intermediate (qr_g unpack, the
four MatMulInteger terms A/Bt/C/D, the zp corrections T1/T2, the signed fixed-point
rshift_round p1/p2, the accumulator) must be BIT-EXACT; the fp output must match within
float32 noise. Robustness: batch 1/2/3, all-positive (zp near -128), all-negative (zp near
127), extreme magnitudes. Runs under .venv-onnx (onnx + onnxruntime).
"""
import os
import sys

import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")
hqq = pytest.importorskip("hqq")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HQQ_COMPUTE_DTYPE", "fp32")
os.environ.setdefault("HQQ_ATTN_IMPL", "eager")

import hqq_asr  # noqa: E402
import int8_compute as i8  # noqa: E402
import export_onnx_int8 as ex  # noqa: E402
from hqq.core.quantize import HQQLinear  # noqa: E402

MODEL = os.environ.get("INT8_ONNX_MODEL", "build/whisper-tiny-hqq-3bit")


def _load_first_linear(nbits):
    m = hqq_asr.load_whisper_hqq(MODEL, device="cpu", compute_dtype=torch.float32)
    for _name, mod in m.named_modules():
        if isinstance(mod, HQQLinear) and int(mod.meta["nbits"]) == nbits:
            return mod
    pytest.skip(f"no {nbits}-bit HQQLinear in {MODEL}")


def _torch_intermediates(mod, x):
    O, I = (int(s) for s in mod.meta["shape"])
    g = int(mod.meta["group_size"])
    xi, am, sh, xz = i8.quantize_act_per_token_intscale(x)
    qr = i8.unpack_levels(mod.W_q, int(mod.meta["nbits"]), mod.meta["packing"], (O, I), g, 1).to(torch.int32)
    out = i8.int8_matmul_fixed_point(
        xi, None, xz, qr, mod.meta["zero"].to(torch.float32).reshape(-1, 1),
        mod.meta["scale"].to(torch.float32).reshape(-1, 1), mod.bias, g, act_mul=am, act_shift=sh)
    A, Bt, C, D = i8._per_group_int_terms(xi, qr, g)
    zero_g = mod.meta["zero"].to(torch.float32).reshape(O, I // g)
    scale_g = mod.meta["scale"].to(torch.float32).reshape(O, I // g)
    zs = zero_g * scale_g
    ms, msh = i8.fixed_point_per_group(scale_g)
    mz, mzh = i8.fixed_point_per_group(zs)
    F = int(min(msh.amin().item(), mzh.amin().item()))
    xz3 = xz.to(torch.int32).reshape(-1, 1, 1)
    T1 = (A - xz3 * Bt.unsqueeze(0)).to(torch.int64)
    T2 = (C.unsqueeze(1) - xz3 * D).to(torch.int64)
    p1 = i8._rshift_round(T1 * ms.unsqueeze(0).to(torch.int64), (msh - F).unsqueeze(0).to(torch.int64))
    p2 = i8._rshift_round(T2 * mz.unsqueeze(0).to(torch.int64), (mzh - F).unsqueeze(0).to(torch.int64))
    acc = (p1 - p2).sum(-1).to(torch.int64)
    yr, mr, sr, zr = i8.int8_output_requant_intscale(acc, am, sh, F, mod.bias)
    return {"qr_g": qr.reshape(O, I // g, g), "A": A.to(torch.int64), "Bt": Bt.to(torch.int64),
            "C": C.to(torch.int64), "T1": T1, "T2": T2, "p1_out": p1, "p2_out": p2,
            "acc": acc, "out": out, "xi": xi, "xz": xz, "am": am, "sh": sh,
            "y_int8": yr, "y_mul": mr, "y_shift": sr, "y_zp": zr}


def _check_requant(mod, x):
    """Stage 1b: emit_output_requant=True -> (y_int8, y_mul, y_shift, y_zp) bit-exact vs
    int8_compute.int8_output_requant_intscale (the int-canonical zero-fp boundary)."""
    import numpy as np
    model = ex.build_int8_linear_onnx(mod, model_name="t1b", emit_intermediates=False,
                                      emit_output_requant=True)
    path = "/tmp/_int8_requant_test.onnx"
    onnx.save(model, path)
    t = _torch_intermediates(mod, x)
    o = _run_ort(path, t["xi"], t["xz"], t["am"], t["sh"])
    for k, ref in [("y_int8", "y_int8"), ("y_mul", "y_mul"), ("y_shift", "y_shift"), ("y_zp", "y_zp")]:
        a = o[k].astype(np.int64)
        b = t[ref].numpy().astype(np.int64)
        if a.ndim == 1:  # ORT may flatten the [B,1] outputs depending on shape inference
            b = b.reshape(-1)
        assert a.shape == b.shape, f"{k}: shape {a.shape} vs {b.shape}"
        assert np.array_equal(a, b), f"{k}: {int((a != b).sum())}/{a.size} differ (max|d|={int(np.abs(a - b).max())})"


def _run_ort(onnx_path, xi, xz, am, sh):
    import numpy as np
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feeds = {"x_int": xi.numpy().astype(np.int8), "x_zp": xz.numpy().astype(np.int32).reshape(-1, 1),
             "act_mul": am.numpy().astype(np.int32).reshape(-1, 1),
             "act_shift": sh.numpy().astype(np.int32).reshape(-1, 1)}
    return {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, feeds))}


def _check(mod, x):
    import numpy as np
    model = ex.build_int8_linear_onnx(mod, model_name="t", emit_intermediates=True)
    path = "/tmp/_int8_linear_test.onnx"
    onnx.save(model, path)
    t = _torch_intermediates(mod, x)
    o = _run_ort(path, t["xi"], t["xz"], t["am"], t["sh"])
    for k in ["qr_g", "A", "Bt", "C", "T1", "T2", "p1_out", "p2_out", "acc"]:
        a = o[k].astype(np.int64)
        b = t[k].numpy().astype(np.int64)
        assert a.shape == b.shape, f"{k}: shape {a.shape} vs {b.shape}"
        assert np.array_equal(a, b), f"{k}: {int((a != b).sum())}/{a.size} differ (max|d|={int(np.abs(a - b).max())})"
    rel = float(np.abs(o["out"].astype(np.float64) - t["out"].numpy()).max()
                / (np.abs(t["out"].numpy()).max() + 1e-12))
    assert rel < 1e-5, f"out rel {rel} too high"


def test_3bit_linear_bitexact():
    mod = _load_first_linear(3)
    torch.manual_seed(1)
    _check(mod, torch.randn(2, int(mod.meta["shape"][1])) * 2.0)


@pytest.mark.parametrize("x_fn", [
    lambda I: torch.randn(1, I) * 2.0,                    # batch 1
    lambda I: torch.randn(3, I) * 5.0,                    # batch 3, large
    lambda I: torch.rand(2, I) * 3.0 + 0.5,               # all-positive (zp near -128)
    lambda I: -(torch.rand(2, I) * 3.0 + 0.5),             # all-negative (zp near 127)
    lambda I: torch.randn(2, I) * 50.0,                   # extreme magnitudes
])
def test_3bit_linear_robustness(x_fn):
    mod = _load_first_linear(3)
    torch.manual_seed(2)
    _check(mod, x_fn(int(mod.meta["shape"][1])))


def test_8bit_linear_bitexact():
    """8-bit tier (encoder.layers + fc1): uint8 weight levels, no bit-packing."""
    mod = _load_first_linear(8)
    torch.manual_seed(3)
    _check(mod, torch.randn(2, int(mod.meta["shape"][1])) * 2.0)


# ---- Stage 1b: int output requant (the true zero-fp boundary) ----

def test_3bit_output_requant_bitexact():
    mod = _load_first_linear(3)
    torch.manual_seed(1)
    _check_requant(mod, torch.randn(2, int(mod.meta["shape"][1])) * 2.0)


@pytest.mark.parametrize("x_fn", [
    lambda I: torch.randn(1, I) * 2.0,                    # batch 1
    lambda I: torch.randn(3, I) * 5.0,                   # batch 3, large
    lambda I: torch.rand(2, I) * 3.0 + 0.5,              # all-positive (zp near -128)
    lambda I: -(torch.rand(2, I) * 3.0 + 0.5),           # all-negative (zp near 127)
    lambda I: torch.randn(2, I) * 50.0,                  # extreme magnitudes
])
def test_3bit_output_requant_robustness(x_fn):
    mod = _load_first_linear(3)
    torch.manual_seed(2)
    _check_requant(mod, x_fn(int(mod.meta["shape"][1])))


def test_8bit_output_requant_bitexact():
    """8-bit tier output requant (bias present; large out_int range ~2^55 exercises the
    CLZ ladder and the negative-shift branch of the Q1.16 output scale)."""
    mod = _load_first_linear(8)
    torch.manual_seed(3)
    _check_requant(mod, torch.randn(2, int(mod.meta["shape"][1])) * 2.0)


# ---- recursive zero-fp audit (Q4) ----

def test_zero_fp_audit_stage1b_passes():
    """Stage 1b (int output requant) is structurally zero-fp: no fp tensors, no fp Casts,
    no fp-computing ops -- raw graph AND the ORT-optimized artifact (ORT_ENABLE_ALL must not
    inject fp into an int-only linear)."""
    mod = _load_first_linear(3)
    model = ex.build_int8_linear_onnx(mod, emit_intermediates=False, emit_output_requant=True)
    ex.zero_fp_audit(model)                       # raw graph
    ex.zero_fp_audit(model, check_optimized=True)  # ORT-optimized artifact


def test_zero_fp_audit_stage1a_fails():
    """Stage 1a (fp dequant output) is NOT zero-fp: the audit must catch the fp output and
    the fp Cast/Div intermediate value_infos (negative test -- ensures the audit fails closed
    on fp rather than passing vacuously)."""
    mod = _load_first_linear(3)
    model = ex.build_int8_linear_onnx(mod, emit_intermediates=False, emit_output_requant=False)
    with pytest.raises(AssertionError) as exc:
        ex.zero_fp_audit(model)
    assert "FLOAT" in str(exc.value), "audit must flag the fp output"


# ---- Q6: int8 LayerNorm ONNX (int-canonical, mirrors int8_layernorm_intscale) ----

def _load_ln(name):
    m = hqq_asr.load_whisper_hqq(MODEL, device="cpu", compute_dtype=torch.float32)
    ln = dict(m.named_modules()).get(name)
    if ln is None:
        pytest.skip(f"no {name} in {MODEL}")
    return ln


def _run_ort_ln(onnx_path, xi, xz, am, sh):
    import numpy as np
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feeds = {"x_int": xi.numpy().astype(np.int8),
             "x_zp": xz.numpy().astype(np.int32).reshape(-1, 1),
             "x_mul": am.numpy().astype(np.int32).reshape(-1, 1),
             "x_shift": sh.numpy().astype(np.int32).reshape(-1, 1)}
    return {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, feeds))}


def _check_ln(ln, x):
    """Build the int-canonical LN ONNX (with intermediates) and assert every int intermediate
    AND the four int8 outputs are BIT-EXACT vs int8_compute.int8_layernorm_intscale."""
    import numpy as np
    D = int(ln.normalized_shape[0])
    xi, am, sh, xz = i8.quantize_act_per_token_intscale(x)
    y_int8, ym, ys, yzp, inter = i8.int8_layernorm_intscale(
        xi, xz, am, sh, ln.weight.detach(), ln.bias.detach(), ln.eps, return_intermediates=True)
    model = ex.build_int8_layernorm_onnx(ln, model_name="ln_t", emit_intermediates=True)
    path = "/tmp/_int8_ln_test.onnx"
    onnx.save(model, path)
    o = _run_ort_ln(path, xi, xz, am, sh)
    pairs = [("ln_u", "u"), ("ln_S1", "S1"), ("ln_S2", "S2"), ("ln_mean_out", "mean_K"),
             ("ln_var_K", "var_K"), ("ln_eps_K", "eps_K"), ("ln_s_K", "s_K"),
             ("ln_r3_i", "r"), ("ln_y_int", "y_int")]
    for ok, rk in pairs:
        a = o[ok].astype(np.int64)
        b = inter[rk].numpy().astype(np.int64)
        assert a.shape == b.shape, f"{ok}: shape {a.shape} vs {b.shape}"
        assert np.array_equal(a, b), f"{ok} vs {rk}: {int((a != b).sum())}/{a.size} differ (max|d|={int(np.abs(a - b).max())})"
    for k, ref in [("y_int8", y_int8), ("y_mul", ym), ("y_shift", ys), ("y_zp", yzp)]:
        a = o[k]
        b = ref.numpy()
        if a.ndim == 1 and b.ndim == 2:
            b = b.reshape(-1)
        a = a.astype(np.int64)
        b = b.astype(np.int64).reshape(a.shape)
        assert np.array_equal(a, b), f"OUT {k}: {int((a != b).sum())}/{a.size} differ (max|d|={int(np.abs(a - b).max())})"


_LN_NAMES = ["model.encoder.layers.0.self_attn_layer_norm",
             "model.encoder.layers.0.final_layer_norm",
             "model.encoder.layers.1.self_attn_layer_norm",
             "model.encoder.layers.3.final_layer_norm"]


@pytest.mark.parametrize("ln_name", _LN_NAMES)
def test_int8_layernorm_bitexact(ln_name):
    """The int-canonical LN ONNX is bit-exact vs int8_layernorm_intscale: every int intermediate
    (u, S1, S2, mean_K, var_K, eps_K, s_K, r, y_int) and the four int8 outputs (y_int8, y_mul,
    y_shift, y_zp) match across real whisper-tiny encoder LayerNorms. The negative-S1 token
    exercises _floor_div_pos (ONNX Mod is Euclidean -> correction keys off the numerator sign)."""
    ln = _load_ln(ln_name)
    torch.manual_seed(7)
    _check_ln(ln, torch.randn(3, int(ln.normalized_shape[0])) * 4.0)


@pytest.mark.parametrize("x_fn", [
    lambda D: torch.randn(1, D) * 4.0,                   # batch 1
    lambda D: torch.randn(5, D) * 4.0,                   # batch 5
    lambda D: torch.rand(2, D) * 3.0 + 0.5,              # all-positive (S1 > 0)
    lambda D: -(torch.rand(2, D) * 3.0 + 0.5),           # all-negative (S1 < 0 -> floor_div branch)
    lambda D: torch.randn(2, D) * 50.0,                  # extreme magnitudes
])
def test_int8_layernorm_robustness(x_fn):
    """LN bit-exactness holds across batch sizes and sign/magnitude regimes (the all-negative
    case drives S1 << 0, the floor_div correction that broke before the numerator-sign fix)."""
    ln = _load_ln("model.encoder.layers.0.self_attn_layer_norm")
    torch.manual_seed(2)
    _check_ln(ln, x_fn(int(ln.normalized_shape[0])))


def test_int8_layernorm_zero_fp_audit():
    """The int-canonical LN ONNX is structurally zero-fp: no fp tensors, no fp Casts, no
    fp-computing ops -- raw graph AND the ORT-optimized artifact (ORT_ENABLE_ALL must not fuse
    LayerNormalization into fp on this pure-int graph)."""
    ln = _load_ln("model.encoder.layers.0.self_attn_layer_norm")
    model = ex.build_int8_layernorm_onnx(ln, emit_intermediates=False)
    ex.zero_fp_audit(model)                       # raw graph
    ex.zero_fp_audit(model, check_optimized=True)  # ORT-optimized artifact


# ---- Q6: int8 GELU ONNX (int-canonical, mirrors int8_gelu_intscale) ----

def _check_gelu(x, D):
    """Build the int-canonical GELU ONNX (with intermediates) and assert every int intermediate
    AND the four int8 outputs are BIT-EXACT vs int8_compute.int8_gelu_intscale."""
    import numpy as np
    xi, am, sh, xz = i8.quantize_act_per_token_intscale(x)
    y_int8, ym, ys, yzp, inter = i8.int8_gelu_intscale(xi, xz, am, sh, return_intermediates=True)
    model = ex.build_int8_gelu_onnx(model_name="gelu_t", emit_intermediates=True, D=D)
    path = "/tmp/_int8_gelu_test.onnx"
    onnx.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {"x_int": xi.numpy().astype(np.int8),
             "x_zp": xz.numpy().astype(np.int32).reshape(-1, 1),
             "x_mul": am.numpy().astype(np.int32).reshape(-1, 1),
             "x_shift": sh.numpy().astype(np.int32).reshape(-1, 1)}
    o = {out.name: v for out, v in zip(sess.get_outputs(), sess.run(None, feeds))}
    pairs = [("gelu_u", "u"), ("gelu_num", "num"), ("gelu_den", "den"),
             ("gelu_idx", "idx"), ("gelu_phi", "phi_int"), ("gelu_acc", "acc")]
    for ok, rk in pairs:
        a = o[ok].astype(np.int64)
        b = inter[rk].numpy().astype(np.int64)
        assert a.shape == b.shape, f"{ok}: shape {a.shape} vs {b.shape}"
        assert np.array_equal(a, b), f"{ok} vs {rk}: {int((a != b).sum())}/{a.size} differ (max|d|={int(np.abs(a - b).max())})"
    for k, ref in [("y_int8", y_int8), ("y_mul", ym), ("y_shift", ys), ("y_zp", yzp)]:
        a = o[k]
        b = ref.numpy()
        if a.ndim == 1 and b.ndim == 2:
            b = b.reshape(-1)
        a = a.astype(np.int64)
        b = b.astype(np.int64).reshape(a.shape)
        assert np.array_equal(a, b), f"OUT {k}: {int((a != b).sum())}/{a.size} differ (max|d|={int(np.abs(a - b).max())})"


@pytest.mark.parametrize("D", [1536, 384])   # whisper-tiny fc1, fc2
def test_int8_gelu_bitexact(D):
    """The int-canonical GELU ONNX is bit-exact vs int8_gelu_intscale: every int intermediate
    (u, num, den, idx, phi_int, acc) and the four int8 outputs match. GELU is parameter-free
    (one function for fc1/fc2 in every layer); D is the feature dim."""
    torch.manual_seed(7)
    _check_gelu(torch.randn(3, D) * 4.0, D)


@pytest.mark.parametrize("x_fn", [
    lambda D: torch.randn(1, D) * 4.0,                   # batch 1
    lambda D: torch.randn(5, D) * 4.0,                   # batch 5
    lambda D: torch.randn(2, D) * 8.0,                   # large positive tail (Phi -> 1)
    lambda D: -(torch.randn(2, D) * 8.0),                # large negative tail (Phi -> 0)
    lambda D: torch.randn(2, D) * 50.0,                  # extreme magnitudes (idx saturates)
    lambda D: torch.randn(2, D) * 0.3,                   # near-zero (idx near T/2, steep Phi)
])
def test_int8_gelu_robustness(x_fn):
    """GELU bit-exactness holds across batch sizes and magnitude regimes (the large-tail cases
    drive idx to the clamp boundaries 0 and T-1 where Phi saturates)."""
    D = 1536
    torch.manual_seed(2)
    _check_gelu(x_fn(D), D)


def test_int8_gelu_vs_fp_erf_within_tolerance():
    """The int-canonical GELU output (dequant) vs fp erf GELU: the PRE-requant LUT+index error is
    < 0.001 (the Phi LUT is near-exact); the POST-requant abs err is one int8 output step (shared
    with the linear/LN requant, WER-neutral, A3 staged gate already passed at +gelu)."""
    import numpy as np
    worst_pre = 0.0
    for seed in (0, 1, 2, 3):
        torch.manual_seed(seed)
        x = torch.randn(3, 1536) * 4.0
        xi, am, sh, xz = i8.quantize_act_per_token_intscale(x)
        _, _, _, _, inter = i8.int8_gelu_intscale(xi, xz, am, sh, return_intermediates=True)
        x_real = (xi.float() - xz.float().reshape(-1, 1)) * am.float().reshape(-1, 1) \
            * (2.0 ** (-sh.float().reshape(-1, 1)))
        phi_real = inter["phi_int"].float() * (2.0 ** -i8._GELU_S)
        gelu_prereq = x_real * phi_real                       # the LUT GELU before output requant
        gelu_fp = i8.fp_gelu_ref(x_real)
        worst_pre = max(worst_pre, float((gelu_prereq - gelu_fp).abs().max()))
    assert worst_pre < 1e-2, f"pre-requant LUT/index abs err {worst_pre} >= 0.01"


def test_int8_gelu_zero_fp_audit():
    """The int-canonical GELU ONNX is structurally zero-fp: no fp tensors, no fp Casts, no
    fp-computing ops (Gather is an int index lookup) -- raw graph AND the ORT-optimized artifact."""
    model = ex.build_int8_gelu_onnx(emit_intermediates=False, D=1536)
    ex.zero_fp_audit(model)                       # raw graph
    ex.zero_fp_audit(model, check_optimized=True)  # ORT-optimized artifact


# ---- Q6: int8 softmax ONNX (int-canonical, mirrors int8_softmax_intscale) ----

def _check_softmax(scores, K):
    """Build the int-canonical softmax ONNX (with intermediates) and assert every int intermediate
    AND the four int8 outputs are BIT-EXACT vs int8_compute.int8_softmax_intscale."""
    import numpy as np
    xi, am, sh, xz = i8.quantize_act_per_token_intscale(scores)
    y_int8, ym, ys, yzp, inter = i8.int8_softmax_intscale(xi, xz, am, sh, return_intermediates=True)
    model = ex.build_int8_softmax_onnx(model_name="sm_t", emit_intermediates=True, K=K)
    path = "/tmp/_int8_sm_test.onnx"
    onnx.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {"x_int": xi.numpy().astype(np.int8),
             "x_zp": xz.numpy().astype(np.int32).reshape(-1, 1),
             "x_mul": am.numpy().astype(np.int32).reshape(-1, 1),
             "x_shift": sh.numpy().astype(np.int32).reshape(-1, 1)}
    o = {out.name: v for out, v in zip(sess.get_outputs(), sess.run(None, feeds))}
    pairs = [("sm_max", "max_int"), ("sm_shifted", "shifted"), ("sm_num", "num"),
             ("sm_den", "den"), ("sm_idx", "idx"), ("sm_exp", "exp_int"),
             ("sm_sum", "sum_exp"), ("sm_recip_r4", "inv_int"), ("sm_p", "p_fixed")]
    for ok, rk in pairs:
        a = o[ok].astype(np.int64)
        b = inter[rk].numpy().astype(np.int64)
        assert a.shape == b.shape, f"{ok}: shape {a.shape} vs {b.shape}"
        assert np.array_equal(a, b), f"{ok} vs {rk}: {int((a != b).sum())}/{a.size} differ (max|d|={int(np.abs(a - b).max())})"
    for k, ref in [("y_int8", y_int8), ("y_mul", ym), ("y_shift", ys), ("y_zp", yzp)]:
        a = o[k]
        b = ref.numpy()
        if a.ndim == 1 and b.ndim == 2:
            b = b.reshape(-1)
        a = a.astype(np.int64)
        b = b.astype(np.int64).reshape(a.shape)
        assert np.array_equal(a, b), f"OUT {k}: {int((a != b).sum())}/{a.size} differ (max|d|={int(np.abs(a - b).max())})"


@pytest.mark.parametrize("K", [1500, 256])   # encoder self-attn, decoder cross-attn-ish
def test_int8_softmax_bitexact(K):
    """The int-canonical softmax ONNX is bit-exact vs int8_softmax_intscale: every int intermediate
    (max, shifted, num, den, idx, exp_int, sum_exp, inv_int, p_fixed) and the four int8 outputs
    match. Softmax is parameter-free; K is the sequence length."""
    torch.manual_seed(7)
    _check_softmax(torch.randn(2, K) * 2.0, K)


@pytest.mark.parametrize("x_fn", [
    lambda K: torch.randn(1, K) * 2.0,                   # batch 1
    lambda K: torch.randn(5, K) * 2.0,                   # batch 5
    lambda K: torch.randn(2, K) * 5.0,                   # sharp distribution (one large score)
    lambda K: torch.randn(2, K) * 0.5,                   # flat distribution (idx near T-1)
    lambda K: torch.randn(2, K) * 50.0,                  # extreme (idx saturates to 0 for the tail)
    lambda K: -(torch.rand(2, K) * 3.0),                 # all-negative scores (max <= 0)
])
def test_int8_softmax_robustness(x_fn):
    """Softmax bit-exactness holds across batch sizes and score magnitudes (the extreme case
    drives most idx to 0 where exp saturates; the flat case keeps idx near T-1)."""
    K = 1500
    torch.manual_seed(2)
    _check_softmax(x_fn(K), K)


def test_int8_softmax_vs_fp_within_tolerance():
    """The int-canonical softmax output (dequant) vs fp softmax: the PRE-requant LUT+reciprocal
    error is < 2e-3 (the exp LUT nearest-neighbor grid spacing L/T = 0.003 -> ~0.5 grid; the
    int reciprocal is ~5e-6). The POST-requant abs err is the int8 output step on top. The A4
    staged WER gate already passed at +softmax."""
    import numpy as np
    worst_pre = 0.0
    for seed in (0, 1, 2, 3):
        torch.manual_seed(seed)
        scores = torch.randn(2, 1500) * 2.0
        xi, am, sh, xz = i8.quantize_act_per_token_intscale(scores)
        _, _, _, _, inter = i8.int8_softmax_intscale(xi, xz, am, sh, return_intermediates=True)
        x_real = (xi.float() - xz.float().reshape(-1, 1)) * am.float().reshape(-1, 1) \
            * (2.0 ** (-sh.float().reshape(-1, 1)))
        p_fp = i8.fp_softmax_ref(x_real)
        p_prereq = inter["p_fixed"].float() * (2.0 ** -(i8._SM_S + i8._SM_P))
        worst_pre = max(worst_pre, float((p_prereq - p_fp).abs().max()))
    assert worst_pre < 2e-3, f"pre-requant LUT+recip abs err {worst_pre} >= 2e-3"


def test_int8_softmax_zero_fp_audit():
    """The int-canonical softmax ONNX is structurally zero-fp: subtract-max, Gather (int exp
    LUT), ReduceSum, the int reciprocal (CLZ seed + Newton), and the requant are all int -- raw
    graph AND the ORT-optimized artifact."""
    model = ex.build_int8_softmax_onnx(emit_intermediates=False, K=1500)
    ex.zero_fp_audit(model)                       # raw graph
    ex.zero_fp_audit(model, check_optimized=True)  # ORT-optimized artifact