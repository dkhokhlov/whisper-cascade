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
    return {"qr_g": qr.reshape(O, I // g, g), "A": A.to(torch.int64), "Bt": Bt.to(torch.int64),
            "C": C.to(torch.int64), "T1": T1, "T2": T2, "p1_out": p1, "p2_out": p2,
            "acc": acc, "out": out, "xi": xi, "xz": xz, "am": am, "sh": sh}


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