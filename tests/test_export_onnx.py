"""Unit tests for the ONNX export (HQQLinearONNX).

Two layers of tests:
  * dequant / single-linear equivalence vs hqq -- pure torch + hqq, run under the CPU .venv
    (`make test`). Build a small HQQLinear (4-bit and 8-bit), wrap it as HQQLinearONNX, and
    assert the dequantized weight and the forward output match hqq for rank-2 and rank-3 x.
  * op probe -- export HQQLinearONNX to ONNX and run it in ONNX Runtime CPU; guard with
    importorskip("onnxruntime") so `make test` (no onnxruntime) skips them and
    `.venv-onnx/bin/python -m pytest` runs them. This validates that torch 2.4.1 emits the
    bitwise dequant as runnable ONNX ops (opset 18). If it fails, export_onnx.py must switch
    to the Div/Mod fallback.
"""

import torch
import torch.nn as nn
import pytest

from hqq.core.quantize import HQQLinear, BaseQuantizeConfig

from export_onnx import HQQLinearONNX


def _make_hqq(nbits, out_f=32, in_f=64):
    """Quantize a small random Linear with HQQ (group=32, axis=1, fp32 compute)."""
    torch.manual_seed(0)
    lin = nn.Linear(in_f, out_f, bias=True)
    cfg = BaseQuantizeConfig(nbits=nbits, group_size=32, axis=1)
    return HQQLinear(lin, cfg, compute_dtype=torch.float32, device="cpu")


def _check_equiv(nbits):
    hqq_lin = _make_hqq(nbits)
    onnx_lin = HQQLinearONNX.from_hqq(hqq_lin)

    # dequantized weight matches hqq exactly (same integer unpack + fp32 sub/mul).
    w_hqq = hqq_lin.dequantize()
    w_onnx = onnx_lin.dequantize()
    assert w_hqq.shape == w_onnx.shape == (hqq_lin.out_features, hqq_lin.in_features)
    assert torch.allclose(w_onnx, w_hqq, atol=1e-6), (
        f"nbits={nbits}: dequant weight mismatch, max abs diff "
        f"{(w_onnx - w_hqq).abs().max().item():.3e}"
    )

    # forward matches for rank-2 (batch, in) and rank-3 (batch, seq, in) inputs.
    for x in (
        torch.randn(3, hqq_lin.in_features),
        torch.randn(2, 5, hqq_lin.in_features),
    ):
        y_hqq = hqq_lin(x)
        y_onnx = onnx_lin(x)
        assert torch.allclose(y_onnx, y_hqq, atol=1e-5), (
            f"nbits={nbits}: forward mismatch (x {tuple(x.shape)}), max abs diff "
            f"{(y_onnx - y_hqq).abs().max().item():.3e}"
        )


def test_dequant_equiv_4bit():
    _check_equiv(4)


def test_dequant_equiv_8bit():
    _check_equiv(8)


def _ort_probe(nbits):
    onnxruntime = pytest.importorskip("onnxruntime")  # noqa: F821
    import io

    hqq_lin = _make_hqq(nbits)
    onnx_lin = HQQLinearONNX.from_hqq(hqq_lin)
    onnx_lin.eval()

    x = torch.randn(3, hqq_lin.in_features)
    y_torch = onnx_lin(x).detach()

    buf = io.BytesIO()
    torch.onnx.export(
        onnx_lin,
        (x,),
        buf,
        opset_version=18,
        input_names=["x"],
        output_names=["y"],
        dynamo=False,
        do_constant_folding=False,
    )
    import onnx

    model = onnx.load(io.BytesIO(buf.getvalue()))
    # Structural check: the graph keeps a uint8 W_q initializer and the dequant ops.
    dtypes = {init.data_type for init in model.graph.initializer}
    op_types = {node.op_type for node in model.graph.node}
    # onnx TensorProto.UINT8 == 2
    assert 2 in dtypes, "no uint8 initializer (W_q not kept packed)"
    assert {"MatMul"} <= op_types, "no MatMul"
    if nbits == 4:
        # 4-bit: unpack via BitwiseAnd/BitShift (or the Div/Mod fallback) must be present.
        assert any(o in op_types for o in ("BitwiseAnd", "BitShift", "Mod", "Div")), (
            "4-bit: no unpack op (BitwiseAnd/BitShift or Div/Mod fallback)"
        )
    else:
        # 8-bit: 1:1, no bit-packing; the "unpack" is just Cast uint8 -> fp32, no bitwise ops.
        assert not (op_types & {"BitwiseAnd", "BitShift"}), (
            "8-bit: unexpected bitwise unpack op (should be a plain Cast)"
        )
        assert "Cast" in op_types, "8-bit: no Cast (uint8 -> fp32)"

    sess = onnxruntime.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    y_ort = sess.run(["y"], {"x": x.numpy()})[0]
    assert torch.allclose(torch.from_numpy(y_ort), y_torch, atol=1e-5), (
        f"nbits={nbits}: ORT vs torch mismatch, max abs diff "
        f"{(torch.from_numpy(y_ort) - y_torch).abs().max().item():.3e}"
    )


def test_ort_probe_4bit():
    _ort_probe(4)


def test_ort_probe_8bit():
    _ort_probe(8)