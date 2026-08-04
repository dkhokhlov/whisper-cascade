#!/usr/bin/env python3
"""Export a saved HQQ Whisper model to ONNX (Path B) for CPU ONNX Runtime inference.

Path B keeps the packed uint8 W_q plus per-group scale/zero as ONNX initializers and emits
the unpack + dequant as ONNX ops, so the exported graph carries the exact HQQ weights (and
the measured WER), not a re-dequantized dense copy. This is a work in progress on the `onnx`
branch; see docs/onnx.md for the spec and the two validation gates.

The export reads HQQ_OUT (a saved HQQ model dir, never written to) and writes the three
ONNX subgraphs plus config/processor/generation files into ONNX_OUT (a separate dir).

This module currently provides:
  * HQQLinearONNX  -- a plain nn.Module that reproduces an HQQLinear's dequant + matmul with
                     standard torch ops (bitwise_and / bitwise_right_shift / cat) so it traces
                     to ONNX BitShift/BitwiseAnd/Concat/Cast/Sub/Mul/Reshape/Transpose/MatMul.
  * swap_hqq_linears -- replace every HQQLinear in a model with HQQLinearONNX.
The optimum ONNX export call (onnx_export_from_model with fn_get_submodels /
custom_onnx_configs) is added once the pinned optimum-onnx 0.0.3 API has been verified
against the installed package; until then main() loads + swaps + reports a summary.
"""

import os
import sys

import torch
import torch.nn as nn

import hqq_asr

HQQ_OUT = os.environ.get("HQQ_OUT", "whisper-tiny-hqq-4bit")
ONNX_OUT = os.environ.get("ONNX_OUT", "whisper-tiny-hqq-onnx")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cpu").strip().lower() or "cpu"


class HQQLinearONNX(nn.Module):
    """Plain-torch reimplementation of an HQQLinear for ONNX export.

    Reproduces the HQQ dequant formula W = (q - zero) * scale, reshaped to (O, I), then
    out = x @ W.T (+ bias). axis=1, group_size=32, fp32 compute (this repo's compute_dtype).

    4-bit: W_q is uint8 (O*I/64, 32); the pre-pack tensor (O*I/32, 32) is split into two
      halves along axis 0 -- first half in the high nibble, second half in the low nibble.
      Unpack: high = (W_q & 0xF0) >> 4, low = W_q & 0x0F, Concat([high, low], axis=0).
    8-bit: W_q is uint8 (O*I/32, 32), 1:1; unpack is the identity (cast to fp32).

    Buffers (W_q, zero, scale, bias) become ONNX graph initializers. No custom autograd, no
    hqq metadata dispatch -- traces cleanly to standard ONNX ops.
    """

    def __init__(self, w_q, zero, scale, bias, nbits, out_features, in_features):
        super().__init__()
        self.register_buffer("W_q", w_q.detach().cpu().contiguous())
        self.register_buffer("zero", zero.detach().cpu().contiguous())
        self.register_buffer("scale", scale.detach().cpu().contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.detach().cpu().contiguous())
        else:
            self.register_buffer("bias", None)
        self.nbits = int(nbits)
        self.out_features = int(out_features)
        self.in_features = int(in_features)

    def dequantize(self):
        """Return the dequantized fp32 weight (O, I); mirrors HQQLinear.dequantize()."""
        q = self.W_q
        if self.nbits == 4:
            high = torch.bitwise_right_shift(
                torch.bitwise_and(q, torch.full_like(q, 0xF0)), torch.full_like(q, 4)
            )
            low = torch.bitwise_and(q, torch.full_like(q, 0x0F))
            qr = torch.cat((high, low), dim=0)
        elif self.nbits == 8:
            qr = q
        else:
            raise NotImplementedError(f"HQQLinearONNX: nbits={self.nbits} not supported")
        w = (qr.to(torch.float32) - self.zero) * self.scale
        return w.reshape(self.out_features, self.in_features)

    def forward(self, x):
        w = self.dequantize()
        y = torch.matmul(x, w.t())
        if self.bias is not None:
            y = y + self.bias
        return y

    @classmethod
    def from_hqq(cls, hqq_linear):
        """Build an HQQLinearONNX from a loaded HQQLinear (copies W_q/zero/scale/bias/meta)."""
        meta = hqq_linear.meta
        nbits = int(meta["nbits"])
        if int(meta.get("axis", 1)) != 1:
            raise NotImplementedError("HQQLinearONNX: only axis=1 is supported (repo default)")
        if meta.get("view_as_float", False):
            raise NotImplementedError("HQQLinearONNX: view_as_float storage not supported")
        out_features, in_features = meta["shape"]
        return cls(
            w_q=hqq_linear.W_q,
            zero=meta["zero"],
            scale=meta["scale"],
            bias=hqq_linear.bias,
            nbits=nbits,
            out_features=out_features,
            in_features=in_features,
        )


def swap_hqq_linears(model):
    """Replace every HQQLinear in model with an HQQLinearONNX (in place). Exempt modules
    (proj_out, embeddings, convs, norms) are not HQQLinear and are left untouched.

    Returns (n_default, n_tier8) -- the 4-bit and 8-bit tier counts.
    """
    from hqq.core.quantize import HQQLinear

    n_default = 0
    n_tier8 = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, HQQLinear):
            continue
        parent = model
        for part in name.split(".")[:-1]:
            parent = getattr(parent, part)
        leaf = name.split(".")[-1]
        nbits = int(module.meta["nbits"])
        if nbits == 4:
            n_default += 1
        elif nbits == 8:
            n_tier8 += 1
        else:
            raise NotImplementedError(f"swap_hqq_linears: nbits={nbits} not supported")
        setattr(parent, leaf, HQQLinearONNX.from_hqq(module))
    return n_default, n_tier8


def main() -> int:
    print(f"loading HQQ model {HQQ_OUT} (device={ASR_DEVICE})", file=sys.stderr)
    model = hqq_asr.load_whisper_hqq(HQQ_OUT, device=ASR_DEVICE)
    model.eval()
    n_default, n_tier8 = swap_hqq_linears(model)
    print(
        f"swapped: 4-bit tier={n_default}, 8-bit tier={n_tier8} (HQQLinear -> HQQLinearONNX)",
        file=sys.stderr,
    )
    # TODO(onnx): call onnx_export_from_model(model, ..., fn_get_submodels=...,
    # custom_onnx_configs=...) with the Whisper encoder / decoder / decoder-with-past configs
    # (dynamo=False, do_constant_folding=False, opset=18, canonical filenames), then copy
    # config.json + processor files + generation_config.json from HQQ_OUT into ONNX_OUT. The
    # exact optimum-onnx 0.0.3 API is verified against the installed package before this is
    # filled in; until then this only loads + swaps + reports.
    summary = {
        "hqq_out": HQQ_OUT,
        "onnx_out": ONNX_OUT,
        "linears_default_bit": n_default,
        "linears_8bit": n_tier8,
        "status": "swapped (ONNX serialization not yet implemented)",
    }
    print(__import__("json").dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())