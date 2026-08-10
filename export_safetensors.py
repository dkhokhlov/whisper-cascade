#!/usr/bin/env python3
"""Export a saved HQQ Whisper model's qmodel.pt to model.safetensors.

The hqq loader stores weights in qmodel.pt, a torch pickle (zip archive). This
script re-exports the same weights as a single safetensors file: a flat
8-byte-JSON-header + raw-tensor-bytes format with no pickle, zero-mappable,
and parseable from C/C++/Rust (useful for deployment host tooling). model.safetensors
is an ADDITIONAL file; qmodel.pt stays the default loader target.

Safetensors stores tensors only, so the per-linear HQQ config scalars (nbits,
group_size, axis, the packing string, the bools) are encoded as tensors via
the HQQLinear encoded_state_dict path. HQQLinear.load_state_dict detects the
"encoded_state_dict" key and decodes them back, so the round-trip needs no
extra metadata file. The export loads on CPU (the saved qmodel.pt is
device-independent); the output safetensors is device-independent too.

Usage:
    python export_safetensors.py
    HQQ_OUT=whisper-base-hqq-4bit python export_safetensors.py
"""

import json
import os
import sys

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import hqq_asr

HQQ_OUT = os.environ.get("HQQ_OUT", "whisper-tiny-hqq-4bit")


def export_safetensors(hqq_out: str) -> dict:
    """Export `<hqq_out>/qmodel.pt` to `<hqq_out>/model.safetensors`.

    Re-exports the saved HQQ weights as a single pickle-free safetensors file
    (flat tensor map; see module docstring for the per-linear config encoding).
    Returns a summary dict. qmodel.pt is left in place as the default loader
    target; model.safetensors is the additional, host-tooling-consumable file.
    """
    from hqq.core.quantize import HQQLinear
    from safetensors.torch import save_file

    out = os.path.join(hqq_out, "model.safetensors")
    print(f"exporting {hqq_out}/qmodel.pt -> {out}", file=sys.stderr)
    # Load on CPU: the export only reads the packed weights, so no GPU is
    # needed and the output is device-independent.
    model = hqq_asr.load_whisper_hqq(hqq_out, device="cpu")
    model.eval()
    # Encode the per-linear HQQ config (scalars/strings/bools) as tensors so the
    # whole state_dict is safetensors-compatible (tensors only). load_state_dict
    # decodes them back via the "encoded_state_dict" flag.
    n_linear = 0
    for module in model.modules():
        if isinstance(module, HQQLinear):
            module.encoded_state_dict = True
            n_linear += 1
        else:
            # Match qmodel.pt compact storage: keep the non-quantized modules
            # (embedding, convs, norms, proj_out) at fp16. HQQLinear overrides
            # .to() as a no-op, so it keeps its fp32 scale/zero/bias. Loading
            # the safetensors upcasts the fp16 weights to fp32 (compute dtype),
            # identical to loading qmodel.pt.
            module.to(hqq_asr.STORE_DTYPE)
    state = model.state_dict()
    # safetensors requires CPU-contiguous tensors.
    flat = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    # proj_out is tied to the decoder embedding (same weight, shared storage).
    # safetensors cannot store shared tensors, so drop the tied duplicate here;
    # embed_tokens keeps the single copy, matching qmodel.pt's dedup. The
    # safetensors loader (hqq_asr.load_weights) re-aliases proj_out to
    # embed_tokens, and load_whisper_hqq re-ties them after from_quantized.
    # WER-neutral: the tie is value-identical.
    flat.pop("proj_out.weight", None)
    save_file(flat, out)

    dtypes = {}
    for t in flat.values():
        dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
    return {
        "out_dir": hqq_out,
        "file": "model.safetensors",
        "size_bytes": os.path.getsize(out),
        "size_mb": round(os.path.getsize(out) / 1e6, 2),
        "n_tensors": len(flat),
        "n_hqq_linears": n_linear,
        "dtypes": dtypes,
    }


def main() -> int:
    summary = export_safetensors(HQQ_OUT)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())