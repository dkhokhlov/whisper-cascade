#!/usr/bin/env python3
"""Quantize a Whisper model with HQQ 4-bit grouped quantization and save it.

The script loads the source model, replaces its linear layers with HQQ
4-bit weights (group_size=32, axis=1), keeps the tied proj_out head in fp16,
stores the non-quantized modules in fp16, and writes the result to HQQ_OUT.
The output dir holds config.json, qmodel.pt, and the processor files, so it
is self-contained for AutoProcessor.from_pretrained and the hqq loader.

Set PUSH=1 to upload HQQ_OUT to the Hugging Face Hub repo HQQ_REPO (default
dkhokhlov/whisper-tiny-hqq-4bit). The upload needs a Hugging Face token with
write access (HF_TOKEN or huggingface-cli login).

Usage:
    python quantize.py
    PUSH=1 HQQ_REPO=dkhokhlov/whisper-tiny-hqq-4bit python quantize.py
"""

import json
import os
import sys

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import hqq_asr
import export_safetensors

MODEL_ASR = os.environ.get("MODEL_ASR", "openai/whisper-tiny")
HQQ_OUT = os.environ.get("HQQ_OUT", "whisper-tiny-hqq-4bit")
HQQ_NBITS = int(os.environ.get("HQQ_NBITS", "4"))
HQQ_GROUP = int(os.environ.get("HQQ_GROUP", "32"))
# axis=1 groups along the input/reduction dim and measures better than axis=0
# on whisper-tiny (0.1622 vs 0.2032 WER on fleurs en_us); axis=0 targets
# GPU-optimized inference kernels.
HQQ_AXIS = int(os.environ.get("HQQ_AXIS", "1"))
# Sensitive linears assigned to the 8-bit tier. HQQ_8BIT_PATTERNS is a
# comma-separated list of name substrings; default keeps the whole encoder
# stack and fc1 (the GELU up-projection) at 8-bit. Measured best on fleurs
# en_us (WER matches the fp32 baseline at ~61% size reduction).
HQQ_8BIT_NBITS = int(os.environ.get("HQQ_8BIT_NBITS", "8"))
HQQ_8BIT_PATTERNS = tuple(
    p.strip() for p in os.environ.get("HQQ_8BIT_PATTERNS", "encoder.layers,fc1").split(",") if p.strip()
)
PUSH = os.environ.get("PUSH", "").strip() not in ("", "0", "false")
HQQ_REPO = os.environ.get("HQQ_REPO", "dkhokhlov/whisper-tiny-hqq-4bit")
# Quantization report copied as the model card when it exists (it holds the
# measured WER results). One report per model size: docs/hqq_report_tiny.md
# (tiny), docs/hqq_report_base.md (base), docs/hqq_report_small.md (small).
# The push step sets this so each upload gets its own card instead of tiny's.
HQQ_REPORT = os.environ.get("HQQ_REPORT", "docs/hqq_report_tiny.md")
# Multilingual generation config (default on): clear the English
# forced_decoder_ids baked into the source config and write a modern
# generation_config.json so the model auto-detects the language and
# transcribes. The English-forced config (v1.3.1) is preserved at git tag
# v1.3.1 / HF revision e43f2bb; English WER is identical under auto-detect.
# Set HQQ_MULTILINGUAL=0 to reproduce the legacy English-forced config.
HQQ_MULTILINGUAL = os.environ.get("HQQ_MULTILINGUAL", "1").strip() not in ("", "0", "false")
# Compute device: "cpu" (default) keeps the original CPU behavior; "cuda"
# quantizes on GPU (needs CUDA torch, e.g. the .venv-gpu env for the A10). The
# saved qmodel.pt is device-independent and loads on CPU or GPU.
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cpu").strip().lower() or "cpu"

MODEL_CARD = """# __REPO__

HQQ 4-bit grouped quantization of `__MODEL__` for CPU inference.

## Quantization

- Method: HQQ (Half-Quadratic Quantization), no calibration data.
- Linear layers: nbits=4, group_size=32, axis=1 (group along input dim; measured best on whisper-tiny).
- 8-bit tier (whole encoder stack + fc1): quantized at 8-bit.
- proj_out (the lm_head): kept tied to the embedding in fp16 (not quantized).
  It shares the weight with the decoder embed_tokens, so quantizing it would
  store the weight twice and add error to the vocab projection.
- Non-quantized modules (embedding, convs, layer norms): stored as fp16.
- Compute dtype: fp32 (fp16 weights are upcast at load time).

## Load

```python
import hqq_asr
pipe = hqq_asr.build_pipeline("__REPO__", quant="hqq")
print(pipe({"array": audio, "sampling_rate": 16000})["text"])
```

## Results

See the repository README for the WER comparison against the fp32 baseline on
google/fleurs en_us (test split).
"""


def dir_size(path):
    """Return the total size in bytes of all files under path."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def main() -> int:
    print(
        f"quantizing {MODEL_ASR} -> {HQQ_OUT} "
        f"(nbits={HQQ_NBITS}, group_size={HQQ_GROUP}, axis={HQQ_AXIS}, "
        f"tier8={HQQ_8BIT_PATTERNS}@{HQQ_8BIT_NBITS}bit, "
        f"multilingual={HQQ_MULTILINGUAL})",
        file=sys.stderr,
    )
    counts = hqq_asr.quantize_whisper(
        MODEL_ASR, HQQ_OUT, nbits=HQQ_NBITS, group_size=HQQ_GROUP, axis=HQQ_AXIS,
        tier8_nbits=HQQ_8BIT_NBITS if HQQ_8BIT_PATTERNS else None,
        tier8_patterns=HQQ_8BIT_PATTERNS, multilingual=HQQ_MULTILINGUAL, device=ASR_DEVICE,
    )

    # Use the quantization report (HQQ_REPORT) as the model card when it exists
    # (it has the measured WER results). Otherwise fall back to the minimal
    # card below.
    card_path = os.path.join(HQQ_OUT, "README.md")
    if os.path.exists(HQQ_REPORT):
        import shutil
        shutil.copyfile(HQQ_REPORT, card_path)
    else:
        with open(card_path, "w", encoding="utf-8") as fh:
            repo = HQQ_REPO if PUSH else HQQ_OUT
            fh.write(MODEL_CARD.replace("__REPO__", repo).replace("__MODEL__", MODEL_ASR))

    # Export model.safetensors alongside qmodel.pt so the pushed repo is
    # consumable from host tooling without a torch pickle loader. Generated for
    # every quantize (so HQQ_OUT matches what `make push` uploads); see
    # export_safetensors.py and docs/ASR.md#safetensors-format.
    st_summary = export_safetensors.export_safetensors(HQQ_OUT)
    size = dir_size(HQQ_OUT)
    summary = {
        "source_model": MODEL_ASR,
        "out_dir": HQQ_OUT,
        "nbits": HQQ_NBITS,
        "group_size": HQQ_GROUP,
        "axis": HQQ_AXIS,
        "multilingual": HQQ_MULTILINGUAL,
        "tier8_patterns": list(HQQ_8BIT_PATTERNS),
        "tier8_nbits": HQQ_8BIT_NBITS if HQQ_8BIT_PATTERNS else None,
        "linears_default_bit": counts["default"],
        "linears_8bit": counts["tier8"],
        "size_bytes": size,
        "size_mb": round(size / 1e6, 2),
        "files": sorted(os.listdir(HQQ_OUT)),
        "safetensors": st_summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if PUSH:
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_TOKEN_WRITE")
        if not tok:
            raise RuntimeError(
                "HF_TOKEN_WRITE is not set. Source it first: "
                "set -a; . ~/.api_keys; set +a"
            )
        api = HfApi(token=tok)
        api.create_repo(repo_id=HQQ_REPO, exist_ok=True, private=False)
        api.upload_folder(folder_path=HQQ_OUT, repo_id=HQQ_REPO, token=tok)
        sha = api.repo_info(repo_id=HQQ_REPO).sha
        # Print the HF revision sha so the code-repo commit can reference it
        # (every upload is tagged with its HF hash).
        print(f"pushed {HQQ_OUT} -> {HQQ_REPO} @ {sha}", file=sys.stderr)
        summary["hf_repo"] = HQQ_REPO
        summary["hf_sha"] = sha
    return 0


if __name__ == "__main__":
    sys.exit(main())