#!/usr/bin/env python3
"""Upload the exported ONNX files into the HQQ Hugging Face repo.

The HQQ repos (dkhokhlov/whisper-<flavor>-hqq-4bit) already hold config.json,
the processor/tokenizer files, and qmodel.pt from the HQQ publish. This script
adds only the .onnx (+ .onnx_data) files from ONNX_OUT (the build/
whisper-<flavor>-hqq-onnx dir produced by `make onnx`), so the repo serves both
load paths: QUANT=hqq (qmodel.pt) and QUANT=onnx (ORTModelForSpeechSeq2Seq over
the .onnx files). Optionally uploads MODEL_CARD as README.md when set (use this
to publish the updated model card that documents the ONNX format).

Needs HF_TOKEN_WRITE (the Makefile push-onnx target sources ~/.api_keys).

Usage:
    ONNX_OUT=build/whisper-tiny-hqq-onnx HQQ_REPO=dkhokhlov/whisper-tiny-hqq-4bit \
        MODEL_CARD=docs/hqq_report_tiny.md python push_onnx.py
"""

import json
import os
import sys

from huggingface_hub import HfApi

ONNX_OUT = os.environ.get("ONNX_OUT", "").strip()
HQQ_REPO = os.environ.get("HQQ_REPO", "").strip()
MODEL_CARD = os.environ.get("MODEL_CARD", "").strip()


def main() -> int:
    if not ONNX_OUT or not HQQ_REPO:
        print("ONNX_OUT and HQQ_REPO are required", file=sys.stderr)
        return 2
    tok = os.environ.get("HF_TOKEN_WRITE")
    if not tok:
        print(
            "HF_TOKEN_WRITE is not set. Source ~/.api_keys first "
            "(set -a; . ~/.api_keys; set +a).",
            file=sys.stderr,
        )
        return 2

    api = HfApi(token=tok)
    api.create_repo(repo_id=HQQ_REPO, exist_ok=True, private=False)

    # Upload only the ONNX files; the repo's config/processor/qmodel.pt stay.
    api.upload_folder(
        folder_path=ONNX_OUT,
        repo_id=HQQ_REPO,
        token=tok,
        allow_patterns=["*.onnx", "*.onnx_data"],
    )
    n_onnx = sum(1 for f in os.listdir(ONNX_OUT) if f.endswith(".onnx"))

    # Optionally publish the updated model card (with the ONNX section) as README.md.
    card_uploaded = False
    if MODEL_CARD and os.path.isfile(MODEL_CARD):
        api.upload_file(
            path_or_fileobj=MODEL_CARD,
            path_in_repo="README.md",
            repo_id=HQQ_REPO,
            token=tok,
        )
        card_uploaded = True

    sha = api.repo_info(repo_id=HQQ_REPO).sha
    summary = {
        "onnx_out": ONNX_OUT,
        "hf_repo": HQQ_REPO,
        "hf_sha": sha,
        "n_onnx_files": n_onnx,
        "model_card_uploaded": card_uploaded,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"pushed {ONNX_OUT} ({n_onnx} .onnx) -> {HQQ_REPO} @ {sha}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())