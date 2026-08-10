#!/usr/bin/env bash
# Full whisper-small benchmark on the A10 GPU (same matrix as tiny/base).
# Small results are prefixed small_ alongside the tiny/base results. Uses the
# CUDA .venv-gpu (make gpu-venv) and ASR_DEVICE=cuda; WER is host-independent.
# Writes under $OUT_DIR/ (default evals/; make bench-matrix overrides
# OUT_DIR=build).
cd "$(dirname "$0")/.."
set -a; . ~/.api_keys; set +a
export ASR_DEVICE=cuda
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
OUT_DIR=${OUT_DIR:-evals}
mkdir -p "$OUT_DIR/multilingual" "$OUT_DIR/telephone"
PY=.venv-gpu/bin/python
run () {  # <dataset> <config> <split> <limit> <model_tag> <out>
  local ds="$1" cfg="$2" sp="$3" lim="$4" tag="$5" out="$6" ma
  if [ "$tag" = "fp32" ]; then ma="MODEL_ASR=openai/whisper-small QUANT="; else ma="MODEL_ASR=./whisper-small-hqq-4bit QUANT=hqq"; fi
  echo "=== small $ds/$cfg split=$sp n=$lim $tag ===" >&2
  for a in 1 2; do
    env $ma ASR_DEVICE=cuda EVAL_DATASET="$ds" EVAL_CONFIG="$cfg" EVAL_SPLIT="$sp" EVAL_LIMIT="$lim" \
        EVAL_OUT="$out" $PY eval_wer.py >/dev/null 2>&1
    if [ -f "$out" ] && $PY -c "import json;json.load(open('$out'))" 2>/dev/null; then
      echo "  ok" >&2; return 0
    fi
  done
  echo "  FAIL $out" >&2; return 1
}
# fleurs multilingual (n=100)
for cfg in en_us es_419 fr_fr de_de hi_in; do
  run google/fleurs "$cfg" test 100 fp32 "$OUT_DIR/multilingual/small_${cfg}_fp32.json"
  run google/fleurs "$cfg" test 100 hqq  "$OUT_DIR/multilingual/small_${cfg}_hqq.json"
done
# talkbank segment (n=100)
for lang in en es fr de; do
  run diabolocom/talkbank_4_stt "$lang" segment 100 fp32 "$OUT_DIR/telephone/small_talkbank_${lang}_fp32.json"
  run diabolocom/talkbank_4_stt "$lang" segment 100 hqq  "$OUT_DIR/telephone/small_talkbank_${lang}_hqq.json"
done
echo "SMALL BENCH DONE" >&2