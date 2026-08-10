#!/usr/bin/env bash
# Full whisper-tiny benchmark (fleurs multilingual + talkbank telephone), robust
# to per-run aborts. Tiny results are unprefixed alongside the base_/small_ ones.
# Writes one JSON per (dataset, config, model) under $OUT_DIR/
# (default evals/; make bench-matrix overrides OUT_DIR=build).
cd "$(dirname "$0")/.."
set -a; . ~/.api_keys; set +a
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
OUT_DIR=${OUT_DIR:-evals}
mkdir -p "$OUT_DIR/multilingual" "$OUT_DIR/telephone"

run () {  # <dataset> <config> <split> <limit> <model_tag> <out>
  local ds="$1" cfg="$2" sp="$3" lim="$4" tag="$5" out="$6" ma
  if [ "$tag" = "fp32" ]; then ma="MODEL_ASR=openai/whisper-tiny QUANT="; else ma="MODEL_ASR=./whisper-tiny-hqq-4bit QUANT=hqq"; fi
  echo "=== $ds/$cfg split=$sp n=$lim $tag ===" >&2
  # Try up to twice; accept the run if a valid JSON was written (the torch
  # shutdown abort can return 134 after the file is already flushed).
  for attempt in 1 2; do
    env $ma EVAL_DATASET="$ds" EVAL_CONFIG="$cfg" EVAL_SPLIT="$sp" EVAL_LIMIT="$lim" \
        EVAL_OUT="$out" python eval_wer.py >/dev/null 2>&1
    if [ -f "$out" ] && python -c "import json,sys;json.load(open('$out'))" 2>/dev/null; then
      echo "  ok (attempt $attempt)" >&2; return 0
    fi
  done
  echo "  FAIL $out" >&2; return 1
}

# fleurs multilingual (n=100)
for cfg in en_us es_419 fr_fr de_de hi_in; do
  run google/fleurs "$cfg" test 100 fp32 "$OUT_DIR/multilingual/${cfg}_fp32.json"
  run google/fleurs "$cfg" test 100 hqq  "$OUT_DIR/multilingual/${cfg}_hqq.json"
done
# talkbank_4_stt, segment split, n=100 per language.
for lang in en es fr de; do
  run diabolocom/talkbank_4_stt "$lang" segment 100 fp32 "$OUT_DIR/telephone/talkbank_${lang}_fp32.json"
  run diabolocom/talkbank_4_stt "$lang" segment 100 hqq  "$OUT_DIR/telephone/talkbank_${lang}_hqq.json"
done

echo "ALL DONE" >&2
