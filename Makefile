VENV  := .venv
PY    := $(VENV)/bin/python
# GPU venv for A10 runs (CUDA torch). Separate from the CPU .venv so the
# CPU-only project contract and pyproject.toml stay untouched. Used by the
# whisper-small eval (make gpu-venv, then run scripts with $(PYGPU)).
VENV_GPU := .venv-gpu
PYGPU    := $(VENV_GPU)/bin/python
# ONNX export/eval venv (CPU). Separate from the CPU .venv so the strict CPU contract and
# pyproject.toml stay untouched, and from .venv-gpu (CUDA). Holds optimum/onnxruntime/onnx
# pinned against transformers 4.44.2 / torch 2.4.1. Used by make onnx / eval-onnx.
VENV_ONNX := .venv-onnx
PYONNX    := $(VENV_ONNX)/bin/python
# Semantic version, sourced from pyproject.toml so it stays in sync.
VERSION := $(shell awk -F'"' '/^version =/ {print $$2; exit}' pyproject.toml)
# Whisper ASR model (override: make asr MODEL_ASR=openai/whisper-tiny.en). The
# default whisper-tiny is multilingual and transcribes the EN/ES/HI samples.
MODEL_ASR ?= openai/whisper-tiny
# Custom audio input: make asr AUDIO=file.wav | dir/ | '*.flac' (space-safe).
export AUDIO
# Text input for make en / make tts: make en TEXT="..." or pipe stdin. File
# paths and globs with spaces stay intact through the env var.
export TEXT
# TTS output path: make tts OUTPUT=out.wav (default tts.wav).
export OUTPUT
# Translate target model (MarianMT; default multilingual -> English).
MODEL_TRANSLATE ?= Helsinki-NLP/opus-mt-mul-en
# Translate target language code (printed in the make en JSON as "lang").
# A future make es sets this to "es" (with a -> Spanish MODEL_TRANSLATE).
TARGET_LANG ?= en
# TTS model (VITS; default English MMS).
MODEL_TTS ?= facebook/mms-tts-eng
# Optional HQQ 4-bit quantization mode for ASR. Set QUANT=hqq to load MODEL_ASR
# as a saved HQQ model (local dir or HF repo). Default: fp32.
export QUANT
# ASR compute device: "cpu" (default) keeps the original CPU behavior; "cuda"
# runs the model on GPU (needs CUDA torch, e.g. the .venv-gpu env for the A10).
# Exported so make asr / quantize / push / eval-* all forward it to the scripts.
ASR_DEVICE ?= cpu
export ASR_DEVICE
# Repo-local, gitignored build dir for transient artifacts: HQQ quantize output,
# ONNX export output, eval/gate manifests, test output/logs. Models and datasets
# otherwise live in the default HF cache. make clean wipes this.
BUILD ?= build
# HQQ quantization output dir, HF repo, and quant config (make quantize / push).
# Defaults are the best measured config on fleurs en_us (WER 0.1367 vs 0.1381
# fp32 baseline): 4-bit, group_size=32, axis=1, the whole encoder stack and
# fc1 assigned to the 8-bit tier.
HQQ_OUT ?= $(BUILD)/whisper-tiny-hqq-4bit
HQQ_REPO ?= dkhokhlov/whisper-tiny-hqq-4bit
# ONNX export output dir. Separate from HQQ_OUT: the optimum exporter may overwrite
# config.json/processor/index files, so it must not write into the canonical HQQ source dir.
# eval-onnx points MODEL_ASR here (not HQQ_OUT). The export sources HQQ weights from
# HQQ_REPO (HF repo id -> default HF cache), so the local HQQ_OUT dir is not needed.
ONNX_OUT ?= $(BUILD)/whisper-tiny-hqq-onnx
HQQ_GROUP ?= 32
HQQ_AXIS ?= 1
HQQ_8BIT_PATTERNS ?= encoder.layers,fc1
HQQ_8BIT_NBITS ?= 8
# WER eval dataset (make eval-baseline / eval-hqq). fleurs (read speech),
# or diabolocom/talkbank_4_stt (telephone, split=segment).
EVAL_DATASET ?= google/fleurs
# WER eval split (fleurs test, talkbank segment).
EVAL_SPLIT ?= test
# WER eval subset size (make eval-baseline / eval-hqq).
EVAL_LIMIT ?= 100
# WER eval config / language code (make eval-baseline / eval-hqq).
EVAL_CONFIG ?= en_us

.PHONY: help info venv gpu-venv onnx-venv samples asr en tts quantize push eval-baseline eval-hqq eval-onnx onnx hqq-reference test test-integration clean clean-all

help: ## Show available targets
	@echo "whisper-cascade v$(VERSION)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make <target> [VAR=value]\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  %-18s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@printf '  %-52s %s\n' 'make asr' 'default samples (en+es+hi, whisper-tiny)'
	@printf '  %-52s %s\n' 'make asr MODEL_ASR=openai/whisper-tiny.en' 'English-only model'
	@printf '  %-52s %s\n' 'make asr AUDIO=hf://datasets/Narsil/asr_dummy/1.flac' 'one HF Hub file'
	@printf '  %-52s %s\n' 'make asr AUDIO=./clips/' 'a directory'
	@printf '  %-52s %s\n' "make asr AUDIO='*.flac'" 'a glob'
	@printf '  %-52s %s\n' 'make en TEXT="Hola"' 'translate text to English'
	@printf '  %-52s %s\n' 'make tts TEXT="Hello"' 'synthesize speech to tts.wav'
	@printf '  %-52s %s\n' 'make asr QUANT=hqq MODEL_ASR=dkhokhlov/whisper-tiny-hqq-4bit' 'HQQ 4-bit ASR from HF'
	@printf '  %-52s %s\n' 'make quantize' 'quantize whisper-tiny -> HQQ_OUT (build/)'
	@printf '  %-52s %s\n' 'make push' 'quantize + upload to HQQ_REPO (needs HF_TOKEN_WRITE)'
	@printf '  %-52s %s\n' 'make eval-baseline EVAL_CONFIG=en_us' 'WER of fp32 MODEL_ASR on a fleurs config'
	@printf '  %-52s %s\n' 'make eval-hqq EVAL_CONFIG=es_419' 'WER of HQQ MODEL_ASR on a fleurs config'
	@printf '  %-52s %s\n' 'make eval-baseline EVAL_DATASET=diabolocom/talkbank_4_stt EVAL_CONFIG=es EVAL_SPLIT=segment' 'talkbank telephone Spanish'
	@printf '  %-52s %s\n' 'make test' 'fast unit tests'
	@echo ""
	@echo "Pipeline (foreign speech -> English speech; each stage prints JSON, jq extracts text):"
	@echo "  make asr AUDIO=hf://datasets/Narsil/asr_dummy/4.flac | jq -r '.[].text' | make en | jq -r '.text' | make tts OUTPUT=4_en.wav"
	@echo "  make asr AUDIO=hf://datasets/Narsil/asr_dummy/4.flac | tee out.txt | jq -r '.[].text' | make en | tee -a out.txt | jq -r '.text' | make tts OUTPUT=4_en.wav >> out.txt && cat out.txt"
	@echo ""

info: ## Show current config and status
	@echo "MODEL_ASR         $(MODEL_ASR)"
	@echo "QUANT             $${QUANT:-<none (fp32)>}"
	@echo "MODEL_TRANSLATE   $(MODEL_TRANSLATE)"
	@echo "MODEL_TTS         $(MODEL_TTS)"
	@echo "EVAL_DATASET      $(EVAL_DATASET)"
	@echo "EVAL_CONFIG       $(EVAL_CONFIG)   EVAL_SPLIT $(EVAL_SPLIT)   EVAL_LIMIT $(EVAL_LIMIT)"
	@echo "VENV              $(VENV) - $$([ -d $(VENV) ] && echo present || echo missing)"
	@echo "SAMPLES           HF cache (Narsil/asr_dummy) - en+es+hi (mlk.flac, 4.flac, hindi.ogg)"
	@echo "AUDIO             $${AUDIO:-<built-in samples>}"
	@echo "TEXT              $${TEXT:-<stdin or built-in none>}"
	@echo "OUTPUT            $${OUTPUT:-tts.wav}"

# Idempotent venv: uv sync only when the stamp is missing or older than
# pyproject/uv.lock. The stamp is touched after sync so a warm .venv does not
# re-sync on every make asr / make test (uv sync does not refresh
# .venv/bin/python, so depending on that file re-syncs every time).
$(VENV)/.stamp: pyproject.toml uv.lock
	@env -u VIRTUAL_ENV uv sync
	@touch $(VENV)/.stamp

venv: $(VENV)/.stamp ## Create the local .venv and install deps with uv (idempotent)

# CUDA venv for the A10 GPU, separate from the CPU .venv. Same pinned versions
# as pyproject.toml but torch/torchaudio from the cu121 wheel index (compatible
# with driver 610 / nvcc 13.3). Not synced from pyproject.toml: that file pins
# the CPU wheel index by design. Idempotent via a stamp, like $(VENV)/.stamp.
$(VENV_GPU)/.stamp:
	@uv venv $(VENV_GPU) --python 3.10
	@uv pip install --python $(PYGPU) --index-url https://download.pytorch.org/whl/cu121 \
	    torch==2.4.1 torchaudio==2.4.1
	@uv pip install --python $(PYGPU) \
	    transformers==4.44.2 soundfile==0.12.1 "numpy<2" sentencepiece==0.2.0 hqq jiwer datasets
	@touch $(VENV_GPU)/.stamp

gpu-venv: $(VENV_GPU)/.stamp ## Create the CUDA .venv-gpu for A10 GPU runs (small model)

# ONNX export/eval venv (CPU). Pinned against transformers 4.44.2 / torch 2.4.1 so the
# exported graph matches the HQQ compute path. Stamp depends on the Makefile so a pin
# change rebuilds the env. torch/torchaudio from the CPU wheel index (like the main .venv).
$(VENV_ONNX)/.stamp: Makefile
	@test -d $(VENV_ONNX) || uv venv $(VENV_ONNX) --python 3.10
	@uv pip install --python $(PYONNX) --index-url https://download.pytorch.org/whl/cpu \
	    torch==2.4.1 torchaudio==2.4.1
	@uv pip install --python $(PYONNX) \
	    transformers==4.44.2 soundfile==0.12.1 "numpy<2" sentencepiece==0.2.0 \
	    hqq==0.2.8.post1 jiwer datasets onnx==1.16.2 onnxruntime==1.19.2 \
	    optimum==2.0.0 "optimum-onnx[onnxruntime]==0.0.3" "pytest>=9.1.1"
	@touch $(VENV_ONNX)/.stamp

onnx-venv: $(VENV_ONNX)/.stamp ## Create the CPU .venv-onnx for ONNX export/eval (Path B)

samples: ## Warm the HF sample cache for the default set - idempotent, no re-download
	@$(PY) -c "from huggingface_hub import hf_hub_download; [hf_hub_download('Narsil/asr_dummy', f, repo_type='dataset') for f in ('mlk.flac','4.flac','hindi.ogg')]" && echo "samples cache warm"

asr: $(VENV)/.stamp ## Run the ASR test: transcribe the default multilingual samples (en+es+hi)
	@MODEL_ASR=$(MODEL_ASR) QUANT=$(QUANT) $(PY) transcribe.py

en: $(VENV)/.stamp ## Translate text to English (stdin/TEXT=; JSON to stdout, jq -r .text for make tts)
	@TARGET_LANG=$(TARGET_LANG) MODEL_TRANSLATE=$(MODEL_TRANSLATE) $(PY) translate.py

tts: $(VENV)/.stamp ## Synthesize speech from text (stdin/TEXT=; writes tts.wav, OUTPUT= to override)
	@MODEL_TTS=$(MODEL_TTS) $(PY) tts.py

quantize: $(VENV)/.stamp ## Quantize MODEL_ASR with HQQ 4-bit -> HQQ_OUT (local dir)
	@MODEL_ASR=$(MODEL_ASR) HQQ_OUT=$(HQQ_OUT) HQQ_GROUP=$(HQQ_GROUP) HQQ_AXIS=$(HQQ_AXIS) \
	 HQQ_8BIT_PATTERNS=$(HQQ_8BIT_PATTERNS) HQQ_8BIT_NBITS=$(HQQ_8BIT_NBITS) $(PY) quantize.py

push: $(VENV)/.stamp ## Quantize and upload HQQ_OUT to HQQ_REPO (needs HF_TOKEN_WRITE from ~/.api_keys)
	@set -a; . ~/.api_keys 2>/dev/null; set +a; \
	 MODEL_ASR=$(MODEL_ASR) HQQ_OUT=$(HQQ_OUT) HQQ_GROUP=$(HQQ_GROUP) HQQ_AXIS=$(HQQ_AXIS) \
	 HQQ_8BIT_PATTERNS=$(HQQ_8BIT_PATTERNS) HQQ_8BIT_NBITS=$(HQQ_8BIT_NBITS) HQQ_REPO=$(HQQ_REPO) PUSH=1 $(PY) quantize.py

eval-baseline: $(VENV)/.stamp ## Measure baseline WER (fp32 MODEL_ASR) on EVAL_DATASET/EVAL_CONFIG/EVAL_SPLIT (EVAL_LIMIT)
	@MODEL_ASR=$(MODEL_ASR) EVAL_DATASET=$(EVAL_DATASET) EVAL_CONFIG=$(EVAL_CONFIG) \
	 EVAL_SPLIT=$(EVAL_SPLIT) EVAL_LIMIT=$(EVAL_LIMIT) EVAL_OUT=eval_baseline.json $(PY) eval_wer.py

eval-hqq: $(VENV)/.stamp ## Measure HQQ WER (QUANT=hqq MODEL_ASR=HQQ_REPO) on EVAL_DATASET/EVAL_CONFIG/EVAL_SPLIT (EVAL_LIMIT)
	@QUANT=hqq MODEL_ASR=$(HQQ_REPO) EVAL_DATASET=$(EVAL_DATASET) EVAL_CONFIG=$(EVAL_CONFIG) \
	 EVAL_SPLIT=$(EVAL_SPLIT) EVAL_LIMIT=$(EVAL_LIMIT) EVAL_OUT=eval_hqq.json $(PY) eval_wer.py

# ONNX (Path B) targets. Runs in .venv-onnx (optimum/onnxruntime). Export reads the
# HQQ weights from HQQ_REPO (HF repo id -> default HF cache) and writes the 3 .onnx
# + config/processor into ONNX_OUT (build/). eval-onnx points MODEL_ASR at ONNX_OUT.
# Per-flavor: override HQQ_REPO + ONNX_OUT + EVAL_OUT + HQQ_REFERENCE_MANIFEST.
onnx: $(VENV_ONNX)/.stamp ## Export HQQ_REPO (HF) to ONNX (3 graphs) -> ONNX_OUT (.venv-onnx)
	@mkdir -p $(dir $(ONNX_OUT))
	@HQQ_OUT=$(HQQ_REPO) ONNX_OUT=$(ONNX_OUT) $(PYONNX) export_onnx.py

# Per-target EVAL_OUT defaults; override on the command line for per-flavor runs
# (make hqq-reference EVAL_OUT=build/hqq_reference_base.json, etc.).
hqq-reference: EVAL_OUT = $(BUILD)/hqq_reference.json
hqq-reference: $(VENV_ONNX)/.stamp ## Write the full 100-sample HQQ reference manifest (the exact-text gate oracle)
	@mkdir -p $(dir $(EVAL_OUT))
	@QUANT=hqq MODEL_ASR=$(HQQ_REPO) EVAL_DATASET=$(EVAL_DATASET) EVAL_CONFIG=$(EVAL_CONFIG) \
	 EVAL_SPLIT=$(EVAL_SPLIT) EVAL_LIMIT=$(EVAL_LIMIT) EVAL_OUT=$(EVAL_OUT) \
	 HQQ_REFERENCE=1 $(PYONNX) eval_wer.py

eval-onnx: EVAL_OUT = $(BUILD)/eval_onnx.json
eval-onnx: HQQ_REFERENCE_MANIFEST = $(BUILD)/hqq_reference.json
eval-onnx: $(VENV_ONNX)/.stamp ## Measure ONNX WER + exact-text gate vs the manifest (QUANT=onnx MODEL_ASR=ONNX_OUT)
	@mkdir -p $(dir $(EVAL_OUT))
	@QUANT=onnx MODEL_ASR=$(ONNX_OUT) EVAL_DATASET=$(EVAL_DATASET) EVAL_CONFIG=$(EVAL_CONFIG) \
	 EVAL_SPLIT=$(EVAL_SPLIT) EVAL_LIMIT=$(EVAL_LIMIT) EVAL_OUT=$(EVAL_OUT) \
	 HQQ_REFERENCE_MANIFEST=$(HQQ_REFERENCE_MANIFEST) $(PYONNX) eval_wer.py

test: $(VENV)/.stamp ## Run the fast unit tests (no model load, no network)
	@$(PY) -m pytest

test-integration: $(VENV)/.stamp ## Run the integration tests (loads the real Whisper/MarianMT/VITS models)
	@$(PY) -m pytest -m integration

clean: ## Remove Python bytecode cache (__pycache__, *.pyc) and the build/ artifact dir; keeps .venv
	@-find . -path ./.venv -prune -o -path ./.venv-onnx -prune -o -path ./.venv-gpu -prune -o \( -name '__pycache__' -o -name '*.pyc' \) -exec rm -rf {} +
	@rm -rf $(BUILD)
	@echo "cleaned pyc noise and build/"

clean-all: clean ## Also remove the local .venv (HF cache is left untouched)
	rm -rf $(VENV)