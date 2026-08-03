VENV  := .venv
PY    := $(VENV)/bin/python
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
# HQQ quantization output dir, HF repo, and group_size (make quantize / push).
HQQ_OUT ?= whisper-tiny-hqq-4bit
HQQ_REPO ?= dkhokhlov/whisper-tiny-hqq-4bit
HQQ_GROUP ?= 64
# WER eval subset size on google/fleurs en_us (make eval-baseline / eval-hqq).
EVAL_LIMIT ?= 100

.PHONY: help info venv samples asr en tts quantize push eval-baseline eval-hqq test test-integration clean clean-all

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
	@printf '  %-52s %s\n' 'make quantize' 'quantize whisper-tiny -> HQQ_OUT (local)'
	@printf '  %-52s %s\n' 'make push' 'quantize + upload to HQQ_REPO (needs HF_TOKEN_WRITE)'
	@printf '  %-52s %s\n' 'make eval-baseline' 'WER of fp32 MODEL_ASR on fleurs en_us'
	@printf '  %-52s %s\n' 'make eval-hqq' 'WER of HQQ MODEL_ASR on fleurs en_us'
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

samples: ## Warm the HF sample cache for the default set - idempotent, no re-download
	@$(PY) -c "from huggingface_hub import hf_hub_download; [hf_hub_download('Narsil/asr_dummy', f, repo_type='dataset') for f in ('mlk.flac','4.flac','hindi.ogg')]" && echo "samples cache warm"

asr: $(VENV)/.stamp ## Run the ASR test: transcribe the default multilingual samples (en+es+hi)
	@MODEL_ASR=$(MODEL_ASR) QUANT=$(QUANT) $(PY) transcribe.py

en: $(VENV)/.stamp ## Translate text to English (stdin/TEXT=; JSON to stdout, jq -r .text for make tts)
	@TARGET_LANG=$(TARGET_LANG) MODEL_TRANSLATE=$(MODEL_TRANSLATE) $(PY) translate.py

tts: $(VENV)/.stamp ## Synthesize speech from text (stdin/TEXT=; writes tts.wav, OUTPUT= to override)
	@MODEL_TTS=$(MODEL_TTS) $(PY) tts.py

quantize: $(VENV)/.stamp ## Quantize MODEL_ASR with HQQ 4-bit -> HQQ_OUT (local dir)
	@MODEL_ASR=$(MODEL_ASR) HQQ_OUT=$(HQQ_OUT) HQQ_GROUP=$(HQQ_GROUP) $(PY) quantize.py

push: $(VENV)/.stamp ## Quantize and upload HQQ_OUT to HQQ_REPO (needs HF_TOKEN_WRITE from ~/.api_keys)
	@set -a; . ~/.api_keys 2>/dev/null; set +a; \
	 MODEL_ASR=$(MODEL_ASR) HQQ_OUT=$(HQQ_OUT) HQQ_GROUP=$(HQQ_GROUP) HQQ_REPO=$(HQQ_REPO) PUSH=1 $(PY) quantize.py

eval-baseline: $(VENV)/.stamp ## Measure baseline WER (fp32 MODEL_ASR) on fleurs en_us (EVAL_LIMIT)
	@MODEL_ASR=$(MODEL_ASR) EVAL_LIMIT=$(EVAL_LIMIT) EVAL_OUT=eval_baseline.json $(PY) eval_wer.py

eval-hqq: $(VENV)/.stamp ## Measure HQQ WER (QUANT=hqq MODEL_ASR=HQQ_REPO) on fleurs en_us (EVAL_LIMIT)
	@QUANT=hqq MODEL_ASR=$(HQQ_REPO) EVAL_LIMIT=$(EVAL_LIMIT) EVAL_OUT=eval_hqq.json $(PY) eval_wer.py

test: $(VENV)/.stamp ## Run the fast unit tests (no model load, no network)
	@$(PY) -m pytest

test-integration: $(VENV)/.stamp ## Run the integration tests (loads the real Whisper/MarianMT/VITS models)
	@$(PY) -m pytest -m integration

clean: ## Remove Python bytecode cache (__pycache__, *.pyc); keeps .venv
	@-find . -path ./.venv -prune -o \( -name '__pycache__' -o -name '*.pyc' \) -exec rm -rf {} +
	@echo "cleaned pyc noise"

clean-all: clean ## Also remove the local .venv (HF cache is left untouched)
	rm -rf $(VENV)