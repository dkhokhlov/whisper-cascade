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

.PHONY: help info venv samples asr en tts test test-integration clean clean-all

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
	@printf '  %-52s %s\n' 'make test' 'fast unit tests'
	@echo ""
	@echo "Pipeline (foreign speech -> English speech; each stage prints JSON, jq extracts text):"
	@echo "  make asr AUDIO=hf://datasets/Narsil/asr_dummy/4.flac | jq -r '.[].text' | make en | jq -r '.text' | make tts OUTPUT=4_en.wav"
	@echo "  make asr AUDIO=hf://datasets/Narsil/asr_dummy/4.flac | tee out.txt | jq -r '.[].text' | make en | tee -a out.txt | jq -r '.text' | make tts OUTPUT=4_en.wav >> out.txt && cat out.txt"
	@echo ""

info: ## Show current config and status
	@echo "MODEL_ASR         $(MODEL_ASR)"
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
	@MODEL_ASR=$(MODEL_ASR) $(PY) transcribe.py

en: $(VENV)/.stamp ## Translate text to English (stdin/TEXT=; JSON to stdout, jq -r .text for make tts)
	@TARGET_LANG=$(TARGET_LANG) MODEL_TRANSLATE=$(MODEL_TRANSLATE) $(PY) translate.py

tts: $(VENV)/.stamp ## Synthesize speech from text (stdin/TEXT=; writes tts.wav, OUTPUT= to override)
	@MODEL_TTS=$(MODEL_TTS) $(PY) tts.py

test: $(VENV)/.stamp ## Run the fast unit tests (no model load, no network)
	@$(PY) -m pytest

test-integration: $(VENV)/.stamp ## Run the integration tests (loads the real Whisper/MarianMT/VITS models)
	@$(PY) -m pytest -m integration

clean: ## Remove Python bytecode cache (__pycache__, *.pyc); keeps .venv
	@-find . -path ./.venv -prune -o \( -name '__pycache__' -o -name '*.pyc' \) -exec rm -rf {} +
	@echo "cleaned pyc noise"

clean-all: clean ## Also remove the local .venv (HF cache is left untouched)
	rm -rf $(VENV)