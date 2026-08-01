VENV  := .venv
PY    := $(VENV)/bin/python
# Whisper model (override: make asr MODEL=openai/whisper-tiny.en). The default
# whisper-tiny is multilingual and transcribes the EN/ES/HI default samples.
MODEL ?= openai/whisper-tiny
# Custom audio input: make asr AUDIO=file.wav | dir/ | '*.flac' (space-safe).
export AUDIO

.PHONY: help info venv samples asr test test-integration clean clean-all

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make <target> [VAR=value]\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  %-18s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@printf '  %-52s %s\n' 'make asr' 'default samples (en+es+hi, whisper-tiny)'
	@printf '  %-52s %s\n' 'make asr MODEL=openai/whisper-tiny.en' 'English-only model'
	@printf '  %-52s %s\n' 'make asr AUDIO=hf://datasets/Narsil/asr_dummy/1.flac' 'one HF Hub file'
	@printf '  %-52s %s\n' 'make asr AUDIO=./clips/' 'a directory'
	@printf '  %-52s %s\n' "make asr AUDIO='*.flac'" 'a glob'
	@printf '  %-52s %s\n' 'make test' 'fast unit tests'

info: ## Show current config and status
	@echo "MODEL       $(MODEL)"
	@echo "VENV        $(VENV) - $$([ -d $(VENV) ] && echo present || echo missing)"
	@echo "SAMPLES     HF cache (Narsil/asr_dummy) - en+es+hi (mlk.flac, 4.flac, hindi.ogg)"
	@echo "AUDIO        $${AUDIO:-<built-in samples>}"

venv: ## Create the local .venv and install deps with uv
	@env -u VIRTUAL_ENV uv sync

samples: ## Warm the HF sample cache for the default set - idempotent, no re-download
	@$(PY) -c "from huggingface_hub import hf_hub_download; [hf_hub_download('Narsil/asr_dummy', f, repo_type='dataset') for f in ('mlk.flac','4.flac','hindi.ogg')]" && echo "samples cache warm"

asr: venv ## Run the ASR test: transcribe the default multilingual samples (en+es+hi)
	@MODEL=$(MODEL) $(PY) transcribe.py

test: venv ## Run the fast unit tests (no model load, no network)
	@$(PY) -m pytest

test-integration: venv ## Run the integration test (loads the real Whisper model)
	@$(PY) -m pytest -m integration

clean: ## Remove Python bytecode cache (__pycache__, *.pyc); keeps .venv
	@-find . -path ./.venv -prune -o \( -name '__pycache__' -o -name '*.pyc' \) -exec rm -rf {} +
	@echo "cleaned pyc noise"

clean-all: clean ## Also remove the local .venv (HF cache is left untouched)
	rm -rf $(VENV)