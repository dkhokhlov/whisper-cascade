VENV  := .venv
PY    := $(VENV)/bin/python
# Override the model: make en MODEL=openai/whisper-base
MODEL ?= openai/whisper-tiny.en

.PHONY: help info venv samples en ml clean clean-all

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make <target> [VAR=value]\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  %-12s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

info: ## Show current config and status
	@echo "MODEL       $(MODEL)"
	@echo "VENV        $(VENV) - $$([ -d $(VENV) ] && echo present || echo missing)"
	@echo "SAMPLES     HF cache (Narsil/asr_dummy) - $${SAMPLES_SET:-en} set"

venv: ## Create the local .venv and install deps with uv
	env -u VIRTUAL_ENV uv sync

samples: ## Warm the HF sample cache (Narsil/asr_dummy) - idempotent, no re-download
	$(PY) -c "from huggingface_hub import hf_hub_download; [hf_hub_download('Narsil/asr_dummy', f, repo_type='dataset') for f in ('mlk.flac','1.flac','2.flac','4.flac','hindi.ogg')]" && echo "samples cache warm"

en: venv ## Run the English test: transcribe the EN samples with the .venv (override model: make en MODEL=openai/whisper-base)
	MODEL=$(MODEL) $(PY) transcribe.py

ml: venv ## Run the multilingual test: transcribe the ES+HI samples with whisper-tiny locally
	SAMPLES_SET=ml MODEL=openai/whisper-tiny $(PY) transcribe.py

clean: ## Remove Python bytecode cache (__pycache__, *.pyc); keeps .venv
	@-find . -path ./.venv -prune -o \( -name '__pycache__' -o -name '*.pyc' \) -exec rm -rf {} +
	@echo "cleaned pyc noise"

clean-all: clean ## Remove .venv and this repo's audio dataset (Narsil/asr_dummy) from the HF cache
	@rm -rf "$${HF_HOME:-$$HOME/.cache/huggingface}/hub/datasets--Narsil--asr_dummy" && echo "removed HF dataset cache (Narsil/asr_dummy)"
	rm -rf $(VENV)