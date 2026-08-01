VENV  := .venv
PY    := $(VENV)/bin/python
# English test model (override: make en MODEL=openai/whisper-base).
MODEL ?= openai/whisper-tiny.en
# Multilingual test model (override: make ml MODEL=openai/whisper-base).
# whisper-tiny is multilingual; the .en model cannot transcribe the ES/HI samples.
ML_MODEL ?= openai/whisper-tiny
# Custom audio input: make en AUDIO=file.wav | dir/ | '*.flac' (space-safe).
export AUDIO

.PHONY: help info venv samples en ml clean clean-all

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make <target> [VAR=value]\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  %-12s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

info: ## Show current config and status
	@echo "MODEL       $(MODEL) (en) / $(ML_MODEL) (ml)"
	@echo "VENV        $(VENV) - $$([ -d $(VENV) ] && echo present || echo missing)"
	@echo "SAMPLES     HF cache (Narsil/asr_dummy) - $${SAMPLES_SET:-en} set"
	@echo "AUDIO        $${AUDIO:-<built-in samples>}"

venv: ## Create the local .venv and install deps with uv
	@env -u VIRTUAL_ENV uv sync

samples: ## Warm the HF sample cache (Narsil/asr_dummy) - idempotent, no re-download
	@$(PY) -c "from huggingface_hub import hf_hub_download; [hf_hub_download('Narsil/asr_dummy', f, repo_type='dataset') for f in ('mlk.flac','1.flac','2.flac','4.flac','hindi.ogg')]" && echo "samples cache warm"

en: venv ## Run the English test (override model: make en MODEL=...; custom audio: make en AUDIO=file.wav)
	@MODEL=$(MODEL) $(PY) transcribe.py

# Target-specific default so `make ml` uses whisper-tiny, but `make ml MODEL=...`
# still overrides (command-line beats target-specific).
ml: MODEL = $(ML_MODEL)
ml: venv ## Run the multilingual test (override model: make ml MODEL=...; custom audio: make ml AUDIO=file.wav)
	@SAMPLES_SET=ml MODEL=$(MODEL) $(PY) transcribe.py

clean: ## Remove Python bytecode cache (__pycache__, *.pyc); keeps .venv
	@-find . -path ./.venv -prune -o \( -name '__pycache__' -o -name '*.pyc' \) -exec rm -rf {} +
	@echo "cleaned pyc noise"

clean-all: clean ## Also remove the local .venv (HF cache is left untouched)
	rm -rf $(VENV)