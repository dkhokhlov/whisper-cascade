IMAGE   := wisper-test
VENV    := .venv
PY      := $(VENV)/bin/python
SAMPLES := samples/mlk.flac samples/1.flac samples/2.flac
ML_SAMPLES := samples/4.flac samples/hindi.ogg
# Override the model: make build MODEL=openai/whisper-base  (and make local MODEL=...)
MODEL   ?= openai/whisper-tiny.en

.PHONY: help info build run run-file offline validate venv samples local ml clean image-rm

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make <target> [VAR=value]\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  %-12s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

info: ## Show current config and status
	@echo "IMAGE       $(IMAGE)"
	@echo "MODEL       $(MODEL)"
	@echo "VENV        $(VENV) - $$([ -d $(VENV) ] && echo present || echo missing)"
	@echo "SAMPLES     samples/ - $$(ls samples/*.flac samples/*.ogg samples/*.wav samples/*.mp3 2>/dev/null | wc -l) audio file(s)"
	@echo "IMAGE built - $$(docker image inspect $(IMAGE) >/dev/null 2>&1 && echo yes || echo no)"

build: ## Build the docker image (override model: make build MODEL=openai/whisper-base)
	docker build --build-arg MODEL=$(MODEL) -t $(IMAGE) .

run: ## Transcribe the baked samples in the container
	docker run --rm $(IMAGE)

run-file: ## Transcribe a host file: make run-file FILE=path/to/audio.wav
	@test -n "$(FILE)" || { echo "Usage: make run-file FILE=path/to/audio.wav"; exit 1; }
	docker run --rm -v "$(abspath $(FILE)):/work/input" $(IMAGE) /work/input

offline: ## Run the container with no network (model is baked in)
	docker run --rm --network=none $(IMAGE)

validate: ## Validate the container JSON output with python3 -m json.tool
	docker run --rm $(IMAGE) 2>/dev/null | python3 -m json.tool >/dev/null && echo "VALID JSON"

venv: ## Create the local .venv and install deps with uv
	env -u VIRTUAL_ENV uv sync

samples: ## Download the FLAC test samples into ./samples
	SAMPLES_DIR=samples PATH="$(CURDIR)/$(VENV)/bin:$$PATH" bash download_samples.sh

local: venv samples ## Transcribe the local English samples with the .venv (override model: make local MODEL=openai/whisper-base)
	MODEL=$(MODEL) $(PY) transcribe.py $(SAMPLES)

ml: venv samples ## Transcribe the multilingual samples (ES+HI) with whisper-tiny locally
	MODEL=openai/whisper-tiny $(PY) transcribe.py $(ML_SAMPLES)

clean: ## Remove the local .venv and samples
	rm -rf $(VENV) samples

image-rm: ## Remove the built docker image
	-docker rmi $(IMAGE)