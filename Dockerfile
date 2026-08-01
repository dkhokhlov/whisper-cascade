FROM python:3.10-slim

# Model id. Override at build time: docker build --build-arg MODEL=openai/whisper-base
ARG MODEL=openai/whisper-tiny.en

# System tools: curl fetches the sample clips. Audio is decoded by soundfile
# (bundled libsndfile), so ffmpeg is not required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv at the same version as the host so the lock file resolves.
RUN pip install --no-cache-dir uv==0.11.14

# Keep the Hugging Face model cache and the uv venv inside the image.
ENV HF_HOME=/app/hf_cache
ENV PYTHONUNBUFFERED=1
ENV SAMPLES_DIR=/app/samples
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies from the lock file. Do not build the project package.
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project

# Put the venv on PATH so ENTRYPOINT uses it.
ENV PATH="/app/.venv/bin:$PATH"

# Bake the chosen model id into the image; transcribe.py reads this env var.
# Set here (after uv sync) so changing the model does not re-run uv sync.
ENV MODEL=${MODEL}

# Pre-cache the chosen model at build time so the image runs offline. Run this
# before COPY of the script so editing transcribe.py does not re-download it.
RUN python -c "import os; from transformers import pipeline; pipeline(task='automatic-speech-recognition', model=os.environ['MODEL'])"

# Copy the application and the sample downloader.
COPY transcribe.py download_samples.sh /app/

# Fetch the sample clips (FLAC) into the image.
RUN bash /app/download_samples.sh

# After the model is cached, forbid network access at runtime.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["python", "/app/transcribe.py"]
CMD ["/app/samples/mlk.flac", "/app/samples/1.flac", "/app/samples/2.flac"]