#!/usr/bin/env bash
# Download public-domain English speech clips (FLAC) into the samples directory.
# Source: Hugging Face dataset Narsil/asr_dummy.
#
# The whisper pipeline reads FLAC directly and resamples the audio to 16 kHz
# internally, so no conversion to WAV is necessary. The container still accepts
# WAV (and other formats) as input from the user.

set -euo pipefail

SAMPLES_DIR="${SAMPLES_DIR:-./samples}"
mkdir -p "${SAMPLES_DIR}"

BASE_URL="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main"
# English: mlk, 1, 2. Multilingual: 4 (Spanish), hindi (Hindi, OGG).
FILES=(mlk.flac 1.flac 2.flac 4.flac hindi.ogg)

for f in "${FILES[@]}"; do
    echo "Downloading ${f} ..."
    curl -fsSL "${BASE_URL}/${f}" -o "${SAMPLES_DIR}/${f}"
done

echo "Samples ready in ${SAMPLES_DIR}:"
ls -1 "${SAMPLES_DIR}"