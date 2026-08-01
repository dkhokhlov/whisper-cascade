# wisper_test

Transcribe audio to text with the `openai/whisper-tiny.en` model and print the
result to stdout as JSON. CPU-only. Local `uv`-managed environment.

The tool uses the Hugging Face `transformers` automatic-speech-recognition
pipeline. Both the model and the samples are cached through the Hugging Face
cache (`huggingface_hub`), the same mechanism used for models and datasets, so
nothing is re-downloaded after the first run.

## Setup

Requires Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
make venv     # create .venv and install deps (torch/torchaudio from the CPU index)
```

## Make targets

Run `make` (no target) to print this help.

| Target | Description |
| --- | --- |
| `make info` | Show current config and status. |
| `make venv` | Create the local `.venv` with uv. |
| `make samples` | Warm the HF sample cache (idempotent, no re-download). |
| `make en` | Run the English test: transcribe the EN samples with the `.venv`. |
| `make ml` | Run the multilingual test: transcribe the ES+HI samples with `whisper-tiny`. |
| `make clean` | Remove Python bytecode cache (`__pycache__`, `*.pyc`); keeps `.venv`. |
| `make clean-all` | Also remove `.venv` and this repo's audio dataset (`Narsil/asr_dummy`) from the HF cache. The model cache is left untouched. |

## Run

English samples (default model `whisper-tiny.en`):

```bash
make en
```

Multilingual samples (uses `whisper-tiny`):

```bash
make ml
```

Transcribe your own file(s):

```bash
.venv/bin/python transcribe.py path/to/audio.wav path/to/other.flac
```

## Output

The script prints one JSON array to stdout. Each element has the keys `file`,
`text`, and `model`:

```json
[
  {"file": "mlk.flac", "text": "...", "model": "openai/whisper-tiny.en"}
]
```

When a file fails, its element has the key `error` instead of `text`, and the
script exits with status 1. The other files are still processed.

## Samples

Samples come from the Hugging Face dataset `Narsil/asr_dummy` and are resolved
from the HF cache on first use (`~/.cache/huggingface` by default):

- English (`make en`): `mlk.flac`, `1.flac`, `2.flac`.
- Multilingual (`make ml`): `4.flac` (Spanish), `hindi.ogg` (Hindi).

The default model is English-only, so `make en` is the coherent default. The
multilingual samples are used only under a multilingual model, which `make ml`
selects automatically.

## Model override

Override the model with the `MODEL` variable. The model id appears in each JSON
element as `"model"`.

```bash
make en MODEL=openai/whisper-base      # English test with a different model
make ml                                # multilingual test (always whisper-tiny)
MODEL=openai/whisper-base .venv/bin/python transcribe.py my.wav
```

The `SAMPLES_SET` env var selects the default sample set when no file args are
given: `en` (default) or `ml`.

## Notes

- CPU only. No GPU dependency.
- Audio is decoded with `soundfile` (no ffmpeg required). WAV, FLAC, OGG, and
  MP3 are accepted.
- The pipeline resamples each input to 16 kHz internally, so no pre-conversion
  is needed.
- Transformers warnings are silenced; stdout is clean JSON, stderr stays quiet.