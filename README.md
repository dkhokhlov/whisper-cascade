# wisper-asr

A minimal, CPU-only speech-to-text tool. It transcribes audio files
(WAV / FLAC / OGG / MP3) and prints the text to stdout as JSON, using
OpenAI's `whisper-tiny.en` model through the Hugging Face `transformers`
pipeline.

No Docker, no ffmpeg, no GPU. Dependencies are managed with [uv](https://docs.astral.sh/uv/).
Both the model and the test samples are cached through the Hugging Face
cache (`huggingface_hub`) — the same mechanism used for models and datasets —
so nothing is re-downloaded after the first run.

## Quick start

Requires Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
make venv     # create .venv and install deps (torch/torchaudio from the CPU index)
make en       # transcribe the built-in English samples
```

Expected output on stdout (JSON):

```json
[
  {
    "file": "mlk.flac",
    "text": "I have a dream that one day this nation will rise up ...",
    "model": "openai/whisper-tiny.en",
    "stats": { "duration_s": 13.0, "elapsed_s": 0.55, "rtf": 0.043, "tokens": 20, "words": 20, "chars": 91 }
  }
]
```

A one-line summary goes to stderr:

```
[stats] files=3 audio=26.71s elapsed=1.32s rtf=0.049 tokens=72 words=53
```

Transcribe your own file:

```bash
.venv/bin/python transcribe.py path/to/audio.wav
```

Multilingual samples (uses `whisper-tiny`):

```bash
make ml
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
| `make clean-all` | Also remove the local `.venv` (HF cache is left untouched). |

## Output

The script prints one JSON array to stdout. Each element has `file`, `text`,
`model`, and a `stats` block. See the Quick start for the shape.

Stats (ASR-style):

- `duration_s` — audio length in seconds.
- `elapsed_s` — transcription wall-clock time in seconds.
- `rtf` — real-time factor (`elapsed_s / duration_s`); below 1 means faster than real time.
- `tokens` — output token count (via the model tokenizer).
- `words` / `chars` — output word and character counts.

When a file fails, its element has the key `error` instead of `text` and
`stats`, and the script exits with status 1. The other files are still
processed.

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
- Transformers warnings are silenced; stdout is clean JSON, stderr stays quiet
  except the stats summary.