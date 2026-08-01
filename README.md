# wisper-asr

A minimal, CPU-only speech-to-text tool. It transcribes audio files
(WAV / FLAC / OGG / MP3) and prints the text to stdout as JSON, using OpenAI's
`whisper-tiny` model (multilingual) through the Hugging Face `transformers`
pipeline.

No Docker, no ffmpeg, no GPU. Dependencies are managed with [uv](https://docs.astral.sh/uv/).
Both the model and the test samples are cached through the Hugging Face
cache (`huggingface_hub`) — the same mechanism used for models and datasets —
so nothing is re-downloaded after the first run.

## Quick start

Requires Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
make venv     # create .venv and install deps (torch/torchaudio from the CPU index)
make asr      # transcribe the built-in multilingual samples (en + es + hi)
```

Expected output on stdout (JSON):

```json
[
  {
    "file": "mlk.flac",
    "text": "I have a dream that one day this nation will rise up ...",
    "model": "openai/whisper-tiny",
    "stats": { "duration_s": 13.0, "elapsed_s": 0.75, "rtf": 0.057, "tokens": 24, "words": 20, "chars": 95 }
  }
]
```

A one-line summary goes to stderr:

```
[stats] files=3 audio=23.71s elapsed=1.64s rtf=0.069 tokens=68 words=43
```

Transcribe your own file:

```bash
.venv/bin/python transcribe.py path/to/audio.wav
```

Transcribe your own audio via the `AUDIO` env var (Hugging Face URL, file,
directory, or glob):

```bash
make asr AUDIO=path/to/audio.wav
make asr AUDIO='./clips/*.flac'   # every FLAC in clips/
```

## Make targets

Run `make` (no target) to print this help.

| Target | Description |
| --- | --- |
| `make info` | Show current config and status. |
| `make venv` | Create the local `.venv` with uv. |
| `make samples` | Warm the HF sample cache for the default set (idempotent, no re-download). |
| `make asr` | Run the ASR test: transcribe the default multilingual samples with the `.venv`. |
| `make test` | Run the fast unit tests (no model load, no network). |
| `make test-integration` | Run the integration test (loads the real Whisper model). |
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

The default sample set comes from the Hugging Face dataset `Narsil/asr_dummy`
and is resolved from the HF cache on first use (`~/.cache/huggingface` by
default):

- `mlk.flac` — English.
- `4.flac` — Spanish.
- `hindi.ogg` — Hindi.

One default set, transcribed with the multilingual `whisper-tiny`. (The
dataset has no German sample; pull one on demand via `AUDIO=hf://...` if you
need it.)

## Model

The default model is `openai/whisper-tiny` (multilingual) — one `MODEL`
variable. Override it for English-only deployments or a bigger model:

```bash
make asr MODEL=openai/whisper-tiny.en   # English-only, slightly better English
make asr MODEL=openai/whisper-base     # a bigger model
MODEL=openai/whisper-base .venv/bin/python transcribe.py my.wav
```

The model id appears in each JSON element as `"model"`.

## Custom audio (AUDIO env var)

Point the tool at your own audio instead of the built-in samples. `AUDIO`
accepts one or more whitespace-separated tokens; each can be a Hugging Face
URL, a file, a directory (its audio files are used, filtered by extension), or
a glob.

```bash
make asr AUDIO=hf://datasets/Narsil/asr_dummy/1.flac   # pull a file from the HF Hub
make asr AUDIO=file.wav
make asr AUDIO='clip1.wav clip2.flac'
make asr AUDIO=./clips/                # a directory
make asr AUDIO='./clips/*.flac'        # a glob
```

An `hf://` URL (`hf://datasets/<ns>/<repo>/<file>`, or `hf://models/...`) is
downloaded to the HF cache on first use and transcribed from there. The
`hf://` scheme is parsed by this tool — `hf_hub_download` takes `repo_id` +
`filename`, not a URL — so no CLI upgrade is needed. A bad or missing HF URL
becomes a per-file `error` element (the other files still process).

Without `AUDIO`, the built-in default sample set (en + es + hi) is used.

## Notes

- CPU only. No GPU dependency.
- Audio is decoded with `soundfile` (no ffmpeg required). WAV, FLAC, OGG, and
  MP3 are accepted.
- The pipeline resamples each input to 16 kHz internally, so no pre-conversion
  is needed.
- Transformers warnings are silenced; stdout is clean JSON, stderr stays quiet
  except the stats summary.