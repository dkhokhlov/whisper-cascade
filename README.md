# whisper-cascade

A minimal, CPU-only speech-to-text tool, with two text helpers that compose
into a UNIX pipeline. `make asr` transcribes audio files (WAV / FLAC / OGG /
MP3) and prints JSON. `make en` translates text to English. `make tts`
synthesizes speech from text. Chained together they turn foreign speech into
English speech:

```bash
make asr AUDIO=hf://datasets/Narsil/asr_dummy/4.flac | jq -r '.[].text' | make en | make tts OUTPUT=4_en.wav
```

The ASR step uses OpenAI's `whisper-tiny` (multilingual); the translate step
uses a MarianMT model; the TTS step uses a VITS model — all through the
Hugging Face `transformers` library.

No Docker, no ffmpeg, no GPU. Dependencies are managed with [uv](https://docs.astral.sh/uv/).
The models and the test samples are cached through the Hugging Face cache
(`huggingface_hub`) — the same mechanism used for models and datasets — so
nothing is re-downloaded after the first run.

## Quick start

Requires Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
make venv     # create .venv and install deps (torch/torchaudio from the CPU index)
make asr      # transcribe the built-in multilingual samples (en + es + hi)
make en TEXT="Hola, ¿cómo estás?"          # -> English text on stdout
make tts TEXT="Hello world" OUTPUT=hi.wav  # -> hi.wav
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
| `make asr` | Transcribe audio to JSON (default multilingual samples, or `AUDIO=`). |
| `make en` | Translate text to English (stdin/`TEXT=`; plain text to stdout). |
| `make tts` | Synthesize speech from text (stdin/`TEXT=`; writes `tts.wav`, `OUTPUT=` to override). |
| `make test` | Run the fast unit tests (no model load, no network). |
| `make test-integration` | Run the integration tests (load the real Whisper/MarianMT/VITS models). |
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

The default ASR model is `openai/whisper-tiny` (multilingual) — the
`MODEL_ASR` variable. Override it for English-only deployments or a bigger
model:

```bash
make asr MODEL_ASR=openai/whisper-tiny.en   # English-only, slightly better English
make asr MODEL_ASR=openai/whisper-base     # a bigger model
MODEL_ASR=openai/whisper-base .venv/bin/python transcribe.py my.wav
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

## Translate (make en)

`make en` reads text (the `TEXT` env var, a file argument, or stdin) and prints
the English translation to stdout as plain text, so it pipes into `make tts`.
A one-line summary goes to stderr. The model is a MarianMT model selected by
`MODEL_TRANSLATE` (default `Helsinki-NLP/opus-mt-mul-en`, multilingual to
English).

```bash
make en TEXT="Hola, ¿cómo estás?"                       # -> "Hey, how are you?"
echo "Ich heiße Max." | make en                         # -> "My name is Max."
MODEL_TRANSLATE=Helsinki-NLP/opus-mt-es-en make en TEXT="Hola"   # Spanish source
```

The target is named by the target language: `make en` runs `translate.py`.
Adding `make es` (translate to Spanish) later is one Make target that sets
`MODEL_TRANSLATE` to a `-> Spanish` model — no code change. For a known single
source language, a pair-specific model (e.g. `opus-mt-es-en`) gives better
quality than the multilingual default.

## TTS (make tts)

`make tts` reads text (the `TEXT` env var, a file argument, or stdin) and
writes the synthesized waveform to `OUTPUT` (default `tts.wav`). A JSON summary
goes to stdout (`output`, `model`, `text`, and a `stats` block). The model is a
VITS model selected by `MODEL_TTS` (default `facebook/mms-tts-eng`, English).
transformers 4.44.2 has no text-to-speech pipeline, so this calls the model
directly (`AutoTokenizer` + `VitsModel`) and writes `out.waveform` with
`soundfile` at `model.config.sampling_rate`.

```bash
make tts TEXT="Hello world"                  # -> tts.wav
make tts TEXT="Hello" OUTPUT=hi.wav          # -> hi.wav
MODEL_TTS=facebook/mms-tts-spa make tts TEXT="Hola"   # another MMS language
```

## Pipeline

The three targets compose as a UNIX pipeline. `make asr` prints JSON; `jq`
extracts the `text` fields (`jq -r '.[].text'`); `make en` translates the text
to English; `make tts` synthesizes English speech. The tools themselves do not
require `jq` — it is only the bridge for this example.

```bash
# Spanish speech -> Spanish text -> English text -> English speech
make asr AUDIO=hf://datasets/Narsil/asr_dummy/4.flac | jq -r '.[].text' | make en | make tts OUTPUT=4_en.wav
```

For the default sample set (three files), `jq` emits one line per file; pass a
single file (as above, via `AUDIO=`) for a one-shot pipeline, or loop over the
files in bash for per-file output.

## Notes

- CPU only. No GPU dependency.
- Audio is decoded with `soundfile` (no ffmpeg required). WAV, FLAC, OGG, and
  MP3 are accepted.
- The ASR pipeline resamples each input to 16 kHz internally, so no
  pre-conversion is needed.
- `make en` needs `sentencepiece` (MarianMT tokenizers); it is in the deps.
- Transformers warnings are silenced; ASR stdout is clean JSON, `make en`
  stdout is plain text, `make tts` stdout is a JSON summary. stderr stays quiet
  except the summary lines.