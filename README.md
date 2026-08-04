# whisper-cascade

A minimal, CPU-only speech-to-text tool, with two text helpers that compose
into a cascade speech translation pipeline.

The three stages:

```
             foreign speech (audio)
                        │
                        ▼
        ┌───────────────────────────────┐
        │              ASR              │
        │ Automatic Speech Recognition  │
        │      openai/whisper-tiny      │
        └───────────────┬───────────────┘
                        │ Spanish text
                        ▼
        ┌───────────────────────────────┐
        │              MT               │
        │      Machine Translation      │
        │  Helsinki-NLP/opus-mt-mul-en  │
        └───────────────┬───────────────┘
                        │ English text
                        ▼
        ┌───────────────────────────────┐
        │              TTS              │
        │        Text To Speech         │
        │     facebook/mms-tts-eng      │
        └───────────────┬───────────────┘
                        │
                        ▼
            English speech (out.wav)
```

`make asr` transcribes audio files (WAV / FLAC / OGG /
MP3) and prints JSON. `make en` translates text to English. `make tts`
synthesizes speech from text. Chained together they turn foreign speech into
English speech:

```bash
make asr AUDIO=hf://datasets/Narsil/asr_dummy/4.flac | jq -r '.[].text' | make en | jq -r '.text' | make tts OUTPUT=4_en.wav
```

The ASR step uses OpenAI's [`whisper-tiny`](https://huggingface.co/openai/whisper-tiny)
(multilingual); the translate step uses a MarianMT model
([`Helsinki-NLP/opus-mt-mul-en`](https://huggingface.co/Helsinki-NLP/opus-mt-mul-en));
the TTS step uses a VITS model
([`facebook/mms-tts-eng`](https://huggingface.co/facebook/mms-tts-eng)) — all
through the Hugging Face `transformers` library.

Per-model runtime weights (single weights file loaded):

| Model | Decoder type | Stored dtype | Loaded as | Loaded file | Size |
|---|---|---|---|---|---|
| [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | autoregressive (enc-dec) | float32 | float32 | `model.safetensors` | 151.06 MB |
| [`Helsinki-NLP/opus-mt-mul-en`](https://huggingface.co/Helsinki-NLP/opus-mt-mul-en) | autoregressive (enc-dec) | float32 (default) | float32 | `pytorch_model.bin` (no safetensors) | 310.39 MB |
| [`facebook/mms-tts-eng`](https://huggingface.co/facebook/mms-tts-eng) | non-autoregressive | float32 | float32 | `model.safetensors` | 145.23 MB |

Combined model weights → ~607 MB (151.06 + 310.39 + 145.23 = 606.68 MB)

No Docker, no ffmpeg, no GPU. Dependencies are managed with [uv](https://docs.astral.sh/uv/).
The models and the test samples are cached through the Hugging Face cache
(`huggingface_hub`) — the same mechanism used for models and datasets — so
nothing is re-downloaded after the first run.

## Quick start

Requires Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
make venv     # create .venv and install deps (torch/torchaudio from the CPU index)
make asr      # transcribe the built-in multilingual samples (en + es + hi)
make en TEXT="Hola, ¿cómo estás?"          # -> JSON with text + stats
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

For multi-file totals, sum the JSON with `jq` (e.g. `make asr | jq '[.[].stats.duration_s] | add'`).

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
| `make en` | Translate text to English (stdin/`TEXT=`; JSON with text + stats to stdout). |
| `make tts` | Synthesize speech from text (stdin/`TEXT=`; writes `tts.wav`, `OUTPUT=` to override). |
| `make quantize` | Quantize `MODEL_ASR` with HQQ 4-bit to `HQQ_OUT` (local dir). |
| `make push` | Quantize and upload `HQQ_OUT` to `HQQ_REPO` (needs `HF_TOKEN_WRITE`). |
| `make eval-baseline` | Measure baseline WER (fp32 `MODEL_ASR`) on `EVAL_DATASET`/`EVAL_CONFIG`/`EVAL_SPLIT` (`EVAL_LIMIT`). |
| `make eval-hqq` | Measure HQQ WER (`QUANT=hqq MODEL_ASR=HQQ_REPO`) on `EVAL_DATASET`/`EVAL_CONFIG`/`EVAL_SPLIT` (`EVAL_LIMIT`). |
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
`stats`, and the script exits with a non-zero status. Run directly, the
script returns 1; run through `make`, the exit code is 2 (make uses 2 for a
failed recipe). The other files are still processed.

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

`make en` reads text from the `TEXT` env var or stdin and prints
a JSON object to stdout with `text` (the English translation), `lang`
(`"en"`), `model`, and a `stats` block (`chars`, `words` of the input,
`out_chars`, `out_words` of the output, `elapsed_s`). Extract the text with
`jq -r '.text'` to pipe into `make tts`. The model is a MarianMT model selected
by `MODEL_TRANSLATE` (default `Helsinki-NLP/opus-mt-mul-en`, multilingual to
English).

```bash
make en TEXT="Hola, ¿cómo estás?" | jq -r '.text'        # -> "Hey, how are you?"
echo "Ich heiße Max." | make en | jq -r '.text'          # -> "My name is Max."
MODEL_TRANSLATE=Helsinki-NLP/opus-mt-es-en make en TEXT="Hola" | jq -r '.text'   # Spanish source
```

`make en` cannot take a file argument (make treats a bare word as a target).
To read a file directly, run the script: `.venv/bin/python translate.py
file.txt` (or `-` for stdin).

The target is named by the target language: `make en` runs `translate.py`.
Adding `make es` (translate to Spanish) later is one Make target that sets
`TARGET_LANG=es` and `MODEL_TRANSLATE` to a `-> Spanish` model — no code
change. For a known single source language, a pair-specific model (e.g.
`opus-mt-es-en`) gives better quality than the multilingual default.

## TTS (make tts)

`make tts` reads text from the `TEXT` env var or stdin and
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
extracts the `text` fields (`jq -r '.[].text'`); `make en` prints JSON (text +
stats); `jq -r '.text'` extracts the translation; `make tts` synthesizes
English speech. The tools themselves do not require `jq` — it is only the
bridge for this example.

```bash
# Spanish speech -> Spanish text -> English text -> English speech
make asr AUDIO=hf://datasets/Narsil/asr_dummy/4.flac | jq -r '.[].text' | make en | jq -r '.text' | make tts OUTPUT=4_en.wav
```

Run the pipeline with `set -o pipefail` so an upstream failure makes the whole
pipeline exit non-zero. Without it, a bad `AUDIO=` (an unmatched glob, a missing
file) lets `make asr` fail silently: `jq` turns its error element into `null`,
which flows into `make en` and on through the cascade.

```bash
set -o pipefail
make asr AUDIO=hf://datasets/Narsil/asr_dummy/4.flac | jq -r '.[].text' | make en | jq -r '.text' | make tts OUTPUT=4_en.wav
echo "exit: $?"   # non-zero if any stage failed
```

For the default sample set (three files), `jq` emits one line per file; pass a
single file (as above, via `AUDIO=`) for a one-shot pipeline, or loop over the
files in bash for per-file output.

To inspect every stage of the cascade, write each output to one file with
`tee` (then `tee -a` and `>>` to append), then `cat` it:

```bash
make asr AUDIO=hf://datasets/Narsil/asr_dummy/4.flac | tee out.txt | jq -r '.[].text' | make en | tee -a out.txt | jq -r '.text' | make tts OUTPUT=4_en.wav >> out.txt && cat out.txt
```

- `tee out.txt` after `make asr` writes the ASR JSON to a fresh `out.txt` and sends it to `jq`.
- `tee -a out.txt` after `make en` appends the en JSON and sends it to `jq`.
- `make tts ... >> out.txt` appends the tts JSON summary.
- `cat out.txt` prints the ASR JSON, the en JSON, and the tts JSON (all with stats).

The first `tee` (no `-a`) starts a fresh file each run, so the file holds only
the latest run (no `rm -f` needed). (`jq` is a transform, not a stage, so it is
not teed.)

## Evaluation

`eval_wer.py` measures the Word Error Rate (WER) of an ASR model on a HF
audio dataset. It supports three dataset sources (dispatched by `EVAL_DATASET`):

- [`google/fleurs`](https://huggingface.co/datasets/google/fleurs): read
  speech, 16 kHz wav. Configs `en_us`, `es_419`, `fr_fr`, `de_de`, `hi_in`.
  Split `test`. Public.
- [`diabolocom/talkbank_4_stt`](https://huggingface.co/datasets/diabolocom/talkbank_4_stt):
  spontaneous telephone conversation, 16 kHz mp3. Configs `en`, `es`, `fr`,
  `de`, `jp`, `zh`. Split `segment` (the `switch` split has long silences
  and a much higher WER, so it is not used).
- [`sierra-research/mu-bench`](https://huggingface.co/datasets/sierra-research/mu-bench):
  customer-service telephone calls to a banking AI agent, 8 kHz mono wav.
  Gated, CC-BY-NC-4.0. Locales `en-US`, `es-MX`, `tr-TR`, `vi-VN`, `zh-CN`.
  The `datasets` builder forces the `torchcodec` encoder for this dataset,
  which this CPU project does not depend on, so the manifest and
  per-utterance wav files are downloaded directly with
  [huggingface_hub](https://github.com/huggingface/huggingface_hub), decoded
  with `soundfile`, and resampled 8 kHz to 16 kHz with
  `torchaudio.functional.resample`. Only aggregate metrics (WER, n, runtime)
  are reported for `mu-bench`; no audio or transcripts are redistributed.

A post-hoc repetition-loop guard (gzip compression ratio > 2.4, the openai
whisper CLI default) is applied to every hypothesis. On short or noisy
telephone segments greedy decoding can loop and emit hundreds of repeated
words, which would dominate corpus WER via insertions. The guard is applied
identically to fp32 and HQQ. (transformers 4.44.2 raises `UnboundLocalError`
from its in-generation `compression_ratio_threshold` fallback when
`return_timestamps` is `False`, so the guard is applied post-hoc.)

Hardware: Intel Xeon W-1290 @ 3.20 GHz, 20 cores, CPU only. WER is
host-independent; runtime is host-specific.

Benchmark for [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny)
and [`openai/whisper-base`](https://huggingface.co/openai/whisper-base), fp32
vs the published HQQ 4-bit models
([`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit),
[`dkhokhlov/whisper-base-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-base-hqq-4bit)).
n=100 for `fleurs` and `talkbank`, n=200 for `mu-bench`. WER, lower is
better. "base vs tiny" = base fp32 minus tiny fp32 (negative means base is
better); the `%` is relative to tiny fp32.

**Cross-reference: whisper-tiny vs whisper-base — fp32 and HQQ 4-bit WER**

| Dataset  | Lang  | tiny fp32 | tiny hqq | base fp32 | base hqq | base vs tiny | base vs tiny % |
|----------|-------|-----------|----------|-----------|----------|--------------|----------------|
| fleurs   | en    | 0.1381    | 0.1367   | 0.0985    | 0.0995   | -0.0396      | -28.7%         |
| fleurs   | de    | 0.3019    | 0.2946   | 0.1994    | 0.1901   | -0.1025      | -34.0%         |
| fleurs   | fr    | 0.4451    | 0.4572   | 0.2960    | 0.2963   | -0.1491      | -33.5%         |
| fleurs   | es    | 0.1899    | 0.2149   | 0.1148    | 0.1200   | -0.0751      | -39.5%         |
| fleurs   | hi    | 1.0579    | 1.0579   | 1.0367    | 1.0340   | -0.0212      | -2.0%          |
| mu-bench | en-US | 0.2777    | 0.2994   | 0.2650    | 0.2662   | -0.0127      | -4.6%          |
| mu-bench | es-MX | 0.6884    | 0.6647   | 0.5707    | 0.5758   | -0.1177      | -17.1%         |
| talkbank | en    | 0.4108    | 0.4073   | 0.3810    | 0.3993   | -0.0298      | -7.3%          |
| talkbank | es    | 0.5246    | 0.5170   | 0.3653    | 0.3794   | -0.1593      | -30.4%         |
| talkbank | fr    | 0.7531    | 0.7256   | 0.5737    | 0.5950   | -0.1794      | -23.8%         |
| talkbank | de    | 0.6354    | 0.6425   | 0.5646    | 0.5770   | -0.0708      | -11.1%         |

HQQ 4-bit is within 5% relative of fp32 for both models on every config.
Size: tiny 151.06 -> 61.66 MB (-59.2%); base 290.38 -> 104.89 MB (-63.9%).
`whisper-base` beats `whisper-tiny` on every config except Hindi (both are
not usable). Per-eval runtime, the relative Delta % column, and the full
quantization reports are in [`hqq_report.md`](hqq_report.md) (the
`whisper-tiny` card) and [`hqq_report_base.md`](hqq_report_base.md) (the
`whisper-base` card). Evidence JSONs are committed under `eval_multilingual/`
and `eval_telephone/` (base files are prefixed `base_`).

## Notes

- CPU only. No GPU dependency.
- Audio is decoded with `soundfile` (no ffmpeg required). WAV, FLAC, OGG, and
  MP3 are accepted.
- The ASR pipeline resamples each input to 16 kHz internally, so no
  pre-conversion is needed.
- `make en` needs `sentencepiece` (MarianMT tokenizers); it is in the deps.
- Transformers warnings are silenced; `make asr`, `make en`, and `make tts`
  each print a JSON object/array to stdout (with stats). stderr stays quiet.