# whisper-cascade

A minimal speech-to-text tool that runs on CPU by default (an optional GPU
path speeds up the larger ASR model), with two text helpers that compose
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
(multilingual); the translate step uses a [MarianMT](https://huggingface.co/docs/transformers/model_doc/marian) model
([`Helsinki-NLP/opus-mt-mul-en`](https://huggingface.co/Helsinki-NLP/opus-mt-mul-en));
the TTS step uses a [VITS](https://huggingface.co/docs/transformers/model_doc/vits) model
([`facebook/mms-tts-eng`](https://huggingface.co/facebook/mms-tts-eng)) — all
through the Hugging Face `transformers` library.

Per-model runtime weights (resident weight RAM: the weight memory a host
holds at run time; for the fp32 models this equals the weights file size):

| Model | Decoder type | Stored dtype | Loaded as | Loaded file | Resident RAM |
|---|---|---|---|---|---|
| [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | autoregressive (enc-dec) | float32 | float32 | `model.safetensors` | 151.06 MB |
| [`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit) | autoregressive (enc-dec) | 4-bit HQQ + fp16 | fp16 | `qmodel.pt` | 57.53 MB |
| [`Helsinki-NLP/opus-mt-mul-en`](https://huggingface.co/Helsinki-NLP/opus-mt-mul-en) | autoregressive (enc-dec) | float32 (default) | float32 | `pytorch_model.bin` (no safetensors) | 310.39 MB |
| [`facebook/mms-tts-eng`](https://huggingface.co/facebook/mms-tts-eng) | non-autoregressive | float32 | float32 | `model.safetensors` | 145.23 MB |

Combined (fp32 ASR cascade) → ~607 MB (151.06 + 310.39 + 145.23 = 606.68 MB)
Combined (HQQ ASR cascade) → ~513 MB (57.53 + 310.39 + 145.23 = 513.15 MB)

The default cascade loads the fp32 ASR model. **Set `QUANT=hqq`** to load the
HQQ 4-bit ASR model instead; the translate and TTS stages are not quantized.
HQQ 4-bit is also published for `whisper-base` and `whisper-small` (see the
[size analysis](docs/ASR.md#size)).

Each stage has its own deep-dive:

- [ASR deep-dive](docs/ASR.md) — model, samples, custom audio, WER evaluation,
  the [HQQ](https://github.com/mobiusml/hqq) quantization config and ablation, and the safetensors format
- [MT deep-dive](docs/MT.md) — [MarianMT](https://huggingface.co/docs/transformers/model_doc/marian) model selection and output
- [TTS deep-dive](docs/TTS.md) — [VITS](https://huggingface.co/docs/transformers/model_doc/vits) model selection and output

No Docker, no ffmpeg, no GPU required. Dependencies are managed with [uv](https://docs.astral.sh/uv/).
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
| `make quantize` | Quantize `MODEL_ASR` with HQQ 4-bit to `HQQ_OUT` (`build/`). |
| `make push` | Quantize and upload `HQQ_OUT` to `HQQ_REPO` (needs `HF_TOKEN_WRITE`). |
| `make eval-baseline` | Measure baseline WER (fp32 `MODEL_ASR`) on `EVAL_DATASET`/`EVAL_CONFIG`/`EVAL_SPLIT` (`EVAL_LIMIT`). |
| `make eval-hqq` | Measure HQQ WER (`QUANT=hqq MODEL_ASR=HQQ_REPO`) on `EVAL_DATASET`/`EVAL_CONFIG`/`EVAL_SPLIT` (`EVAL_LIMIT`). |
| `make onnx` | Export `HQQ_REPO` (HF) to ONNX (packed weights, 2 graphs) -> `ONNX_OUT` (`build/`; `.venv-onnx`). |
| `make hqq-reference` | Write the full 100-sample HQQ text manifest (the ONNX exact-text gate oracle). |
| `make eval-onnx` | ONNX WER + exact-text gate vs the manifest (`QUANT=onnx MODEL_ASR=ONNX_OUT`; `.venv-onnx`). |
| `make bench-matrix` | Run the full WER matrix (tiny+base+small) into `build/` (gitignored; small needs `.venv-gpu`). |
| `make push-onnx` | Upload `ONNX_OUT/*.onnx` (+ `MODEL_CARD`) into `HQQ_REPO` (needs `HF_TOKEN_WRITE`). |
| `make test` | Run the fast unit tests (no model load, no network). |
| `make test-integration` | Run the integration tests (load the real Whisper/MarianMT/VITS models). |
| `make clean` | Remove bytecode cache + the `build/` artifact dir (incl. `bench-matrix` output); keeps `.venv`. |
| `make clean-all` | Also remove all local venvs (`.venv`, `.venv-gpu`, `.venv-onnx`); HF cache is left untouched. |

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

## Notes

- `make en` needs `sentencepiece` (MarianMT tokenizers); it is in the deps.
- Transformers warnings are silenced; `make asr`, `make en`, and `make tts`
  each print a JSON object/array to stdout (with stats). stderr stays quiet.
