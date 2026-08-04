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
(multilingual); the translate step uses a MarianMT model
([`Helsinki-NLP/opus-mt-mul-en`](https://huggingface.co/Helsinki-NLP/opus-mt-mul-en));
the TTS step uses a VITS model
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

The default cascade loads the fp32 ASR model. Set `QUANT=hqq` to load the
HQQ 4-bit ASR model instead; the translate and TTS stages are not quantized.
HQQ 4-bit is also published for `whisper-base` and `whisper-small` (see the
size analysis below).

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

### WER

`eval_wer.py` measures the Word Error Rate (WER) of an ASR model on a HF
audio dataset. It supports two dataset sources (dispatched by `EVAL_DATASET`):

- [`google/fleurs`](https://huggingface.co/datasets/google/fleurs): read
  speech, 16 kHz wav. Configs `en_us`, `es_419`, `fr_fr`, `de_de`, `hi_in`.
  Split `test`. Public.
- [`diabolocom/talkbank_4_stt`](https://huggingface.co/datasets/diabolocom/talkbank_4_stt):
  spontaneous telephone conversation, 16 kHz mp3. Configs `en`, `es`, `fr`,
  `de`, `jp`, `zh`. Split `segment` (the `switch` split has long silences
  and a much higher WER, so it is not used).

A post-hoc repetition-loop guard (gzip compression ratio > 2.4, the openai
whisper CLI default) is applied to every hypothesis. On short or noisy
telephone segments greedy decoding can loop and emit hundreds of repeated
words, which would dominate corpus WER via insertions. The guard is applied
identically to fp32 and HQQ. (transformers 4.44.2 raises `UnboundLocalError`
from its in-generation `compression_ratio_threshold` fallback when
`return_timestamps` is `False`, so the guard is applied post-hoc.)

Hardware: `whisper-tiny` and `whisper-base` were evaluated on Intel Xeon
W-1290 @ 3.20 GHz, 20 cores, CPU only; `whisper-small` was evaluated on an
NVIDIA A10 GPU (24 GB) for speed. WER is host-independent; runtime is
host-specific.

Benchmark for [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny),
[`openai/whisper-base`](https://huggingface.co/openai/whisper-base), and
[`openai/whisper-small`](https://huggingface.co/openai/whisper-small), fp32
vs the published HQQ 4-bit models
([`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit),
[`dkhokhlov/whisper-base-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-base-hqq-4bit),
[`dkhokhlov/whisper-small-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-small-hqq-4bit)).
n=100 for `fleurs` and `talkbank`. WER, lower is
better. "base vs tiny" = base fp32 minus tiny fp32; "small vs base" = small
fp32 minus base fp32 (negative means the larger model is better); the `%`
is relative to the smaller model's fp32.

**Cross-reference: whisper-tiny vs whisper-base vs whisper-small — fp32 and HQQ 4-bit WER**

| Dataset  | Lang  | tiny fp32 | tiny hqq | base fp32 | base hqq | small fp32 | small hqq | base vs tiny | base vs tiny % | small vs base | small vs base % |
|----------|-------|-----------|----------|-----------|----------|------------|-----------|--------------|----------------|---------------|-----------------|
| fleurs   | en    | 0.1381    | 0.1367   | 0.0985    | 0.0995   | 0.0660     | 0.0636    | -0.0396      | -28.7%         | -0.0325       | -33.0%          |
| fleurs   | de    | 0.3019    | 0.2946   | 0.1994    | 0.1901   | 0.0974     | 0.0965    | -0.1025      | -34.0%         | -0.1020       | -51.2%          |
| fleurs   | fr    | 0.4451    | 0.4572   | 0.2960    | 0.2963   | 0.1624     | 0.1589    | -0.1491      | -33.5%         | -0.1336       | -45.1%          |
| fleurs   | es    | 0.1899    | 0.2149   | 0.1148    | 0.1200   | 0.0616     | 0.0648    | -0.0751      | -39.5%         | -0.0532       | -46.3%          |
| fleurs   | hi    | 1.0579    | 1.0579   | 1.0367    | 1.0340   | 0.6777     | 0.7117    | -0.0212      | -2.0%          | -0.3590       | -34.6%          |
| talkbank | en    | 0.4108    | 0.4073   | 0.3810    | 0.3993   | 0.2952     | 0.2929    | -0.0298      | -7.3%          | -0.0858       | -22.5%          |
| talkbank | es    | 0.5246    | 0.5170   | 0.3653    | 0.3794   | 0.2529     | 0.2564    | -0.1593      | -30.4%         | -0.1124       | -30.8%          |
| talkbank | fr    | 0.7531    | 0.7256   | 0.5737    | 0.5950   | 0.4236     | 0.4338    | -0.1794      | -23.8%         | -0.1501       | -26.2%          |
| talkbank | de    | 0.6354    | 0.6425   | 0.5646    | 0.5770   | 0.4850     | 0.4920    | -0.0708      | -11.1%         | -0.0796       | -14.1%          |

HQQ 4-bit is within 5% relative of fp32 for all three models on every
config. `whisper-base` beats `whisper-tiny` on every config except Hindi
(both are not usable); `whisper-small` beats `whisper-base` on every config.

Per-eval runtime, the relative Delta % columns, and the full quantization
reports are in [`hqq_report.md`](hqq_report.md) (the `whisper-tiny` card),
[`hqq_report_base.md`](hqq_report_base.md) (the `whisper-base` card), and
[`hqq_report_small.md`](hqq_report_small.md) (the `whisper-small` card).
Evidence JSONs are committed under `eval_multilingual/` and `eval_telephone/`
(base files are prefixed `base_`, small files are prefixed `small_`).

### Size

Model size is resident weight RAM: the weight memory a host holds at run
time. Both the unquantized original and the HQQ 4-bit model are measured
in fp16 storage + fp16 compute (the deployment mode). fp16 compute is
WER-neutral (measured within n=100 noise), so the deployment mode matches
the published WER. The unquantized original is `openai/whisper-*` (fp32 on
Hugging Face) loaded as fp16 (`torch_dtype=fp16`); whisper is trained in
fp16, so this is near-lossless. The `proj_out` head and the decoder
embedding are tied (one shared weight), held once.

| model  | unquantized fp16 | HQQ 4-bit fp16 | reduction vs fp16 original |
|--------|------------------|----------------|----------------------------|
| tiny   | 75.52 MB         | 57.53 MB       | -23.8%                     |
| base   | 145.19 MB        | 97.22 MB       | -33.0%                     |
| small  | 483.47 MB        | 267.59 MB      | -44.6%                     |

The HQQ model quantizes only the linear weights (the 4-bit and 8-bit
tiers; see [Quantization config](#quantization-config)). Each quantized
linear stores packed weights `W_q` plus a per-group scale and zero (both
fp16 in RAM). PyTorch has no native 4-bit or 8-bit integer dtype, so HQQ
packs the weights into `uint8` — the container dtype, not the precision:
the 4-bit tier packs two values per byte, the 8-bit tier stores one value
per byte. Both are dequantized per group at compute time; the `uint8`
bytes are only how the weights sit in RAM. The embedding, convs, layer
norms, and positional embedding stay fp16. The embedding is the largest
resident component and is not quantized, so the quantization gain is
bounded by the linear-weight share (larger for the deeper small model).

Resident weight RAM by component (HQQ 4-bit, fp16 storage + fp16 compute,
MB, measured after load, deduplicated by storage pointer so the tied
embedding counts once):

| Component (dtype in RAM) | tiny | base | small |
|---|---:|---:|---:|
| Embedding, tied `embed_tokens`=`proj_out` (fp16) | 39.83 | 53.11 | 79.66 |
| HQQ packed weights `W_q` (4/8-bit, `uint8`) | 12.98 | 34.60 | 155.71 |
| HQQ scale (fp16) | 1.03 | 2.75 | 12.39 |
| HQQ zero (fp16) | 1.03 | 2.75 | 12.39 |
| Positional embedding (fp16) | 1.50 | 1.99 | 2.99 |
| Conv `conv1`/`conv2` (fp16) | 1.07 | 1.82 | 3.91 |
| HQQ bias (fp16) | 0.06 | 0.12 | 0.35 |
| LayerNorm (fp16) | 0.03 | 0.07 | 0.19 |
| Resident weight RAM (total) | 57.53 | 97.22 | 267.59 |

The published WER benchmark runs fp32 compute (for cross-model
comparability); its resident RAM is larger because the fp16 non-quantized
weights upcast to fp32 at load time. fp16 compute is the deployment mode
and is WER-neutral.

## Quantization config

All three models use the same mixed-precision HQQ config, tuned once on
`whisper-tiny` (see [Ablation](#ablation)) and applied to `whisper-base` and
`whisper-small` without a separate sweep. No calibration data; the
quantization is direct (grouped, `axis=1`).

- **4-bit tier** (`nbits=4, group_size=32, axis=1`): the decoder self-attention
  projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`), the decoder
  cross-attention projections, and `fc2`.
- **8-bit tier** (`nbits=8, group_size=32, axis=1`): the whole encoder stack
  (`encoder.layers.*` self-attention `q/k/v/out` and `fc1/fc2`) and `fc1` in
  the decoder. The encoder is the acoustic front-end (cheap to keep at 8-bit);
  `fc1` is the GELU up-projection, the more sensitive half of the FFN. Keeping
  these sensitive linears at 8-bit removes almost all of the 4-bit WER gap.
- **Exempt** (not quantized): `proj_out` (the lm_head, tied to the decoder
  embedding — one shared weight, kept fp16), the embedding, `conv1`, `conv2`,
  layer norms, and the positional embedding (all stored fp16; whisper is
  trained in fp16, so this is near-lossless).
- Compute dtype: fp32 for the published WER benchmark (cross-model
  comparability); fp16 for deployment (WER-neutral). See [Size](#size).
- `axis=1` groups along the input/reduction dim. It measured better than
  `axis=0` on `whisper-tiny` (0.1622 vs 0.2032 WER at `group_size=64`);
  `axis=0` targets GPU-optimized inference kernels. `axis=1` is kept for all
  three so the WER is directly comparable.

The 8-bit-tier patterns (`encoder.layers,fc1` substring match) scale
automatically to the deeper stack; no per-model change.

| model | linears quantized | 4-bit | 8-bit | exempt (tied) |
|---|---:|---:|---:|---|
| whisper-tiny  | 64  | 36  | 28  | `proj_out` |
| whisper-base  | 96  | 54  | 42  | `proj_out` |
| whisper-small | 192 | 108 | 84  | `proj_out` |

The 8-bit tier is set by `HQQ_8BIT_PATTERNS` (comma-separated name substrings,
default `encoder.layers,fc1`) and its bit width by `HQQ_8BIT_NBITS` (default
8). The 4-bit tier is the default for every other linear; `proj_out` is
hardcoded exempt.

### Why not transformers `HqqConfig`

transformers `HqqConfig` needs a GPU for both quantization and loading, and it
writes a `quantization_config` to `config.json` that triggers a GPU check on
every `from_pretrained`, which breaks CPU loading. `whisper-tiny` and
`whisper-base` are CPU-only, so the [`hqq`](https://github.com/mobiusml/hqq)
library is used directly instead: linear layers are replaced with
`HQQLinear` on CPU, the model is saved with `save_quantized`, and a small
subclass of `AutoHQQHFModel` loads it on CPU (the base loader builds
`WhisperModel` without `generate`; the subclass builds
`WhisperForConditionalGeneration`). `whisper-small` uses the same `hqq`-lib
path (not `HqqConfig`) on the A10 (`ASR_DEVICE=cuda`), so the saved
`qmodel.pt` is the same CPU-loadable format as tiny/base.

### Ablation

Config sweep on `whisper-tiny`, same 100 English (`fleurs` `en_us`) samples.
All rows use `axis=1` except row A. The **8-bit tier** column names the
linears assigned to the 8-bit tier; the rest are 4-bit. `group` is the
`group_size`. The winner (row H) is the published config. These runs predate
the repetition-loop guard; English `en_us` does not loop, so the guard does
not change these values.

| Row | group | 8-bit tier             | WER    | Size (on-disk) |
|-----|-------|------------------------|--------|----------------|
| -   | 64    | (none, all 4-bit)      | 0.1622 | 54.86 MB       |
| A   | 64    | (axis=0, all 4-bit)    | 0.2032 | -              |
| B   | 32    | (none, all 4-bit)      | 0.1480 | 56.93 MB       |
| C   | 32    | encoder_attn           | 0.1513 | 58.11 MB       |
| D   | 64    | encoder_attn           | 0.1537 | 56.05 MB       |
| E   | 16    | (none, all 4-bit)      | 0.1499 | 61.06 MB       |
| F   | 32    | fc1                    | 0.1457 | 59.29 MB       |
| G   | 32    | encoder.layers         | 0.1433 | 60.48 MB       |
| H   | 32    | encoder.layers, fc1    | 0.1367 | 61.66 MB       |

What the sweep shows:
- `axis=1` beats `axis=0` decisively (row A vs the 64/all-4-bit row).
- `group_size=32` beats `group_size=64` (row B vs the 64/all-4-bit row);
  `group_size=16` (row E) did not improve over 32, so 32 is the sweet spot.
- Assigning the cross-attention to the 8-bit tier (rows C, D) did not help;
  cross-attention K/V are computed once from the already-clean encoder output,
  so their quantization error does not compound.
- Assigning `fc1` to the 8-bit tier (row F) helps; assigning the whole encoder
  stack (row G) helps more; doing both (row H) matches the fp32 baseline.

## safetensors format

Each published HQQ repo ships `model.safetensors` next to `qmodel.pt`.
`qmodel.pt` is a torch pickle (zip archive) and the default loader target.
`model.safetensors` is an alternative single file: a flat
8-byte-JSON-header + raw-tensor-bytes format with no pickle, zero-mappable,
and parseable from C/C++/Rust. Use it for host tooling that cannot read a
torch pickle (for example a deployment loader).

Safetensors stores tensors only, so the per-linear HQQ config scalars
(`nbits`, `group_size`, `axis`, the packing string, the bools) are encoded as
tensors via the `HQQLinear` `encoded_state_dict` path.
`HQQLinear.load_state_dict` detects the `encoded_state_dict` key and decodes
them back, so the round-trip needs no extra metadata file. The export keeps
the non-quantized modules (embedding, convs, norms, `proj_out`) at fp16 and
`HQQLinear` overrides `.to()` as a no-op so it keeps its scale/zero/bias;
loading either format upcasts the fp16 weights to the compute dtype
identically. The two formats give the same WER (lossless round-trip).

Load the safetensors with `HQQ_FORMAT=safetensors`; the default (no
`HQQ_FORMAT`) loads `qmodel.pt`.

```python
import os, hqq_asr
os.environ["HQQ_FORMAT"] = "safetensors"
pipe = hqq_asr.build_pipeline("dkhokhlov/whisper-tiny-hqq-4bit", quant="hqq")
```

Export it from a local quantized dir with `python export_safetensors.py` (set
`HQQ_OUT=` for base/small). The export loads on CPU; the saved `qmodel.pt` is
device-independent, so no GPU is needed.

## Notes

- CPU only by default. No GPU dependency. `whisper-small` was quantized and
  evaluated on an NVIDIA A10 GPU for speed via an optional CUDA venv
  (`make gpu-venv`, `ASR_DEVICE=cuda`); the saved HQQ model is
  device-independent and loads on CPU or GPU. See
  [`hqq_report_small.md`](hqq_report_small.md).
- Each published HQQ repo ships `qmodel.pt` (torch pickle, the default loader
  target) and `model.safetensors` (a flat tensor map, no pickle, parseable
  from C/C++/Rust for host tooling such as a deployment loader). Set
  `HQQ_FORMAT=safetensors` to load the safetensors; both formats give the same
  WER. See [safetensors format](#safetensors-format).
- `proj_out` and the embedding are fp16, not 4-bit. A smaller model is
  possible if the embedding is also quantized, but that raises the WER risk
  on the vocab projection and was not done here.
- The quantization config was tuned on English (`whisper-tiny`) and applied
  to base/small without a separate sweep. Under the repetition-loop guard,
  HQQ is within 5% relative of fp32 on every tested config. Use the fp32
  model when the lowest WER is required and the size is acceptable.
- Evaluated on `fleurs` (en, es, fr, de, hi) and `talkbank_4_stt` (en, es, fr,
  de). Other languages and datasets were not measured. WER on conversational
  telephone speech is high because the domain is hard; this is a base-model
  property. Use a larger Whisper model for this domain when lower WER is
  needed. Hindi is not production-quality at any model size.
- Audio is decoded with `soundfile` (no ffmpeg required). WAV, FLAC, OGG, and
  MP3 are accepted.
- The ASR pipeline resamples each input to 16 kHz internally, so no
  pre-conversion is needed.
- `make en` needs `sentencepiece` (MarianMT tokenizers); it is in the deps.
- Transformers warnings are silenced; `make asr`, `make en`, and `make tts`
  each print a JSON object/array to stdout (with stats). stderr stays quiet.

## Citation

If you use these HQQ models or the benchmark, cite this repository:

```bibtex
@misc{khokhlov_whisper_cascade,
  author       = {Dmitri Khokhlov},
  title        = {whisper-cascade: CPU speech-to-text cascade with HQQ 4-bit Whisper},
  year         = {2026},
  url          = {https://github.com/dkhokhlov/whisper-cascade},
  note         = {HQQ 4-bit quantized Whisper (tiny/base/small) for deployment},
}
```

Upstream:
- Whisper: Radford et al., "Robust Speech Recognition via Large-Scale Weak
  Supervision", 2022. <https://cdn.openai.com/papers/whisper.pdf>
- HQQ: Mobius Labs, "HQQ: Half-Quadratic Quantization".
  <https://github.com/mobiusml/hqq>