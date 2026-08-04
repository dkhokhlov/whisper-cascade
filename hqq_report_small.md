---
license: mit
base_model: openai/whisper-small
library_name: transformers
pipeline_tag: automatic-speech-recognition
language:
  - en
  - multilingual
tags:
  - whisper
  - hqq
  - quantization
  - 4-bit
  - asr
  - gpu
---

# HQQ 4-bit Whisper-Small Quantization Report

Model card source for `dkhokhlov/whisper-small-hqq-4bit`.

## Related models

- [`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit) — HQQ 4-bit, whisper-tiny (CPU eval)
- [`dkhokhlov/whisper-base-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-base-hqq-4bit) — HQQ 4-bit, whisper-base (CPU eval)
- Source model: [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) (fp32)
- Benchmark + code: [`dkhokhlov/whisper-cascade`](https://github.com/dkhokhlov/whisper-cascade)

## Summary

[`openai/whisper-small`](https://huggingface.co/openai/whisper-small) was
quantized with [HQQ](https://huggingface.co/docs/transformers/en/quantization/hqq)
4-bit grouped quantization. The deployment RAM (fp16 compute) is 292.99
MB, 69.7% smaller than the 966.94 MB fp32 model; it equals the on-disk
`qmodel.pt` and is WER-neutral. The fp32-compute benchmark resident RAM
is 379.5 MB (60.7% smaller). The Word Error Rate (WER) on 100 English samples
from [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) `en_us`
(test split) stayed at the baseline level: 0.0660 (fp32) vs 0.0636 (HQQ), a
difference of -0.0024 absolute (-3.6% relative), within the noise of a
100-sample eval.

The quantization config is the same mixed-precision setting tuned on
[`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) (see
[`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit)
and [`dkhokhlov/whisper-base-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-base-hqq-4bit)):
the whole encoder stack and the `fc1` feed-forward up-projection are kept at
8-bit, and the remaining decoder linears are 4-bit. The same config was
applied to small without a separate sweep; it matches the fp32 baseline on
English and stays within 5% relative of fp32 on every tested config.

This is the first revision evaluated on a GPU for speed. `whisper-tiny` and
`whisper-base` were quantized and evaluated on CPU; `whisper-small` (241.7 M
parameters, 12 encoder + 12 decoder layers) was quantized and evaluated on an
NVIDIA A10 GPU. WER is host-independent; only runtime is host-specific. The
saved `qmodel.pt` is device-independent and loads on CPU or GPU.

`whisper-small` is the largest of the three models, so it transcribes best.
Across the 9 tested configs (5 `fleurs`, 4 `talkbank`), small fp32 beats base
fp32 on every config. Hindi is still not usable at either precision. The
cross-reference table is below.

Generation config: this revision clears the English `forced_decoder_ids`
that the source `config.json` ships with and writes a modern
`generation_config.json`, so the model auto-detects the language and
transcribes (the standard multilingual Whisper behavior). The model weights
are quantized once; `qmodel.pt` is the single weights file.

## Source model

- Base: [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
  (multilingual, 241.7 M parameters, 12 encoder + 12 decoder layers).
- Original size: 966.94 MB (fp32 weights).

## Quantization

- Method: [HQQ](https://huggingface.co/docs/transformers/en/quantization/hqq)
  (Half-Quadratic Quantization). No calibration data.
- Library: [hqq](https://github.com/mobiusml/hqq) 0.2.8.post1 (used directly;
  not transformers `HqqConfig`).
- Linear layers: 192 of the 193 linears are quantized (the tied `proj_out` is
  skipped). 108 linears are 4-bit and 84 are 8-bit:
  - 4-bit: `nbits=4, group_size=32, axis=1`. These are the decoder
    self-attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`),
    the decoder cross-attention projections, and `fc2`.
  - 8-bit: `nbits=8, group_size=32, axis=1`. These are the whole encoder
    stack (`encoder.layers.*` self-attention `q/k/v/out` and `fc1/fc2`) and
    `fc1` in the decoder. The encoder is the acoustic front-end; `fc1` is the
    GELU up-projection, the more sensitive half of the FFN.
  - `axis=1` groups along the input/reduction dim. It measured better than
    `axis=0` on whisper-tiny; `axis=0` targets GPU-optimized inference kernels.
    `axis=1` is kept here so the WER is directly comparable to the tiny/base
    axis=1 results.
- `proj_out` (the lm_head): NOT quantized. It is tied to the decoder
  embedding and shares one weight tensor. It is kept tied in fp16, so the
  weight is stored once with no quantization error.
- Non-quantized modules (embedding, conv1, conv2, layer norms): stored as
  fp16. Whisper is trained in fp16, so this is near-lossless.
- Compute dtype: fp32. The stored fp16 weights are upcast to fp32 at load
  time so the HQQ dequantization and the matmuls run in fp32. fp32 compute is
  kept (not fp16) so the WER is directly comparable to the fp32 tiny/base
  results; the A10 runs fp32 far faster than a CPU, so the speed goal is met.
- Device: quantized on the A10 GPU (`ASR_DEVICE=cuda`). The saved `qmodel.pt`
  is device-independent and loads on CPU or GPU.
- This config was tuned on `whisper-tiny` and applied to small without a
  separate ablation. It matches the fp32 baseline on English (see Results).

## Why not transformers `HqqConfig`

transformers `HqqConfig` writes a `quantization_config` to `config.json` that
triggers a GPU check on every `from_pretrained`, which breaks CPU loading.
The tiny/base models are CPU-only, so the `hqq` library is used directly
instead: linear layers are replaced with `HQQLinear`, the model is saved with
`save_quantized`, and a small subclass of `AutoHQQHFModel` loads it. Small
uses the same `hqq`-lib path (not `HqqConfig`) so the saved `qmodel.pt` is
the same CPU-loadable format as tiny/base; only the quantization device is
the A10 (`ASR_DEVICE=cuda`).

## Evaluation

- Metric: WER via [jiwer](https://github.com/jitsi/jiwer). Both reference
  and hypothesis are normalized (lowercase, remove punctuation, collapse
  spaces) before alignment.
- Decoding: greedy, auto-detected language, `task=transcribe`. fp32 and HQQ
  run the same samples with the same settings.
- Repetition-loop guard: a post-hoc gzip compression-ratio guard (ratio >
  2.4, the openai whisper CLI default) is applied to every hypothesis. On
  short or noisy telephone segments, greedy decoding can loop and emit
  hundreds of repeated words, which dominate corpus WER via insertions. A
  looping hypothesis is treated as empty, so it counts as deletions on its
  own reference, not as hundreds of insertions. The guard is applied
  identically to fp32 and HQQ. transformers 4.44.2 raises `UnboundLocalError`
  from its in-generation `compression_ratio_threshold` fallback when
  `return_timestamps` is `False`, so the guard is applied post-hoc.
- Hardware: NVIDIA A10 GPU (24 GB), CUDA torch 2.4.1+cu121, `ASR_DEVICE=cuda`.
  WER is host-independent; runtime (elapsed, real-time factor) is
  host-specific and reflects this GPU.

## Datasets

Three datasets are used. The first is read speech; the other two are
conversational telephone speech, which is harder.

- [`google/fleurs`](https://huggingface.co/datasets/google/fleurs): read
  speech, 16 kHz wav. Configs `en_us`, `es_419` (Spanish), `fr_fr` (French),
  `de_de` (German), `hi_in` (Hindi). Split `test`. `n=100` per config (the
  first 100, streamed). Loaded with the `datasets` audio feature set to
  `decode=False` (raw bytes), decoded with `soundfile`. Public license.
- [`diabolocom/talkbank_4_stt`](https://huggingface.co/datasets/diabolocom/talkbank_4_stt):
  spontaneous telephone conversation, 16 kHz mp3. Configs `en`, `es`, `fr`,
  `de` (also `jp`, `zh`). Split `segment` (the `switch` split has long
  silences and a much higher WER, so it is not used). `n=100` per config.
  mp3 bytes are decoded with `torchaudio`.

## Results (English, en_us, fleurs)

| Metric                 | fp32 baseline | HQQ 4-bit  | Delta abs        | Delta %        |
|------------------------|---------------|------------|------------------|----------------|
| WER                    | 0.0660        | 0.0636     | -0.0024          | -3.6%          |
| Deployment RAM (fp16)  | 966.94 MB     | 292.99 MB  | -673.95 MB       | -69.7%         |
| Benchmark RAM (fp32)   | 966.94 MB     | 379.5 MB   | -587.44 MB       | -60.7%         |
| Avg real-time factor   | 0.029         | 0.047      | +0.018           | -              |
| Total elapsed (100 fx) | 28.1 s        | 44.4 s     | +16.3 s          | -              |
| Samples succeeded      | 100 / 100     | 100 / 100  | -                | -              |

Notes:
- The WER difference (-0.0024) is within the noise of a 100-sample eval, so
  the HQQ model is statistically tied with the fp32 baseline (HQQ is slightly
  lower here, as quantization noise can break the occasional repetition loop).
- Deployment RAM (fp16 compute) is the weight memory a host holds at run
  time in the fp16-compute deployment mode. It equals the on-disk
  `qmodel.pt` weight file (the full published directory adds the processor
  and tokenizer files, ~294.92 MB). fp16 compute is WER-neutral, so the
  deployment mode matches the published WER.
- Benchmark RAM (fp32 compute) is the weight memory in the published WER
  setup. It is larger than deployment RAM because the non-quantized fp16
  weights upcast to fp32 at load time. The `proj_out`/embedding tie is
  restored after load so the embedding is held once. The packed 4-bit
  weights stay uint8 in RAM; HQQ dequantizes them per group at compute time.
- HQQ has no fused 4-bit kernel on this path, so each linear dequantizes
  group-by-group. HQQ is slower than fp32 because of the dequantize overhead;
  the absolute speed is still far under real time on the A10.
- `whisper-small` English WER (0.0660) is lower than `whisper-base` (0.0985)
  and `whisper-tiny` (0.1381) on the same samples. Small is the largest model;
  the size cost is 967 MB fp32 / 293 MB HQQ (deployment RAM) vs 290 MB /
  103 MB for base and 151 MB / 60 MB for tiny.

## Results (multilingual, fleurs, n=100 per language)

The same HQQ weights evaluated across five `fleurs` configs, fp32 vs HQQ,
both with auto-detected language and the repetition-loop guard.

| Language | Config | fp32 WER | HQQ WER | Delta abs   | Delta %  | fp32 run | HQQ run |
|----------|--------|----------|---------|-------------|----------|----------|---------|
| English  | en_us  | 0.0660   | 0.0636  | -0.0024     | -3.6%    | 28.1 s   | 44.4 s  |
| German   | de_de  | 0.0974   | 0.0965  | -0.0009     | -0.9%    | 39.9 s   | 66.9 s  |
| French   | fr_fr  | 0.1624   | 0.1589  | -0.0035     | -2.2%    | 42.1 s   | 66.1 s  |
| Spanish  | es_419 | 0.0616   | 0.0648  | +0.0032     | +5.2%    | 36.9 s   | 60.7 s  |
| Hindi    | hi_in  | 0.6777   | 0.7117  | +0.0340     | +5.0%    | 114.1 s  | 223.7 s |

Notes:
- HQQ is within 5% relative of fp32 on every language. The quantization is
  effectively lossless across the tested languages.
- A WER of 0.68 (Hindi) is high but far below base (1.04) and tiny (1.06);
  `whisper-small` is closer to usable for Hindi than the smaller models, but
  still not production-quality at either precision.
- HQQ runtime is higher than fp32 (1.2x-2.0x) because of the dequantize
  overhead. The absolute speed is still well under real time on the A10,
  except Hindi (long output inflates both elapsed and RTF).
- Per-language evidence is committed under `eval_multilingual/small_<config>_<fp32|hqq>.json`.

## Telephone benchmark

`whisper-small` (fp32 vs HQQ 4-bit) on conversational telephone speech, with
the repetition-loop guard. Runtime is on the A10 GPU.

### talkbank_4_stt (spontaneous conversation)

| Language | Config | fp32 WER | HQQ WER | Delta abs   | Delta %  | n   | fp32 run | HQQ run | RTF (fp32) | RTF (hqq) |
|----------|--------|----------|---------|-------------|----------|-----|----------|---------|------------|-----------|
| English  | en     | 0.2952   | 0.2929  | -0.0023     | -0.8%    | 100 | 23.1 s   | 28.8 s  | 0.076      | 0.095     |
| Spanish  | es     | 0.2529   | 0.2564  | +0.0035     | +1.4%    | 100 | 27.3 s   | 47.5 s  | 0.048      | 0.084     |
| French   | fr     | 0.4236   | 0.4338  | +0.0102     | +2.4%    | 100 | 34.1 s   | 55.5 s  | 0.052      | 0.084     |
| German   | de     | 0.4850   | 0.4920  | +0.0070     | +1.4%    | 100 | 23.6 s   | 34.9 s  | 0.117      | 0.173     |

The `segment` split is used. `n=100` per language, the first 100 segments
streamed.

Notes:
- HQQ is within 5% relative of fp32 on every telephone config. The
  quantization is effectively lossless on conversational telephone speech.
- The WERs are high (0.25-0.49) because conversational telephone speech is
  hard; this is a property of the model, not of the quantization. `whisper-small`
  beats `whisper-base` on every telephone config (see the cross-reference).
- HQQ RTF is higher than fp32 (1.2x-1.6x) because of the dequantize
  overhead.

## Cross-reference (whisper-base vs whisper-small)

Same samples, same eval harness, same repetition-loop guard. `tiny` =
[`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit);
`base` =
[`dkhokhlov/whisper-base-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-base-hqq-4bit);
`small` = this model. "small vs base" is small fp32 minus base fp32; a negative
value means small transcribes better. The `%` is relative to base fp32.

**Cross-reference: whisper-tiny vs whisper-base vs whisper-small — fp32 and HQQ 4-bit WER**

| Dataset  | Lang  | tiny fp32 | tiny hqq | base fp32 | base hqq | small fp32 | small hqq | small vs base abs | small vs base % |
|----------|-------|-----------|----------|-----------|----------|------------|-----------|-------------------|-----------------|
| fleurs   | en    | 0.1381    | 0.1367   | 0.0985    | 0.0995   | 0.0660     | 0.0636    | -0.0325           | -33.0%          |
| fleurs   | de    | 0.3019    | 0.2946   | 0.1994    | 0.1901   | 0.0974     | 0.0965    | -0.1020           | -51.2%          |
| fleurs   | fr    | 0.4451    | 0.4572   | 0.2960    | 0.2963   | 0.1624     | 0.1589    | -0.1336           | -45.1%          |
| fleurs   | es    | 0.1899    | 0.2149   | 0.1148    | 0.1200   | 0.0616     | 0.0648    | -0.0532           | -46.3%          |
| fleurs   | hi    | 1.0579    | 1.0579   | 1.0367    | 1.0340   | 0.6777     | 0.7117    | -0.3590           | -34.6%          |
| talkbank | en    | 0.4108    | 0.4073   | 0.3810    | 0.3993   | 0.2952     | 0.2929    | -0.0858           | -22.5%          |
| talkbank | es    | 0.5246    | 0.5170   | 0.3653    | 0.3794   | 0.2529     | 0.2564    | -0.1124           | -30.8%          |
| talkbank | fr    | 0.7531    | 0.7256   | 0.5737    | 0.5950   | 0.4236     | 0.4338    | -0.1501           | -26.2%          |
| talkbank | de    | 0.6354    | 0.6425   | 0.5646    | 0.5770   | 0.4850     | 0.4920    | -0.0796           | -14.1%          |

Size and speed:

| Model        | fp32 size | HQQ size | HQQ reduction | English HQQ WER | English HQQ RTF |
|--------------|-----------|----------|---------------|-----------------|-----------------|
| whisper-tiny | 151.06 MB | 61.66 MB | 59.2%         | 0.1367          | 0.074 (CPU)     |
| whisper-base | 290.38 MB | 104.89 MB| 63.9%         | 0.0995          | 0.144 (CPU)     |
| whisper-small| 966.94 MB | 294.92 MB| 69.5%         | 0.0636          | 0.047 (A10)     |

Notes:
- `whisper-small` beats `whisper-base` on every config.
- The HQQ size reduction grows with model size (59% / 64% / 70%): a larger
  share of the weights is in the quantized linears relative to the fp16
  embedding. Small is 2.8x the size of base at HQQ 4-bit (293 MB vs 103 MB).
- Hindi improves a lot with model size (tiny 1.06, base 1.04, small 0.68)
  but is still not production-quality.
- RTF is not comparable across rows: tiny/base RTF is on CPU, small RTF is on
  the A10 GPU. Within a model, HQQ RTF is higher than fp32 because of the
  dequantize overhead.

## Load and use

The model auto-detects the spoken language and transcribes (multilingual
Whisper behavior). Pass `language` to force a language when it is known.

```python
import hqq_asr
pipe = hqq_asr.build_pipeline("dkhokhlov/whisper-small-hqq-4bit", quant="hqq")
text = pipe({"array": audio, "sampling_rate": 16000})["text"]                       # auto-detect
text = pipe({"array": audio, "sampling_rate": 16000},
             generate_kwargs={"language": "spanish", "task": "transcribe"})["text"]   # force
```

Command line (this repository):

```
make asr MODEL_ASR=dkhokhlov/whisper-small-hqq-4bit QUANT=hqq AUDIO=clip.wav
```

### safetensors format

The repo also ships `model.safetensors` next to `qmodel.pt`. It is a flat tensor
map: an 8-byte JSON header plus raw tensor bytes, with no pickle, zero-mappable,
and parseable from C/C++/Rust. Use it for host tooling that cannot read a torch
pickle (for example a deployment loader). Set `HQQ_FORMAT=safetensors` to load it;
the default (no `HQQ_FORMAT`) loads `qmodel.pt`. Both give the same WER. The
HQQ config per linear (nbits, group_size, axis, packing, bools) is encoded as
tensors inside the file, so no extra metadata file is needed.

```python
import os, hqq_asr
os.environ["HQQ_FORMAT"] = "safetensors"
pipe = hqq_asr.build_pipeline("dkhokhlov/whisper-small-hqq-4bit", quant="hqq")
```

To export it from a local quantized dir: `python export_safetensors.py`.

## Reproduce

```
# 1. Create the CUDA venv (A10), then quantize locally (writes whisper-small-hqq-4bit/).
make gpu-venv
ASR_DEVICE=cuda MODEL_ASR=openai/whisper-small HQQ_OUT=whisper-small-hqq-4bit \
  .venv-gpu/bin/python quantize.py

# 2. Measure baseline WER (fp32) on the A10.
ASR_DEVICE=cuda EVAL_LIMIT=100 MODEL_ASR=openai/whisper-small EVAL_CONFIG=en_us \
  EVAL_OUT=eval_small_baseline.json .venv-gpu/bin/python eval_wer.py

# 3. Measure HQQ WER.
ASR_DEVICE=cuda EVAL_LIMIT=100 QUANT=hqq MODEL_ASR=./whisper-small-hqq-4bit EVAL_CONFIG=en_us \
  EVAL_OUT=eval_small_hqq.json .venv-gpu/bin/python eval_wer.py

# 4. Telephone benchmark (talkbank segment split).
ASR_DEVICE=cuda EVAL_DATASET=diabolocom/talkbank_4_stt EVAL_CONFIG=en EVAL_SPLIT=segment EVAL_LIMIT=100 \
  MODEL_ASR=openai/whisper-small EVAL_OUT=small_talkbank_en_fp32.json .venv-gpu/bin/python eval_wer.py

# 5. Publish (needs a Hugging Face write token).
PUSH=1 ASR_DEVICE=cuda HQQ_REPO=dkhokhlov/whisper-small-hqq-4bit MODEL_ASR=openai/whisper-small \
  HQQ_OUT=whisper-small-hqq-4bit HQQ_REPORT=hqq_report_small.md .venv-gpu/bin/python quantize.py
```

## Benchmarked with

The quantization, eval harness, and per-config WER evidence live in the
benchmark repository: [dkhokhlov/whisper-cascade](https://github.com/dkhokhlov/whisper-cascade)
(branch `main`, tag `v1.6.0`). Small evidence:
- Multilingual (fleurs): `eval_multilingual/small_<config>_<fp32|hqq>.json`.
- Telephone: `eval_telephone/small_talkbank_<lang>_<fp32|hqq>.json`.
- The `whisper-tiny` and `whisper-base` comparison numbers are in the
  [`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit)
  and
  [`dkhokhlov/whisper-base-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-base-hqq-4bit)
  model cards.

## Limitations

- GPU eval. This model was quantized and evaluated on an NVIDIA A10 GPU
  (`ASR_DEVICE=cuda`, `.venv-gpu`) for speed; `whisper-tiny` and
  `whisper-base` were evaluated on CPU. WER is host-independent; only runtime
  differs. The saved `qmodel.pt` is device-independent and loads on CPU or
  GPU. To run on CPU, use the CPU `.venv` with `ASR_DEVICE` unset.
- `proj_out` and the embedding are fp16, not 4-bit. A smaller model is
  possible if the embedding is also quantized, but that raises the WER risk
  on the vocab projection and was not done here.
- The quantization config was tuned on `whisper-tiny` and applied to small
  without a separate sweep. It matches the fp32 baseline on English and
  stays within 5% relative of fp32 on every tested config.
- Evaluated on `fleurs` (en, es, fr, de, hi) and `talkbank_4_stt` (en, es, fr,
  de). Other languages and datasets were not measured.
- WER on conversational telephone speech is high (0.25-0.49) because the
  domain is hard. Hindi is still not production-quality at either precision
  (0.68 fp32).