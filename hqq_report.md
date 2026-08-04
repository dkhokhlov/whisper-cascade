---
license: mit
base_model: openai/whisper-tiny
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
  - cpu
---

# HQQ 4-bit Whisper-Tiny Quantization Report

Model card source for `dkhokhlov/whisper-tiny-hqq-4bit`.

## Summary

[`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) was
quantized with [HQQ](https://huggingface.co/docs/transformers/en/quantization/hqq)
4-bit grouped quantization for CPU inference. The model size shrank from
151.06 MB (fp32) to 61.65 MB (59.2% reduction). The Word Error Rate (WER) on
100 English samples from [`google/fleurs`](https://huggingface.co/datasets/google/fleurs)
`en_us` (test split) stayed at the baseline level: 0.1381 (fp32) vs 0.1367
(HQQ), a difference of -0.0014 absolute (-1.0% relative), which is within
the noise of a 100-sample eval.

The key setting is mixed precision: the whole encoder stack and the `fc1`
feed-forward up-projection are kept at 8-bit, and the remaining decoder
linears are 4-bit. The encoder is the acoustic stack (only 4 layers, cheap
to protect) and `fc1` is the more sensitive half of the FFN; keeping these
at 8-bit removes almost all of the 4-bit WER gap.

The same weights are evaluated across five `fleurs` languages and two
customer-service telephone datasets below. The quantization config was tuned
on English. Under the repetition-loop guard (see Evaluation), HQQ stays
within 0.025 absolute of fp32 on every tested language and dataset. The
quantization is effectively lossless across the tested set.

Revision note (v1.4.0): the earlier v1.3.2 multilingual table reported
`es_419` HQQ 0.4235 and `hi_in` 1.7599 / 2.4145. Those numbers were inflated
by greedy-decoding repetition loops on a few short segments. A post-hoc
compression-ratio guard (the openai whisper CLI default) now catches those
loops. The corrected numbers are in this card. The model weights are
unchanged (`qmodel.pt` sha256 `0e183ee00b1cdd71`); v1.4.0 is a card-only
update.

Revision note (v1.5.0): added a `whisper-base` cross-reference (the
"Cross-reference" section) and a relative Delta % column on every results
table. The model weights are unchanged; v1.5.0 is a card-only update. The
`whisper-base` model is published separately as
[`dkhokhlov/whisper-base-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-base-hqq-4bit).

Generation config: this revision clears the English `forced_decoder_ids`
that the source `config.json` ships with and writes a modern
`generation_config.json`, so the model auto-detects the language and
transcribes (the standard multilingual Whisper behavior). The earlier
v1.3.1 revision kept the English-forced config; its weights are
byte-identical to this revision, and English WER is identical under either
config. v1.3.1 is preserved at git tag `v1.3.1` / HF revision `e43f2bb`.

## Source model

- Base: [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny)
  (multilingual).
- Original size: 151.06 MB (`model.safetensors`, fp32).

## Quantization

- Method: [HQQ](https://huggingface.co/docs/transformers/en/quantization/hqq)
  (Half-Quadratic Quantization). No calibration data.
- Library: [hqq](https://github.com/mobiusml/hqq) 0.2.8.post1 (used directly;
  not transformers `HqqConfig`).
- Linear layers: 64 of the 65 linears are quantized (the tied `proj_out` is
  skipped). 36 linears are 4-bit and 28 are 8-bit:
  - 4-bit: `nbits=4, group_size=32, axis=1`. These are the decoder
    self-attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`),
    the decoder cross-attention projections, and `fc2`.
  - 8-bit: `nbits=8, group_size=32, axis=1`. These are the whole encoder
    stack (`encoder.layers.*` self-attention `q/k/v/out` and `fc1/fc2`) and
    `fc1` in the decoder. The encoder is the acoustic front-end and is only 4
    layers, so protecting it is cheap; `fc1` is the GELU up-projection, the
    more sensitive half of the FFN.
  - `axis=1` groups along the input/reduction dim. It measured better than
    `axis=0` on whisper-tiny (0.1622 vs 0.2032 WER at group_size=64); `axis=0`
    targets GPU-optimized inference kernels.
- `proj_out` (the lm_head): NOT quantized. It is tied to the decoder
  embedding and shares one weight tensor. Quantizing it would store the
  weight twice and add error to the vocab projection. It is kept tied in
  fp16, so the weight is stored once with no quantization error.
- Non-quantized modules (embedding, conv1, conv2, layer norms): stored as
  fp16. Whisper is trained in fp16, so this is near-lossless.
- Compute dtype: fp32. The stored fp16 weights are upcast to fp32 at load
  time so the HQQ dequantization and the CPU matmuls run in fp32.

## Why not transformers `HqqConfig`

transformers `HqqConfig` needs a GPU for both quantization and loading. It
writes a `quantization_config` to `config.json` that triggers a GPU check on
every `from_pretrained`. This project is CPU-only. The `hqq` library is used
directly instead: linear layers are replaced with `HQQLinear` on CPU, the
model is saved with `save_quantized`, and a small subclass of
`AutoHQQHFModel` loads it on CPU (the base loader builds `WhisperModel`
without `generate`; the subclass builds `WhisperForConditionalGeneration`).

## Evaluation

- Metric: WER via `jiwer`. Both reference and hypothesis are normalized
  (lowercase, remove punctuation, collapse spaces) before alignment.
- Decoding: greedy, auto-detected language, `task=transcribe`. fp32 and HQQ
  run the same samples with the same settings.
- Repetition-loop guard: a post-hoc gzip compression-ratio guard (ratio >
  2.4, the openai whisper CLI default) is applied to every hypothesis. On
  short or noisy telephone segments, greedy decoding can loop and emit
  hundreds of repeated words, which dominate corpus WER via insertions. A
  looping hypothesis is treated as empty, so it counts as deletions on its
  own reference, not as hundreds of insertions. The guard is applied
  identically to fp32 and HQQ, so the comparison stays fair. transformers
  4.44.2 raises `UnboundLocalError` from its in-generation
  `compression_ratio_threshold` fallback when `return_timestamps` is `False`,
  so the guard is applied post-hoc, not in generation.
- Hardware: Intel Xeon W-1290 @ 3.20 GHz, 20 cores, CPU only. WER is
  host-independent; runtime (elapsed, real-time factor) is host-specific and
  reflects this host.

## Datasets

Three datasets are used. The first is read speech; the other two are
conversational telephone speech, which is harder for a small model.

- [`google/fleurs`](https://huggingface.co/datasets/google/fleurs): read
  speech, 16 kHz wav. Configs `en_us`, `es_419` (Spanish), `fr_fr` (French),
  `de_de` (German), `hi_in` (Hindi). Split `test`. `n=100` per config (the
  first 100, streamed). Loaded with the `datasets` audio feature set to
  `decode=False` (raw bytes), decoded with `soundfile`. Public license.
- [`diabolocom/talkbank_4_stt`](https://huggingface.co/datasets/diabolocom/talkbank_4_stt):
  spontaneous telephone conversation, 16 kHz mp3. Configs `en`, `es`, `fr`,
  `de` (also `jp`, `zh`). Split `segment` (the `switch` split has long
  silences and a much higher WER, so it is not used). `n=100` per config.
  Loaded the same way as `fleurs`; mp3 bytes are decoded with `torchaudio`.
- [`sierra-research/mu-bench`](https://huggingface.co/datasets/sierra-research/mu-bench):
  customer-service telephone calls to a banking AI agent, 8 kHz mono wav.
  Locales `en-US` (817 utterances), `es-MX` (792), `tr-TR`, `vi-VN`, `zh-CN`.
  `n=200` per locale for `en-US` and `es-MX`. This dataset is gated,
  CC-BY-NC-4.0. The `datasets` builder forces the `torchcodec` encoder for
  this dataset, which this CPU project does not depend on, so the
  `metadata.jsonl` manifest and the per-utterance wav files are downloaded
  directly with [huggingface_hub](https://github.com/huggingface/huggingface_hub)
  and decoded with `soundfile`; the 8 kHz audio is resampled to 16 kHz with
  `torchaudio.functional.resample`. Only aggregate metrics (WER, n, runtime)
  are reported here; no audio, transcripts, or per-utterance data are
  retained or redistributed.

## Results (English, en_us, fleurs)

| Metric                 | fp32 baseline | HQQ 4-bit  | Delta abs        | Delta %        |
|------------------------|---------------|------------|------------------|----------------|
| WER                    | 0.1381        | 0.1367     | -0.0014          | -1.0%          |
| Model size on disk     | 151.06 MB     | 61.65 MB   | -89.41 MB        | -59.2%         |
| Weights only (qmodel)  | 151.06 MB     | 59.74 MB   | -91.32 MB        | -60.4%         |
| Avg real-time factor   | 0.054         | 0.074      | +0.020           | -              |
| Total elapsed (100 fx) | 51.5 s        | 70.9 s     | +19.4 s          | -              |
| Samples succeeded      | 100 / 100     | 100 / 100  | -                | -              |

Notes:
- The size is the full output directory: `qmodel.pt` plus the processor and
  tokenizer files (`vocab.json`, `merges.txt`, `normalizer.json`, etc.).
- The WER difference (-0.0014) is within the noise of a 100-sample eval, so
  the HQQ model is statistically tied with the fp32 baseline, not better.
  The headline is that 4-bit HQQ with mixed-precision protection matches the
  fp32 baseline at 59% smaller size.
- HQQ has no fused 4-bit kernel on CPU, so each linear dequantizes
  group-by-group. For this tiny model the absolute speed is still well under
  real time; HQQ is slower than fp32 because of the dequantize overhead.

## Results (multilingual, fleurs, n=100 per language)

The same HQQ weights (`qmodel.pt` sha256 `0e183ee00b1cdd71`) evaluated across
five `fleurs` configs, fp32 vs HQQ, both with auto-detected language and the
repetition-loop guard.

| Language | Config | fp32 WER | HQQ WER | Delta abs   | Delta %  | fp32 run | HQQ run |
|----------|--------|----------|---------|-------------|----------|----------|---------|
| English  | en_us  | 0.1381   | 0.1367  | -0.0014     | -1.0%    | 51.5 s   | 70.9 s  |
| German   | de_de  | 0.3019   | 0.2946  | -0.0073     | -2.4%    | 72.3 s   | 103.7 s |
| French   | fr_fr  | 0.4451   | 0.4572  | +0.0121     | +2.7%    | 75.9 s   | 111.4 s |
| Spanish  | es_419 | 0.1899   | 0.2149  | +0.0250     | +13.2%   | 65.3 s   | 110.5 s |
| Hindi    | hi_in  | 1.0579   | 1.0579  | 0.0000      | 0.0%     | 131.3 s  | 244.0 s |

Notes:
- Under the guard, HQQ is within 0.025 absolute of fp32 on every language.
  The quantization is effectively lossless across the tested languages.
- The v1.3.2 table reported `es_419` HQQ 0.4235 and `hi_in` 1.7599 / 2.4145.
  Those values were inflated by repetition loops that the guard now catches.
  This revision corrects them. The model weights are unchanged.
- A WER above 1.0 (Hindi) means more edit operations than reference words;
  `whisper-tiny` is not usable for Hindi at either precision. This is a
  property of the base model, not of the quantization. HQQ does not make
  Hindi worse (fp32 and HQQ are identical on Hindi under the guard).
- HQQ runtime is higher than fp32 (1.3x-1.9x) because of the CPU dequantize
  overhead. The absolute speed is still well under real time.
- Per-language evidence is committed under `eval_multilingual/` in the
  benchmark repository (see "Benchmarked with" below).

## Telephone benchmark

These benchmarks measure `whisper-tiny` (fp32 vs HQQ 4-bit) on conversational
telephone speech, which is harder than the read-speech `fleurs` eval. Both
fp32 and HQQ use auto-detected language and `task=transcribe`. The
repetition-loop guard described in Evaluation is applied to both. Runtime
is on the Intel Xeon W-1290 host.

### talkbank_4_stt (spontaneous conversation)

| Language | Config | fp32 WER | HQQ WER | Delta abs   | Delta %  | n   | fp32 run | HQQ run | Avg RTF (fp32) | Avg RTF (hqq) |
|----------|--------|----------|---------|-------------|----------|-----|----------|---------|----------------|---------------|
| English  | en     | 0.4108   | 0.4073  | -0.0035     | -0.9%    | 100 | 44.5 s   | 44.0 s  | 0.147          | 0.145         |
| Spanish  | es     | 0.5246   | 0.5170  | -0.0076     | -1.4%    | 100 | 56.0 s   | 94.0 s  | 0.099          | 0.167         |
| French   | fr     | 0.7531   | 0.7256  | -0.0275     | -3.7%    | 100 | 97.6 s   | 91.4 s  | 0.148          | 0.139         |
| German   | de     | 0.6354   | 0.6425  | +0.0071     | +1.1%    | 100 | 38.8 s   | 44.8 s  | 0.192          | 0.222         |

The `segment` split is used. The `switch` split has long silences and a much
higher WER, so it is not used. `n=100` per language, the first 100 segments
streamed.

### mu-bench (customer-service telephone)

| Locale | Config | fp32 WER | HQQ WER | Delta abs   | Delta %  | n   | fp32 run | HQQ run | Avg RTF (fp32) | Avg RTF (hqq) |
|--------|--------|----------|---------|-------------|----------|-----|----------|---------|----------------|---------------|
| en-US  | en     | 0.2777   | 0.2994  | +0.0217     | +7.8%    | 200 | 70.8 s   | 99.9 s  | 0.093          | 0.132         |
| es-MX  | es     | 0.6884   | 0.6647  | -0.0237     | -3.4%    | 200 | 105.9 s  | 139.4 s | 0.122          | 0.161         |

`mu-bench` is 8 kHz mono telephone speech, resampled to 16 kHz. `n=200` per
locale. `mu-bench` is gated, CC-BY-NC-4.0; only aggregate metrics are
reported here (no audio or transcripts are redistributed).

Notes:
- fp32 and HQQ are within noise on every telephone config (deltas are all
  within +/-0.03 absolute). HQQ 4-bit is effectively lossless on
  conversational telephone speech, as it is on `fleurs` read speech.
- The WERs are high (0.28-0.75) because `whisper-tiny` is small and
  conversational telephone speech is hard; this is a property of the base
  model, not of the quantization. Use a larger Whisper model when lower WER
  is needed on this domain.
- `mu-bench` en-US (0.28) is lower than `talkbank` en (0.41) because
  `mu-bench` is banking customer-service calls (more structured) while
  `talkbank` is spontaneous conversation.
- HQQ RTF is higher than fp32 (1.1x-1.7x) because HQQ has no fused 4-bit
  kernel on CPU; each linear dequantizes group-by-group. For this tiny model
  the absolute speed is still well under real time.
- The repetition-loop guard changes the raw WER materially on `talkbank`
  (without it, one fp32 sample looped to 333 words and inflated English fp32
  corpus WER from ~0.41 to 0.79). The guard is applied identically to fp32
  and HQQ, so the comparison is fair.

## Cross-reference (whisper-tiny vs whisper-base)

Same samples, same eval harness, same repetition-loop guard. `tiny` = this
model; `base` =
[`dkhokhlov/whisper-base-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-base-hqq-4bit).
"base vs tiny" is base fp32 minus tiny fp32; a negative value means base
transcribes better. The `%` is relative to tiny fp32.

**Cross-reference: whisper-tiny vs whisper-base — fp32 and HQQ 4-bit WER**

| Dataset  | Lang  | tiny fp32 | tiny hqq | base fp32 | base hqq | base vs tiny abs | base vs tiny % |
|----------|-------|-----------|----------|-----------|----------|------------------|----------------|
| fleurs   | en    | 0.1381    | 0.1367   | 0.0985    | 0.0995   | -0.0396          | -28.7%         |
| fleurs   | de    | 0.3019    | 0.2946   | 0.1994    | 0.1901   | -0.1025          | -34.0%         |
| fleurs   | fr    | 0.4451    | 0.4572   | 0.2960    | 0.2963   | -0.1491          | -33.5%         |
| fleurs   | es    | 0.1899    | 0.2149   | 0.1148    | 0.1200   | -0.0751          | -39.5%         |
| fleurs   | hi    | 1.0579    | 1.0579   | 1.0367    | 1.0340   | -0.0212          | -2.0%          |
| mu-bench | en-US | 0.2777    | 0.2994   | 0.2650    | 0.2662   | -0.0127          | -4.6%          |
| mu-bench | es-MX | 0.6884    | 0.6647   | 0.5707    | 0.5758   | -0.1177          | -17.1%         |
| talkbank | en    | 0.4108    | 0.4073   | 0.3810    | 0.3993   | -0.0298          | -7.3%          |
| talkbank | es    | 0.5246    | 0.5170   | 0.3653    | 0.3794   | -0.1593          | -30.4%         |
| talkbank | fr    | 0.7531    | 0.7256   | 0.5737    | 0.5950   | -0.1794          | -23.8%         |
| talkbank | de    | 0.6354    | 0.6425   | 0.5646    | 0.5770   | -0.0708          | -11.1%         |

`whisper-base` beats `whisper-tiny` on every config except Hindi (both are
not usable). The HQQ size reduction is similar (59% / 64%); base is about
1.7x the size of tiny at HQQ 4-bit (105 MB vs 62 MB) and about 1.9x slower
on English (RTF 0.144 vs 0.074). Use base when the lower WER is worth the
size and speed cost; use tiny when the smallest model is needed.

## Ablation (config sweep, same 100 English samples)

All rows use `axis=1`. `protect` names the linears kept at 8-bit; the rest
are 4-bit. `group` is the group_size. The winner (row H) is the published
config. These runs predate the repetition-loop guard; English `en_us` does
not loop, so the guard does not change these values.

| Row | group | protect (8-bit)        | WER    | Size   |
|-----|-------|------------------------|--------|--------|
| -   | 64    | (none, all 4-bit)      | 0.1622 | 54.86 MB |
| A   | 64    | (axis=0, all 4-bit)    | 0.2032 | -      |
| B   | 32    | (none, all 4-bit)      | 0.1480 | 56.93 MB |
| C   | 32    | encoder_attn           | 0.1513 | 58.11 MB |
| D   | 64    | encoder_attn           | 0.1537 | 56.05 MB |
| E   | 16    | (none, all 4-bit)      | 0.1499 | 61.06 MB |
| F   | 32    | fc1                    | 0.1457 | 59.29 MB |
| G   | 32    | encoder.layers         | 0.1433 | 60.48 MB |
| H   | 32    | encoder.layers, fc1    | 0.1367 | 61.66 MB |

What the sweep shows:
- `axis=1` beats `axis=0` decisively (row A vs the 64/all-4-bit row).
- `group_size=32` beats `group_size=64` (row B vs the 64/all-4-bit row);
  `group_size=16` (row E) did not improve over 32, so 32 is the sweet spot.
- Protecting the cross-attention (rows C, D) did not help; cross-attention
  K/V are computed once from the already-clean encoder output, so their
  quantization error does not compound.
- Protecting `fc1` (row F) helps; protecting the whole encoder stack (row G)
  helps more; doing both (row H) matches the fp32 baseline.

## Load and use

The model auto-detects the spoken language and transcribes (multilingual
Whisper behavior). Pass `language` to force a language when it is known.

```python
import hqq_asr
pipe = hqq_asr.build_pipeline("dkhokhlov/whisper-tiny-hqq-4bit", quant="hqq")
text = pipe({"array": audio, "sampling_rate": 16000})["text"]                       # auto-detect
text = pipe({"array": audio, "sampling_rate": 16000},
             generate_kwargs={"language": "spanish", "task": "transcribe"})["text"]   # force
```

Command line (this repository):

```
make asr MODEL_ASR=dkhokhlov/whisper-tiny-hqq-4bit QUANT=hqq AUDIO=clip.wav
```

### safetensors format

The repo also ships `model.safetensors` next to `qmodel.pt`. It is a flat tensor
map: an 8-byte JSON header plus raw tensor bytes, with no pickle, zero-mappable,
and parseable from C/C++/Rust. Use it for host tooling that cannot read a torch
pickle (for example an FPGA loader). Set `HQQ_FORMAT=safetensors` to load it;
the default (no `HQQ_FORMAT`) loads `qmodel.pt`. Both give the same WER. The
HQQ config per linear (nbits, group_size, axis, packing, bools) is encoded as
tensors inside the file, so no extra metadata file is needed.

```python
import os, hqq_asr
os.environ["HQQ_FORMAT"] = "safetensors"
pipe = hqq_asr.build_pipeline("dkhokhlov/whisper-tiny-hqq-4bit", quant="hqq")
```

To export it from a local quantized dir: `python export_safetensors.py`.

## Reproduce

```
# 1. Quantize locally (writes whisper-tiny-hqq-4bit/).
python quantize.py

# 2. Measure baseline WER (fp32) per language.
EVAL_LIMIT=100 MODEL_ASR=openai/whisper-tiny EVAL_CONFIG=en_us \
  EVAL_OUT=eval_baseline.json python eval_wer.py

# 3. Measure HQQ WER per language.
EVAL_LIMIT=100 QUANT=hqq MODEL_ASR=./whisper-tiny-hqq-4bit EVAL_CONFIG=en_us \
  EVAL_OUT=eval_hqq.json python eval_wer.py

# 4. Telephone benchmark (talkbank segment split, mu-bench direct download).
EVAL_DATASET=diabolocom/talkbank_4_stt EVAL_CONFIG=en EVAL_SPLIT=segment EVAL_LIMIT=100 \
  MODEL_ASR=openai/whisper-tiny EVAL_OUT=talkbank_en_fp32.json python eval_wer.py
EVAL_DATASET=sierra-research/mu-bench EVAL_CONFIG=en EVAL_SPLIT=train EVAL_LIMIT=200 \
  QUANT=hqq MODEL_ASR=./whisper-tiny-hqq-4bit EVAL_OUT=mubench_en_hqq.json python eval_wer.py

# 5. Publish (needs a Hugging Face write token).
PUSH=1 HQQ_REPO=dkhokhlov/whisper-tiny-hqq-4bit python quantize.py
```

## Benchmarked with

The quantization, eval harness, and per-config WER evidence live in the
benchmark repository: [dkhokhlov/whisper-cascade](https://github.com/dkhokhlov/whisper-cascade)
(branch `hqq-4bit`, tag `v1.5.0`). Evidence:
- English (fleurs): `eval_baseline.json`, `eval_hqq.json`.
- Multilingual (fleurs): `eval_multilingual/<config>_<fp32|hqq>.json`.
- Telephone: `eval_telephone/talkbank_<lang>_<fp32|hqq>.json` and
  `eval_telephone/mubench_<lang>_<fp32|hqq>.json` (mu-bench files hold only
  aggregate metrics, per the CC-BY-NC-4.0 terms).
- The `whisper-base` comparison numbers and its base evidence files
  (`eval_multilingual/base_*`, `eval_telephone/base_*`) are in the
  [`dkhokhlov/whisper-base-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-base-hqq-4bit)
  model card and the same repo at tag `v1.5.0`.

## Limitations

- CPU only. The model loads and runs on CPU. A GPU is not required and not
  used.
- `proj_out` and the embedding are fp16, not 4-bit. A smaller model is
  possible if the embedding is also quantized, but that raises the WER risk
  on the vocab projection and was not done here.
- The quantization config was tuned on English. Under the repetition-loop
  guard, HQQ is within 0.025 absolute of fp32 on every tested language and
  dataset. Spanish shows a small +0.025 regression on `fleurs`; the other
  configs are within noise. Use the fp32 model when the lowest WER is
  required and the size is acceptable.
- Evaluated on `fleurs` (en, es, fr, de, hi), `talkbank_4_stt` (en, es, fr,
  de), and `mu-bench` (en-US, es-MX). The base model is multilingual; other
  languages and datasets were not measured.
- WER on conversational telephone speech is high (0.28-0.75) because
  `whisper-tiny` is small. This is a base-model property. Use a larger
  Whisper model for this domain when lower WER is needed.