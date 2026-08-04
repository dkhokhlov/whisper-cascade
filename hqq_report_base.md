---
license: mit
base_model: openai/whisper-base
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

# HQQ 4-bit Whisper-Base Quantization Report

Model card source for `dkhokhlov/whisper-base-hqq-4bit`.

## Summary

[`openai/whisper-base`](https://huggingface.co/openai/whisper-base) was
quantized with [HQQ](https://huggingface.co/docs/transformers/en/quantization/hqq)
4-bit grouped quantization for CPU inference. The model size shrank from
290.38 MB (fp32) to 104.89 MB (63.9% reduction). The Word Error Rate (WER)
on 100 English samples from [`google/fleurs`](https://huggingface.co/datasets/google/fleurs)
`en_us` (test split) stayed at the baseline level: 0.0985 (fp32) vs 0.0995
(HQQ), a difference of +0.0010 absolute (+1.0% relative), within the noise
of a 100-sample eval.

The quantization config is the same mixed-precision setting tuned on
[`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) (see
[`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit)):
the whole encoder stack and the `fc1` feed-forward up-projection are kept at
8-bit, and the remaining decoder linears are 4-bit. The same config was
applied to base without a separate sweep; it matches the fp32 baseline on
English and stays within 5% relative of fp32 on every tested config.

`whisper-base` is a larger model than `whisper-tiny`, so it transcribes
better. Across the 11 tested configs (5 `fleurs`, 4 `talkbank`, 2 `mu-bench`),
base fp32 beats tiny fp32 by 2-40% relative (except Hindi, where both models
are not usable). The cross-reference table is below.

Generation config: this revision clears the English `forced_decoder_ids`
that the source `config.json` ships with and writes a modern
`generation_config.json`, so the model auto-detects the language and
transcribes (the standard multilingual Whisper behavior). The model weights
are quantized once; `qmodel.pt` is the single weights file. This is the first
published revision of the base model (tag `v1.5.0` of the benchmark repo).

## Source model

- Base: [`openai/whisper-base`](https://huggingface.co/openai/whisper-base)
  (multilingual, 72.6 M parameters).
- Original size: 290.38 MB (`model.safetensors`, fp32).

## Quantization

- Method: [HQQ](https://huggingface.co/docs/transformers/en/quantization/hqq)
  (Half-Quadratic Quantization). No calibration data.
- Library: [hqq](https://github.com/mobiusml/hqq) 0.2.8.post1 (used directly;
  not transformers `HqqConfig`).
- Linear layers: 96 of the 97 linears are quantized (the tied `proj_out` is
  skipped). 54 linears are 4-bit and 42 are 8-bit:
  - 4-bit: `nbits=4, group_size=32, axis=1`. These are the decoder
    self-attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`),
    the decoder cross-attention projections, and `fc2`.
  - 8-bit: `nbits=8, group_size=32, axis=1`. These are the whole encoder
    stack (`encoder.layers.*` self-attention `q/k/v/out` and `fc1/fc2`) and
    `fc1` in the decoder. The encoder is the acoustic front-end; `fc1` is the
    GELU up-projection, the more sensitive half of the FFN.
  - `axis=1` groups along the input/reduction dim. It measured better than
    `axis=0` on whisper-tiny; `axis=0` targets GPU-optimized inference kernels.
- `proj_out` (the lm_head): NOT quantized. It is tied to the decoder
  embedding and shares one weight tensor. It is kept tied in fp16, so the
  weight is stored once with no quantization error.
- Non-quantized modules (embedding, conv1, conv2, layer norms): stored as
  fp16. Whisper is trained in fp16, so this is near-lossless.
- Compute dtype: fp32. The stored fp16 weights are upcast to fp32 at load
  time so the HQQ dequantization and the CPU matmuls run in fp32.
- This config was tuned on `whisper-tiny` and applied to base without a
  separate ablation. It matches the fp32 baseline on English (see Results).

## Why not transformers `HqqConfig`

transformers `HqqConfig` needs a GPU for both quantization and loading. It
writes a `quantization_config` to `config.json` that triggers a GPU check on
every `from_pretrained`. This project is CPU-only. The `hqq` library is used
directly instead: linear layers are replaced with `HQQLinear` on CPU, the
model is saved with `save_quantized`, and a small subclass of
`AutoHQQHFModel` loads it on CPU.

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
- Hardware: Intel Xeon W-1290 @ 3.20 GHz, 20 cores, CPU only. WER is
  host-independent; runtime (elapsed, real-time factor) is host-specific and
  reflects this host.

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
- [`sierra-research/mu-bench`](https://huggingface.co/datasets/sierra-research/mu-bench):
  customer-service telephone calls to a banking AI agent, 8 kHz mono wav.
  Locales `en-US` (817 utterances), `es-MX` (792), `tr-TR`, `vi-VN`, `zh-CN`.
  `n=200` per locale for `en-US` and `es-MX`. Gated, CC-BY-NC-4.0. The
  `datasets` builder forces the `torchcodec` encoder for this dataset, which
  this CPU project does not depend on, so the `metadata.jsonl` manifest and
  the per-utterance wav files are downloaded directly with
  [huggingface_hub](https://github.com/huggingface/huggingface_hub), decoded
  with `soundfile`, and resampled 8 kHz to 16 kHz with
  `torchaudio.functional.resample`. Only aggregate metrics (WER, n, runtime)
  are reported here; no audio, transcripts, or per-utterance data are
  retained or redistributed.

## Results (English, en_us, fleurs)

| Metric                 | fp32 baseline | HQQ 4-bit  | Delta abs        | Delta %        |
|------------------------|---------------|------------|------------------|----------------|
| WER                    | 0.0985        | 0.0995     | +0.0010          | +1.0%          |
| Model size on disk     | 290.38 MB     | 104.89 MB  | -185.49 MB       | -63.9%         |
| Weights only (qmodel)  | 290.38 MB     | 102.97 MB  | -187.41 MB       | -64.5%         |
| Avg real-time factor   | 0.093         | 0.144      | +0.051           | -              |
| Total elapsed (100 fx) | 88.9 s        | 137.3 s    | +48.4 s          | -              |
| Samples succeeded      | 100 / 100     | 100 / 100  | -                | -              |

Notes:
- The WER difference (+0.0010) is within the noise of a 100-sample eval, so
  the HQQ model is statistically tied with the fp32 baseline.
- HQQ has no fused 4-bit kernel on CPU, so each linear dequantizes
  group-by-group. HQQ is slower than fp32 because of the dequantize overhead;
  the absolute speed is still well under real time.
- `whisper-base` English WER (0.0985) is lower than `whisper-tiny` (0.1381)
  on the same samples. Base is the larger model; the size cost is 290 MB
  fp32 / 105 MB HQQ vs 151 MB / 62 MB for tiny.

## Results (multilingual, fleurs, n=100 per language)

The same HQQ weights evaluated across five `fleurs` configs, fp32 vs HQQ,
both with auto-detected language and the repetition-loop guard.

| Language | Config | fp32 WER | HQQ WER | Delta abs   | Delta %  | fp32 run | HQQ run |
|----------|--------|----------|---------|-------------|----------|----------|---------|
| English  | en_us  | 0.0985   | 0.0995  | +0.0010     | +1.0%    | 88.9 s   | 137.3 s |
| German   | de_de  | 0.1994   | 0.1901  | -0.0093     | -4.7%    | 131.3 s  | 211.0 s |
| French   | fr_fr  | 0.2960   | 0.2963  | +0.0003     | +0.1%    | 122.4 s  | 193.3 s |
| Spanish  | es_419 | 0.1148   | 0.1200  | +0.0052     | +4.5%    | 114.3 s  | 180.0 s |
| Hindi    | hi_in  | 1.0367   | 1.0340  | -0.0027     | -0.3%    | 237.6 s  | 486.5 s |

Notes:
- HQQ is within 5% relative of fp32 on every language. The quantization is
  effectively lossless across the tested languages.
- A WER above 1.0 (Hindi) means more edit operations than reference words;
  `whisper-base` is not usable for Hindi at either precision. This is a
  property of the base model, not of the quantization.
- HQQ runtime is higher than fp32 (1.3x-2.0x) because of the CPU dequantize
  overhead. The absolute speed is still well under real time, except Hindi
  (long hallucinated output inflates both elapsed and RTF).
- Per-language evidence is committed under `eval_multilingual/base_<config>_<fp32|hqq>.json`.

## Call-center benchmark

`whisper-base` (fp32 vs HQQ 4-bit) on conversational telephone speech, with
the repetition-loop guard. Runtime is on the Intel Xeon W-1290 host.

### talkbank_4_stt (spontaneous conversation)

| Language | Config | fp32 WER | HQQ WER | Delta abs   | Delta %  | n   | fp32 run | HQQ run | RTF (fp32) | RTF (hqq) |
|----------|--------|----------|---------|-------------|----------|-----|----------|---------|------------|-----------|
| English  | en     | 0.3810   | 0.3993  | +0.0183     | +4.8%    | 100 | 81.2 s   | 105.0 s | 0.268      | 0.346     |
| Spanish  | es     | 0.3653   | 0.3794  | +0.0141     | +3.9%    | 100 | 83.9 s   | 139.4 s | 0.149      | 0.247     |
| French   | fr     | 0.5737   | 0.5950  | +0.0213     | +3.7%    | 100 | 114.6 s  | 185.6 s | 0.174      | 0.281     |
| German   | de     | 0.5646   | 0.5770  | +0.0124     | +2.2%    | 100 | 65.3 s   | 103.5 s | 0.323      | 0.512     |

The `segment` split is used. `n=100` per language, the first 100 segments
streamed.

### mu-bench (customer-service telephone)

| Locale | Config | fp32 WER | HQQ WER | Delta abs   | Delta %  | n   | fp32 run | HQQ run | RTF (fp32) | RTF (hqq) |
|--------|--------|----------|---------|-------------|----------|-----|----------|---------|------------|-----------|
| en-US  | en     | 0.2650   | 0.2662  | +0.0012     | +0.5%    | 200 | 139.4 s  | 194.2 s | 0.184      | 0.257     |
| es-MX  | es     | 0.5707   | 0.5758  | +0.0051     | +0.9%    | 200 | 183.1 s  | 311.6 s | 0.212      | 0.360     |

`mu-bench` is 8 kHz mono telephone speech, resampled to 16 kHz. `n=200` per
locale. Gated, CC-BY-NC-4.0; only aggregate metrics are reported here (no
audio or transcripts are redistributed).

Notes:
- HQQ is within 5% relative of fp32 on every call-center config. The
  quantization is effectively lossless on conversational telephone speech.
- The WERs are high (0.27-0.60) because conversational telephone speech is
  hard; this is a property of the model, not of the quantization. `whisper-base`
  beats `whisper-tiny` on every call-center config (see the cross-reference).
- HQQ RTF is higher than fp32 (1.2x-1.6x) because of the CPU dequantize
  overhead.

## Cross-reference (whisper-tiny vs whisper-base)

Same samples, same eval harness, same repetition-loop guard. `tiny` =
[`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit);
`base` = this model. "base vs tiny" is base fp32 minus tiny fp32; a negative
value means base transcribes better. The `%` is relative to tiny fp32.

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

Size and speed:

| Model        | fp32 size | HQQ size | HQQ reduction | English HQQ WER | English HQQ RTF |
|--------------|-----------|----------|---------------|-----------------|-----------------|
| whisper-tiny | 151.06 MB | 61.66 MB | 59.2%         | 0.1367          | 0.074           |
| whisper-base | 290.38 MB | 104.89 MB| 63.9%         | 0.0995          | 0.144           |

Notes:
- `whisper-base` beats `whisper-tiny` on every config except Hindi (where
  both are not usable). The improvement is largest on Spanish and French
  (30-40% lower WER).
- The HQQ size reduction is similar for both models (59% / 64%). Base is
  about 1.7x the size of tiny at HQQ 4-bit (105 MB vs 62 MB).
- HQQ RTF is higher for base (0.144 vs 0.074 on English) because base has
  more linears to dequantize. Both stay well under real time.

## Load and use

The model auto-detects the spoken language and transcribes (multilingual
Whisper behavior). Pass `language` to force a language when it is known.

```python
import hqq_asr
pipe = hqq_asr.build_pipeline("dkhokhlov/whisper-base-hqq-4bit", quant="hqq")
text = pipe({"array": audio, "sampling_rate": 16000})["text"]                       # auto-detect
text = pipe({"array": audio, "sampling_rate": 16000},
             generate_kwargs={"language": "spanish", "task": "transcribe"})["text"]   # force
```

Command line (this repository):

```
make asr MODEL_ASR=dkhokhlov/whisper-base-hqq-4bit QUANT=hqq AUDIO=clip.wav
```

## Reproduce

```
# 1. Quantize locally (writes whisper-base-hqq-4bit/).
MODEL_ASR=openai/whisper-base HQQ_OUT=whisper-base-hqq-4bit python quantize.py

# 2. Measure baseline WER (fp32).
EVAL_LIMIT=100 MODEL_ASR=openai/whisper-base EVAL_CONFIG=en_us \
  EVAL_OUT=eval_base_baseline.json python eval_wer.py

# 3. Measure HQQ WER.
EVAL_LIMIT=100 QUANT=hqq MODEL_ASR=./whisper-base-hqq-4bit EVAL_CONFIG=en_us \
  EVAL_OUT=eval_base_hqq.json python eval_wer.py

# 4. Call-center benchmark.
EVAL_DATASET=diabolocom/talkbank_4_stt EVAL_CONFIG=en EVAL_SPLIT=segment EVAL_LIMIT=100 \
  MODEL_ASR=openai/whisper-base EVAL_OUT=base_talkbank_en_fp32.json python eval_wer.py
EVAL_DATASET=sierra-research/mu-bench EVAL_CONFIG=en EVAL_SPLIT=train EVAL_LIMIT=200 \
  QUANT=hqq MODEL_ASR=./whisper-base-hqq-4bit EVAL_OUT=base_mubench_en_hqq.json python eval_wer.py

# 5. Publish (needs a Hugging Face write token).
PUSH=1 HQQ_REPO=dkhokhlov/whisper-base-hqq-4bit MODEL_ASR=openai/whisper-base \
  HQQ_OUT=whisper-base-hqq-4bit python quantize.py
```

## Benchmarked with

The quantization, eval harness, and per-config WER evidence live in the
benchmark repository: [dkhokhlov/whisper-cascade](https://github.com/dkhokhlov/whisper-cascade)
(branch `hqq-4bit`, tag `v1.5.0`). Base evidence:
- Multilingual (fleurs): `eval_multilingual/base_<config>_<fp32|hqq>.json`.
- Call-center: `eval_callcenter/base_talkbank_<lang>_<fp32|hqq>.json` and
  `eval_callcenter/base_mubench_<lang>_<fp32|hqq>.json` (mu-bench files hold
  only aggregate metrics, per the CC-BY-NC-4.0 terms).
- The `whisper-tiny` comparison numbers are in the
  [`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit)
  model card.

## Limitations

- CPU only. The model loads and runs on CPU. A GPU is not required and not
  used.
- `proj_out` and the embedding are fp16, not 4-bit. A smaller model is
  possible if the embedding is also quantized, but that raises the WER risk
  on the vocab projection and was not done here.
- The quantization config was tuned on `whisper-tiny` and applied to base
  without a separate sweep. It matches the fp32 baseline on English and
  stays within 5% relative of fp32 on every tested config. A base-specific
  sweep might tighten the small Spanish/French gap further.
- Evaluated on `fleurs` (en, es, fr, de, hi), `talkbank_4_stt` (en, es, fr,
  de), and `mu-bench` (en-US, es-MX). Other languages and datasets were not
  measured.
- WER on conversational telephone speech is high (0.27-0.60) because the
  model is small. Use a larger Whisper model for this domain when lower WER
  is needed. `whisper-base` is not usable for Hindi at either precision.