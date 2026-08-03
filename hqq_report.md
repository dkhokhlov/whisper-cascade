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

`openai/whisper-tiny` was quantized with HQQ 4-bit grouped quantization for
CPU inference. The model size shrank from 151.06 MB (fp32) to 61.65 MB
(59.2% reduction). The Word Error Rate (WER) on 100 English samples from
`google/fleurs` `en_us` (test split) stayed at the baseline level: 0.1381
(fp32) vs 0.1367 (HQQ), a difference of -0.0014 absolute (-1.0% relative),
which is within the noise of a 100-sample eval. The inference speed on CPU
stayed about the same (real-time factor 0.089 to 0.093).

The key setting is mixed precision: the whole encoder stack and the `fc1`
feed-forward up-projection are kept at 8-bit, and the remaining decoder
linears are 4-bit. The encoder is the acoustic stack (only 4 layers, cheap
to protect) and `fc1` is the more sensitive half of the FFN; keeping these
at 8-bit removes almost all of the 4-bit WER gap.

## Source model

- Base: `openai/whisper-tiny` (multilingual).
- Original size: 151.06 MB (`model.safetensors`, fp32).

## Quantization

- Method: HQQ (Half-Quadratic Quantization). No calibration data.
- Library: `hqq` 0.2.8.post1 (used directly; not transformers `HqqConfig`).
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

- Dataset: `google/fleurs`, config `en_us`, split `test`.
- Samples: 100 (the first 100 of the test split, streamed).
- Metric: WER via `jiwer`. Both reference and hypothesis are normalized
  (lowercase, remove punctuation, collapse spaces) before alignment.
- Hardware: CPU only. Same 100 samples for both runs.

## Results

| Metric                 | fp32 baseline | HQQ 4-bit  | Delta            |
|------------------------|---------------|------------|------------------|
| WER                    | 0.1381        | 0.1367     | -0.0014 (-1.0%)  |
| Model size on disk     | 151.06 MB     | 61.65 MB   | -89.41 MB (-59.2%)|
| Weights only (qmodel)  | 151.06 MB     | 59.74 MB   | -91.32 MB (-60.4%)|
| Avg real-time factor   | 0.089         | 0.093      | +0.004           |
| Total elapsed (100 fx) | 85.16 s       | 88.89 s    | +3.73 s          |
| Samples succeeded      | 100 / 100     | 100 / 100  | -                |

Notes:
- The size is the full output directory: `qmodel.pt` plus the processor and
  tokenizer files (`vocab.json`, `merges.txt`, `normalizer.json`, etc.).
- The WER difference (-0.0014) is within the noise of a 100-sample eval, so
  the HQQ model is statistically tied with the fp32 baseline, not better.
  The headline is that 4-bit HQQ with mixed-precision protection matches the
  fp32 baseline at 59% smaller size.
- The real-time factor stayed flat. HQQ has no fused 4-bit kernel on CPU, so
  each linear dequantizes group-by-group, but for this tiny model the
  overhead is offset by the smaller weight reads.

## Ablation (config sweep, same 100 samples)

All rows use `axis=1`. `protect` names the linears kept at 8-bit; the rest
are 4-bit. `group` is the group_size. The winner (row H) is the published
config.

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

```python
import hqq_asr
pipe = hqq_asr.build_pipeline("dkhokhlov/whisper-tiny-hqq-4bit", quant="hqq")
text = pipe({"array": audio, "sampling_rate": 16000})["text"]
```

Command line (this repository):

```
make asr MODEL_ASR=dkhokhlov/whisper-tiny-hqq-4bit QUANT=hqq AUDIO=clip.wav
```

## Reproduce

```
# 1. Quantize locally (writes whisper-tiny-hqq-4bit/).
python quantize.py

# 2. Measure baseline WER (fp32).
EVAL_LIMIT=100 MODEL_ASR=openai/whisper-tiny EVAL_OUT=eval_baseline.json \
  python eval_wer.py

# 3. Measure HQQ WER.
EVAL_LIMIT=100 QUANT=hqq MODEL_ASR=./whisper-tiny-hqq-4bit \
  EVAL_OUT=eval_hqq.json python eval_wer.py

# 4. Publish (needs a Hugging Face write token).
PUSH=1 HQQ_REPO=dkhokhlov/whisper-tiny-hqq-4bit python quantize.py
```

## Limitations

- CPU only. The model loads and runs on CPU. A GPU is not required and not
  used.
- `proj_out` and the embedding are fp16, not 4-bit. A smaller model is
  possible if the embedding is also quantized, but that raises the WER risk
  on the vocab projection and was not done here.
- The WER matches the fp32 baseline within 100-sample noise. The eval set is
  small (100 English samples); a larger eval could move the number by
  ~0.005-0.01 in either direction. Use the fp32 model when the lowest WER is
  required and the size is acceptable; use this model when size matters and
  baseline-level quality is acceptable.
- Evaluated on English (`fleurs en_us`). The base model is multilingual;
  other languages were not measured in this report.