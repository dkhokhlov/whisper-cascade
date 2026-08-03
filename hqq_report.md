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
CPU inference. The model size shrank from 151.06 MB (fp32) to 54.86 MB
(63.7% reduction). The Word Error Rate (WER) rose from 0.1381 to 0.1622
(+0.0241 absolute, +17.45% relative) on 100 English samples from
`google/fleurs` `en_us` (test split). The inference speed on CPU stayed
about the same (real-time factor 0.089 to 0.088).

## Source model

- Base: `openai/whisper-tiny` (multilingual).
- Original size: 151.06 MB (`model.safetensors`, fp32).

## Quantization

- Method: HQQ (Half-Quadratic Quantization). No calibration data.
- Library: `hqq` 0.2.8.post1 (used directly; not transformers `HqqConfig`).
- Linear layers: 64 layers quantized to `nbits=4, group_size=64, axis=1`.
  These are the encoder and decoder attention projections (`q_proj`,
  `k_proj`, `v_proj`, `out_proj`) and feed-forward layers (`fc1`, `fc2`),
  plus the decoder cross-attention projections.
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
| WER                    | 0.1381        | 0.1622     | +0.0241 (+17.45%)|
| Model size on disk     | 151.06 MB     | 54.86 MB   | -96.20 MB (-63.7%)|
| Weights only (qmodel)  | 151.06 MB     | 52.96 MB   | -98.10 MB (-65.0%)|
| Avg real-time factor   | 0.089         | 0.088      | -0.001           |
| Total elapsed (100 fx) | 85.16 s       | 83.87 s    | -1.29 s          |
| Samples succeeded      | 100 / 100     | 100 / 100  | -                |

Notes:
- The size is the full output directory: `qmodel.pt` plus the processor and
  tokenizer files (`vocab.json`, `merges.txt`, `normalizer.json`, etc.).
- The real-time factor did not rise. HQQ has no fused 4-bit kernel on CPU,
  so each linear dequantizes group-by-group, but for this tiny model the
  overhead is offset by the smaller weight reads. The net CPU speed is
  about the same as fp32.
- The WER increase (13.81% to 16.22%) comes from the 4-bit linear weights.
  The embedding and `proj_out` add no error (fp16). Most errors are short
  function words and proper nouns, the same failure mode as the fp32 model,
  slightly more frequent.

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
- The WER is higher than the fp32 baseline. Use the fp32 model when the
  lowest WER is required and the size is acceptable; use this model when
  size matters more than the last 2.4 WER points.
- Evaluated on English (`fleurs en_us`). The base model is multilingual;
  other languages were not measured in this report.