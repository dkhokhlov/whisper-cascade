---
license: mit
base_model: openai/whisper-base
base_model_relation: quantized
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
  - onnx
---

# HQQ 4-bit Whisper-Base

**Includes an ONNX export — the same HQQ model runs on CPU ONNX Runtime with no
HQQ runtime.** HQQ ships no ONNX exporter; this repo adds one (see
[ONNX export](#onnx-export-cpu-onnx-runtime)).

Model card source for `dkhokhlov/whisper-base-hqq-4bit`.

## Related models

- [`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit) — HQQ 4-bit, whisper-tiny (CPU eval)
- [`dkhokhlov/whisper-small-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-small-hqq-4bit) — HQQ 4-bit, whisper-small (A10 GPU eval)
- Source model: [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) (fp32)
- Benchmark + code: [`dkhokhlov/whisper-cascade`](https://github.com/dkhokhlov/whisper-cascade)

## Summary

[`openai/whisper-base`](https://huggingface.co/openai/whisper-base) quantized
with [HQQ](https://huggingface.co/docs/transformers/en/quantization/hqq) 4-bit
grouped quantization for CPU inference. Resident weight RAM (fp16 compute, the
deployment mode) is 97.22 MB, 33.0% smaller than the unquantized fp16 model
(145.19 MB). fp16 compute is WER-neutral; the published WER benchmark uses
fp32 compute for cross-model comparability. The config is the same
mixed-precision setting tuned on `whisper-tiny` (whole encoder stack + `fc1`
at 8-bit, rest 4-bit), applied to base without a separate sweep (see the repo
README).

English (`fleurs` `en_us`, n=100) WER is 0.0995 vs 0.0985 fp32 (+1.0%),
within n=100 noise. HQQ is within 5% relative of fp32 on every tested config
(5 `fleurs` + 4 `talkbank`). `whisper-base` beats `whisper-tiny` on every
config except Hindi (both not usable).

## Results

English (`fleurs` `en_us`, n=100, fp32 compute):

| Metric | unquantized fp32 | HQQ 4-bit | Delta % |
|---|---|---|---|
| WER                    | 0.0985 | 0.0995 | +1.0% |
| Resident RAM (fp16)    | 145.19 MB | 97.22 MB | -33.0% |
| Samples succeeded      | 100 / 100 | 100 / 100 | - |

HQQ is within 5% relative of fp32 on every tested config. The full
multilingual and telephone WER tables, the cross-reference against
`whisper-tiny`/`whisper-small`, and the size-by-component breakdown are in
the repo [README](https://github.com/dkhokhlov/whisper-cascade#readme).

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

## ONNX export (CPU ONNX Runtime)

This repo also ships an **ONNX** export of the same HQQ model. HQQ itself ships
no ONNX exporter, so this is a unique addition: the quantized model runs on
CPU ONNX Runtime with no HQQ runtime dependency. The export keeps the packed
uint8 `W_q` and the per-group `scale`/`zero` as ONNX
initializers and emits the unpack + dequant as standard ONNX ops (opset 18),
so the graph carries the exact HQQ weights (and the measured WER), not a
re-dequantized dense copy. Whisper is an encoder-decoder model, so the export
is two ONNX graphs; the autoregressive generation loop (argmax, KV-cache, EOS
stop) runs in Python in `ORTModelForSpeechSeq2Seq`, calling the encoder once
and the decoder once per token:

| File | Role | Input | Output | Runs |
|---|---|---|---|---|
| `encoder_model.onnx` | encoder | audio mel-spectrogram | hidden states | once per utterance |
| `decoder_model_merged.onnx` | decoder | encoder hidden states + KV cache | next text token | once per token (loop) |

The merged decoder carries the no-past (first step) and with-past (cached
steps) branches behind one control-flow switch, so one session handles the
whole generation; the separate un-merged decoder files optimum emits are not
shipped.

The export reproduces the HQQ output exactly. An exact-text gate over 100
`fleurs` `en_us` samples compares the ONNX transcript to the HQQ manifest and
returns `exact_match=true` with 0 mismatches (WER 0.0995, identical to HQQ).

Load via ONNX Runtime:

```python
import hqq_asr
pipe = hqq_asr.build_pipeline("dkhokhlov/whisper-base-hqq-4bit", quant="onnx")
text = pipe({"array": audio, "sampling_rate": 16000})["text"]
```

Reproduce the export and the gate:

```
make onnx HQQ_REPO=dkhokhlov/whisper-base-hqq-4bit ONNX_OUT=build/whisper-base-hqq-onnx
make hqq-reference HQQ_REPO=dkhokhlov/whisper-base-hqq-4bit EVAL_OUT=build/hqq_reference_base.json
make eval-onnx ONNX_OUT=build/whisper-base-hqq-onnx \
     HQQ_REFERENCE_MANIFEST=build/hqq_reference_base.json EVAL_OUT=build/eval_onnx_base.json
```

The export spec and the two validation gates are in `docs/onnx.md` in the repo.

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

# 4. Telephone benchmark (talkbank segment split).
EVAL_DATASET=diabolocom/talkbank_4_stt EVAL_CONFIG=en EVAL_SPLIT=segment EVAL_LIMIT=100 \
  MODEL_ASR=openai/whisper-base EVAL_OUT=base_talkbank_en_fp32.json python eval_wer.py

# 5. Publish (needs a Hugging Face write token).
PUSH=1 HQQ_REPO=dkhokhlov/whisper-base-hqq-4bit MODEL_ASR=openai/whisper-base \
  HQQ_OUT=whisper-base-hqq-4bit python quantize.py
```

## License

MIT. Derived from [`openai/whisper-base`](https://huggingface.co/openai/whisper-base)
(Apache-2.0) and [HQQ](https://github.com/mobiusml/hqq). The quantized
weights inherit the openai/whisper license terms.

## Citation

See the repo [README](https://github.com/dkhokhlov/whisper-cascade#citation)
for the BibTeX entry.

## Full details

Quantization config, config-sweep ablation, safetensors format, the full WER
tables (multilingual `fleurs`, `talkbank` telephone, cross-reference), and the
resident-RAM-by-component breakdown are in the repo
[README](https://github.com/dkhokhlov/whisper-cascade#readme). Per-config WER
evidence JSONs are committed under `eval_multilingual/` (prefix `base_`) and
`eval_telephone/` (prefix `base_`) in
[`dkhokhlov/whisper-cascade`](https://github.com/dkhokhlov/whisper-cascade).