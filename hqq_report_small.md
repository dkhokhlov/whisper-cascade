---
license: mit
base_model: openai/whisper-small
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
  - gpu
  - onnx
---

# HQQ 4-bit Whisper-Small

**Includes fp16 + fp32 ONNX exports — the same HQQ model runs on CPU ONNX
Runtime with no HQQ runtime.** The fp16 export is a fp32-free graph for
fp32-less deployment hardware; the fp32 export matches the published benchmark
compute. HQQ ships no ONNX exporter; this repo adds one (see
[ONNX export](#onnx-export-cpu-onnx-runtime)).

Model card source for `dkhokhlov/whisper-small-hqq-4bit`.

## Related models

- [`dkhokhlov/whisper-tiny-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-tiny-hqq-4bit) — HQQ 4-bit, whisper-tiny (CPU eval)
- [`dkhokhlov/whisper-base-hqq-4bit`](https://huggingface.co/dkhokhlov/whisper-base-hqq-4bit) — HQQ 4-bit, whisper-base (CPU eval)
- Source model: [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) (fp32)
- Benchmark + code: [`dkhokhlov/whisper-cascade`](https://github.com/dkhokhlov/whisper-cascade)

## Summary

[`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
quantized with [HQQ](https://huggingface.co/docs/transformers/en/quantization/hqq)
4-bit grouped quantization. Resident weight RAM (fp16 compute, the deployment
mode) is 267.59 MB, 44.6% smaller than the unquantized fp16 model (483.47
MB). fp16 compute is WER-neutral; the published WER benchmark uses fp32
compute for cross-model comparability. The config is the same mixed-precision
setting tuned on `whisper-tiny` (whole encoder stack + `fc1` at 8-bit, rest
4-bit), applied to small without a separate sweep (see the repo README).

This is the first model in the set evaluated on a GPU for speed.
`whisper-tiny` and `whisper-base` were quantized and evaluated on CPU;
`whisper-small` (241.7 M parameters, 12+12 layers) was quantized and
evaluated on an NVIDIA A10 GPU (`ASR_DEVICE=cuda`). WER is host-independent;
only runtime is host-specific. The saved `qmodel.pt` is device-independent
and loads on CPU or GPU.

English (`fleurs` `en_us`, n=100) WER is 0.0636 vs 0.0660 fp32 (-3.6%),
within n=100 noise. HQQ is within 5% relative of fp32 on every tested config
(5 `fleurs` + 4 `talkbank`). `whisper-small` beats `whisper-base` on every
config.

## Results

English (`fleurs` `en_us`, n=100, fp32 compute):

| Metric | unquantized fp32 | HQQ 4-bit | Delta % |
|---|---|---|---|
| WER                    | 0.0660 | 0.0636 | -3.6% |
| Resident RAM (fp16)    | 483.47 MB | 267.59 MB | -44.6% |
| Samples succeeded      | 100 / 100 | 100 / 100 | - |

HQQ is within 5% relative of fp32 on every tested config. The full
multilingual and telephone WER tables, the cross-reference against
`whisper-tiny`/`whisper-base`, and the size-by-component breakdown are in
the repo [README](https://github.com/dkhokhlov/whisper-cascade#readme).

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

Command line (this repository, GPU venv):

```
ASR_DEVICE=cuda make asr MODEL_ASR=dkhokhlov/whisper-small-hqq-4bit QUANT=hqq AUDIO=clip.wav
```

## ONNX export (CPU ONNX Runtime)

This repo ships **two ONNX exports** of the same HQQ model, differing only in
compute dtype:

- **fp16 (default): `encoder_model.onnx` + `decoder_model_merged.onnx`** — the
  deployment compute. The graph is fp16-only (zero fp32 ops), for hardware with
  no fp32 unit. It uses eager attention so the attention scale stays a fp16
  `Mul` (SDPA would decompose it to `Sqrt`→`Div` in fp32). On CPU ONNX Runtime
  it runs slower than the fp32 export (ORT-CPU upcasts fp16 to fp32 internally)
  but loads less RAM, and it is the graph the deployment hardware imports.
- **fp32: `encoder_model-fp32.onnx` + `decoder_model_merged-fp32.onnx`** — the
  benchmark compute. Faster on CPU ORT; use it for host-side validation that
  matches the published fp32 WER benchmark.

Both keep the packed uint8 `W_q` and the per-group `scale`/`zero` as ONNX
initializers and emit the unpack + dequant as standard ONNX ops (opset 18), so
each graph carries the exact HQQ weights, not a re-dequantized dense copy.
Whisper is an encoder-decoder model, so each export is two ONNX graphs; the
autoregressive generation loop (argmax, KV-cache, EOS stop) runs in Python in
`ORTModelForSpeechSeq2Seq`, calling the encoder once and the decoder once per
token:

| File (fp16 / fp32) | Role | Input | Output | Runs |
|---|---|---|---|---|
| `encoder_model.onnx` / `encoder_model-fp32.onnx` | encoder | audio mel-spectrogram | hidden states | once per utterance |
| `decoder_model_merged.onnx` / `decoder_model_merged-fp32.onnx` | decoder | encoder hidden states + KV cache | next text token | once per token (loop) |

The merged decoder carries the no-past (first step) and with-past (cached
steps) branches behind one control-flow switch, so one session handles the
whole generation; the separate un-merged decoder files optimum emits are not
shipped.

Both exports reproduce the HQQ WER (0.0636). The fp32 export exact-matches the
HQQ manifest (0 mismatches); the fp16 export matches the WER and differs from
the fp32 manifest by at most a few case-only tokens (fp16-vs-fp32 rounding,
WER-neutral). The ONNX path runs on CPU (`make onnx` / `make eval-onnx` use
`.venv-onnx`), independent of the GPU eval used for the HQQ WER above.

Load via ONNX Runtime (the no-suffix files are the fp16 default):

```python
import hqq_asr
pipe = hqq_asr.build_pipeline("dkhokhlov/whisper-small-hqq-4bit", quant="onnx")
text = pipe({"array": audio, "sampling_rate": 16000})["text"]
```

Reproduce the fp16 export and the gate (CPU `.venv-onnx`; set
`HQQ_COMPUTE_DTYPE=fp32` for the fp32 export):

```
make onnx HQQ_REPO=dkhokhlov/whisper-small-hqq-4bit ONNX_OUT=build/whisper-small-hqq-onnx-fp16
make hqq-reference HQQ_REPO=dkhokhlov/whisper-small-hqq-4bit EVAL_OUT=build/hqq_reference_small_fp16.json
make eval-onnx ONNX_OUT=build/whisper-small-hqq-onnx-fp16 \
     HQQ_REFERENCE_MANIFEST=build/hqq_reference_small_fp16.json EVAL_OUT=build/eval_onnx_small_fp16.json
```

The export spec and the two validation gates are in `docs/onnx.md` in the repo.

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

## License

MIT. Derived from [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
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
evidence JSONs are committed under `eval_multilingual/` (prefix `small_`) and
`eval_telephone/` (prefix `small_`) in
[`dkhokhlov/whisper-cascade`](https://github.com/dkhokhlov/whisper-cascade).