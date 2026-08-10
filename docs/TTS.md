# TTS

The TTS stage **synthesizes English speech from text** with a [VITS](https://huggingface.co/docs/transformers/model_doc/vits) model. It
is **not quantized** — only the ASR stage is — so it loads and runs as
float32. This document covers model selection and output shape. For the
cascade overview and the ASR and MT stages, see the
[repo README](../README.md).

## Contents

- [Model](#model)
- [Output](#output)
- [Notes](#notes)

## Model

The default TTS model is **`facebook/mms-tts-eng`** (English) — the
`MODEL_TTS` variable. It is a **non-autoregressive** VITS model, stored and
loaded as **float32** from `model.safetensors`, about **145 MB** resident at
run time.

Swap in another MMS language with `MODEL_TTS`:

```bash
make tts TEXT="Hello world"                  # -> tts.wav
make tts TEXT="Hello" OUTPUT=hi.wav          # -> hi.wav
MODEL_TTS=facebook/mms-tts-spa make tts TEXT="Hola"   # another MMS language
```

## Output

`make tts` reads text from the `TEXT` env var or stdin and writes the
synthesized waveform to `OUTPUT` (default **`tts.wav`**). A **JSON summary**
goes to stdout with `output` (the wav path), `model`, `text`, and a `stats`
block.

The waveform is written with **`soundfile`** at
`model.config.sampling_rate`.

## Notes

- transformers 4.44.2 has **no text-to-speech pipeline**, so this calls the
  model directly (`AutoTokenizer` + `VitsModel`) and writes `out.waveform`
  with `soundfile` at `model.config.sampling_rate.
- The TTS stage is **not quantized**; only the ASR stage is. It loads as
  float32 and its resident RAM (~145 MB) is the weights file size.
- The model id appears in the JSON summary as `"model"`.