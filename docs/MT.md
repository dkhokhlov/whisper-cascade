# MT

The MT stage **translates text to English** with a [MarianMT](https://huggingface.co/docs/transformers/model_doc/marian) model. It is
**not quantized** — only the ASR stage is — so it loads and runs as float32.
This document covers model selection and output shape. For the cascade
overview and the ASR and TTS stages, see the [repo README](../README.md).

## Contents

- [Model](#model)
- [Output](#output)
- [Notes](#notes)

## Model

The default MT model is **`Helsinki-NLP/opus-mt-mul-en`** (multilingual to
English) — the `MODEL_TRANSLATE` variable. It is an **autoregressive
encoder-decoder** transformer, stored and loaded as **float32** from
`pytorch_model.bin` (the repo ships no safetensors for it), about
**310 MB** resident at run time.

Override it for a pair-specific source language (better quality than the
multilingual default when the source language is known):

```bash
make en TEXT="Hola, ¿cómo estás?" | jq -r '.text'        # -> "Hey, how are you?"
echo "Ich heiße Max." | make en | jq -r '.text'          # -> "My name is Max."
MODEL_TRANSLATE=Helsinki-NLP/opus-mt-es-en make en TEXT="Hola" | jq -r '.text'   # Spanish source
```

The target is named by the target language: `make en` runs `translate.py`.
Adding `make es` (translate to Spanish) later is one Make target that sets
`TARGET_LANG=es` and `MODEL_TRANSLATE` to a `-> Spanish` model — no code
change. For more info and other language pairs, see the
[Helsinki-NLP MarianMT collection](https://huggingface.co/Helsinki-NLP).

## Output

`make en` prints a **JSON object** to stdout with `text` (the English
translation), `lang` (`"en"`), `model`, and a `stats` block. Extract the text
with `jq -r '.text'` to pipe into `make tts`.

Stats:

- `chars`, `words` — input character and word counts.
- `out_chars`, `out_words` — output character and word counts.
- `elapsed_s` — translation wall-clock time in seconds.

`make en` cannot take a file argument (make treats a bare word as a target).
To read a file directly, run the script: `.venv/bin/python translate.py
file.txt` (or `-` for stdin).

## Notes

- `make en` needs [`sentencepiece`](https://github.com/google/sentencepiece) (MarianMT tokenizers); it is in the deps.
- The MT stage is **not quantized**; only the ASR stage is. It loads as
  float32 and its resident RAM (~310 MB) is the weights file size.
- The model id appears in the JSON output as `"model"`.
- Inspect the MarianMT block graphs with `make viz` →
  `build/viz/opus-mt-mul-en/encoder.svg` and `decoder.svg` (see the
  [repo README](../README.md#model-graphs)).