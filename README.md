# wisper_test

A CPU-only Docker container that transcribes audio with the
`openai/whisper-tiny.en` model and prints the text to stdout as JSON.

The container uses the Hugging Face `transformers` automatic-speech-recognition
pipeline. The model is cached at build time, so the container runs offline.

## Build

```bash
docker build -t wisper-test .
```

## Make targets

Run `make` (no target) to print this help.

| Target | Description |
| --- | --- |
| `make info` | Show current config and status (image, model, venv, samples). |
| `make build` | Build the docker image. |
| `make run` | Transcribe the baked samples in the container. |
| `make run-file FILE=x.wav` | Transcribe a host file (mounted into the container). |
| `make offline` | Run the container with no network. |
| `make validate` | Validate the container JSON output. |
| `make venv` | Create the local `.venv` with uv. |
| `make samples` | Download the FLAC test samples. |
| `make local` | Transcribe the samples with the local `.venv` (no docker). |
| `make ml` | Transcribe the multilingual samples (ES + HI) with `whisper-tiny` locally. |
| `make clean` | Remove the local `.venv` and `samples/`. |
| `make image-rm` | Remove the built docker image. |

## Samples

The baked samples come from the Hugging Face dataset `Narsil/asr_dummy`:

- English (default `make run`): `mlk.flac`, `1.flac`, `2.flac`.
- Multilingual (only under a multilingual model): `4.flac` (Spanish), `hindi.ogg` (Hindi).

The default model is `whisper-tiny.en` (English-only), so the default `make run`
transcribes only the English samples. To use the multilingual samples:

```bash
make ml                                                    # local .venv
make build MODEL=openai/whisper-tiny                        # docker: bake multilingual model
docker run --rm wisper-test /app/samples/4.flac /app/samples/hindi.ogg
```

## Model override

The model defaults to `openai/whisper-tiny.en`. Override it with the `MODEL`
variable. The chosen model is baked into the image at build time, so the
container still runs offline.

```bash
make build MODEL=openai/whisper-base      # docker: bake a different model
make local MODEL=openai/whisper-base       # local .venv: use a different model
```

The model id appears in each JSON element as `"model"`. For docker, the
override is build-time (the model is pre-cached); for local runs, it is
runtime (the model is downloaded on first use).

## Run

Transcribe the sample clips that are baked into the image (FLAC from the
Hugging Face dataset `Narsil/asr_dummy`):

```bash
docker run --rm wisper-test
```

Transcribe a WAV file on the host:

```bash
docker run --rm -v "$(pwd)/my.wav:/work/x.wav" wisper-test /work/x.wav
```

Transcribe several files:

```bash
docker run --rm -v "$(pwd):/work" wisper-test /work/a.wav /work/b.flac
```

## Output

The container prints one JSON array to stdout. Each element has the keys
`file`, `text`, and `model`:

```json
[
  {"file": "/app/samples/mlk.flac", "text": "...", "model": "openai/whisper-tiny.en"}
]
```

When a file fails, its element has the key `error` instead of `text`, and the
container exits with status 1. The other files are still processed.

## Verify the JSON

```bash
docker run --rm wisper-test | python3 -m json.tool
```

## Run offline

```bash
docker run --rm --network=none wisper-test
```

The model is cached in the image, so no network access is necessary.

## Local development with uv

A `.venv` is used for local development:

```bash
uv sync                  # install deps into .venv (torch from the CPU index)
bash download_samples.sh # fetch the FLAC samples into ./samples
.venv/bin/python transcribe.py ./samples/mlk.flac
```

## Notes

- The image is CPU only. There is no GPU dependency.
- The pipeline resamples each input to 16 kHz mono. You do not have to convert
  the input before you give it to the container. WAV, FLAC, MP3, and OGG are
  accepted.
- The baked samples are FLAC, not WAV. This matches the Hugging Face Whisper
  example and keeps the image small. The container still accepts WAV input.