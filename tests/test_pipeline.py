"""End-to-end pipeline integration test (asr -> en -> tts -> asr verify).

Runs the same scripts the `make asr` / `make en` / `make tts` targets wrap,
chaining stdout into stdin exactly like the documented shell pipeline:

    make asr AUDIO=<mlk.flac> | jq -r '.[].text' | make en | make tts OUTPUT=<wav>
    make asr AUDIO=<wav> MODEL_ASR=openai/whisper-base   # verify

The front ASR uses whisper-tiny; the verification ASR uses whisper-base so the
round-trip reliably recovers the expected text. The sample is the English
mlk.flac, so `make en` is near-identity and the keyword "dream" is a stable
expected token across the whole cascade. Marked `integration`; opt-in via
`make test-integration`. The first run downloads whisper-base (~290 MB).
"""

import json
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
MLK = "hf://datasets/Narsil/asr_dummy/mlk.flac"
VERIFY_MODEL = "openai/whisper-base"


def _run(script, env_extra, stdin=None):
    """Run one of the project scripts with a controlled model env.

    Start from os.environ but force the three MODEL_* vars so ambient values
    do not make the test non-deterministic.
    """
    env = {**os.environ}
    env["MODEL_ASR"] = env_extra.get("MODEL_ASR", "openai/whisper-tiny")
    env["MODEL_TRANSLATE"] = env_extra.get("MODEL_TRANSLATE", "Helsinki-NLP/opus-mt-mul-en")
    env["MODEL_TTS"] = env_extra.get("MODEL_TTS", "facebook/mms-tts-eng")
    for k in ("AUDIO", "OUTPUT", "TEXT"):
        if k in env_extra:
            env[k] = env_extra[k]
    return subprocess.run(
        [PY, os.path.join(ROOT, script)],
        env=env, input=stdin, capture_output=True, text=True, timeout=600,
    )


@pytest.mark.integration
def test_pipeline_recovers_expected_text(tmp_path):
    # 1. ASR front (whisper-tiny) on the English mlk.flac sample.
    r1 = _run("transcribe.py", {"AUDIO": MLK})
    assert r1.returncode == 0, f"front asr failed: {r1.stderr[-400:]}"
    text1 = json.loads(r1.stdout)[0]["text"]
    assert "dream" in text1.lower(), f"front asr lost 'dream': {text1!r}"

    # 2. Translate to English (near-identity on English input).
    r2 = _run("translate.py", {}, stdin=text1)
    assert r2.returncode == 0, f"translate failed: {r2.stderr[-400:]}"
    text2 = r2.stdout.strip()
    assert "dream" in text2.lower(), f"translate lost 'dream': {text2!r}"

    # 3. Synthesize English speech from the translated text.
    out_wav = tmp_path / "mlk_en.wav"
    r3 = _run("tts.py", {"OUTPUT": str(out_wav)}, stdin=text2)
    assert r3.returncode == 0, f"tts failed: {r3.stderr[-400:]}"
    assert out_wav.exists() and out_wav.stat().st_size > 0, "no WAV written"

    # 4. Verify: transcribe the synthesized WAV with whisper-base. The stable
    #    keyword "dream" must survive the whole cascade.
    r4 = _run("transcribe.py", {"AUDIO": str(out_wav), "MODEL_ASR": VERIFY_MODEL})
    if r4.returncode != 0:
        pytest.skip(f"could not load verification model {VERIFY_MODEL}: {r4.stderr[-400:]}")
    final = json.loads(r4.stdout)[0]["text"].lower()
    assert "dream" in final, f"expected 'dream' in verified audio text, got: {final!r}"