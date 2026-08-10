"""Integration test for the HQQ safetensors export round-trip.

Locks in the qmodel.pt -> model.safetensors export and the reverse load:
the safetensors path drops the tied proj_out.weight at export (safetensors
cannot store shared tensors), records the tie in the header metadata for
host tooling, re-aliases proj_out -> embed_tokens in load_weights, and
load_whisper_hqq re-ties them. The loaded model must be bit-identical to the
qmodel.pt load (encoder output + generated ids) with proj_out tied to the
decoder embedding.

Also locks in two export invariants:

- The export reads qmodel.pt even when HQQ_FORMAT=safetensors is set and the
  existing safetensors is corrupt (the export forces the qmodel.pt source).
- The export is non-destructive: a failed qmodel.pt load leaves the last good
  safetensors in place (the export does not delete it before loading).
- The safetensors header records the proj_out -> embed_tokens tie metadata so
  a host loader (no torch) can reconstruct proj_out.weight.

Marked `integration` (opt-in via `make test-integration`): it quantizes a
real whisper-tiny, so it needs the model in the HF cache and is slow. The
default `make test` excludes it.
"""

import glob
import os
import re

import pytest
import torch
from safetensors import safe_open

import export_safetensors
import hqq_asr


# Skip only for environmental (network/cache) errors; re-raise anything else
# so a real logic bug fails the test instead of being silently skipped.
_ENV_ERR = re.compile(
    r"(connection|offline|local entry|resolve|hostname|timeout|trust_remote"
    r"|unreachable|temporarily unavailable|huggingface|hf_hub|http error)",
    re.IGNORECASE,
)


def _skip_if_env(exc):
    if _ENV_ERR.search(str(exc)) or isinstance(exc, (OSError, ConnectionError)):
        pytest.skip(f"environment/model-load error: {exc}")
    raise exc


def _cached_whisper_tiny():
    return glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--openai--whisper-tiny/snapshots/*/config.json"
        )
    )


@pytest.mark.integration
def test_safetensors_roundtrip(monkeypatch, tmp_path):
    if not _cached_whisper_tiny():
        pytest.skip("openai/whisper-tiny not in HF cache")

    out = tmp_path / "hqq"
    try:
        hqq_asr.quantize_whisper(
            "openai/whisper-tiny", str(out),
            nbits=4, group_size=32, axis=1,
            tier8_nbits=8, tier8_patterns=("encoder.layers", "fc1"),
            multilingual=True, device="cpu",
        )
        export_safetensors.export_safetensors(str(out))
    except Exception as exc:
        _skip_if_env(exc)

    st_path = out / "model.safetensors"
    assert (out / "qmodel.pt").exists()
    assert st_path.exists()

    # P2-2: the proj_out -> embed_tokens tie is in the header metadata so a
    # host loader (no torch) can reconstruct proj_out.weight.
    with safe_open(str(st_path), framework="pt") as f:
        meta = f.metadata()
    assert meta is not None, "safetensors header has no metadata"
    assert meta.get("proj_out.weight.tied_to") == "model.decoder.embed_tokens.weight"

    mel = torch.randn(1, 80, 3000, dtype=hqq_asr.COMPUTE_DTYPE)

    # Load the qmodel.pt source first (HQQ_FORMAT unset) as the reference.
    m_pt = hqq_asr.load_whisper_hqq(str(out), device="cpu")
    m_pt.eval()
    with torch.no_grad():
        enc_pt = m_pt.model.encoder(mel).last_hidden_state
        gen_pt = m_pt.generate(mel, max_new_tokens=16, do_sample=False)

    # P2-1 source invariant: the export must read qmodel.pt, not the existing
    # safetensors, even when HQQ_FORMAT=safetensors is set. Corrupt the
    # safetensors so reading it would fail, then re-export under
    # HQQ_FORMAT=safetensors; a qmodel.pt source succeeds and overwrites the
    # corrupt file. (A pre-F1 export that read the stale safetensors would
    # raise here instead.)
    st_path.write_bytes(b"not a safetensors file")
    monkeypatch.setenv("HQQ_FORMAT", "safetensors")
    export_safetensors.export_safetensors(str(out))  # must not raise

    # Reload the re-exported safetensors and verify the lossless round-trip.
    m_st = hqq_asr.load_whisper_hqq(str(out), device="cpu")
    m_st.eval()
    with torch.no_grad():
        enc_st = m_st.model.encoder(mel).last_hidden_state
        gen_st = m_st.generate(mel, max_new_tokens=16, do_sample=False)

    # proj_out is re-tied to the decoder embedding in the safetensors load.
    assert m_st.proj_out.weight.data_ptr() == m_st.model.decoder.embed_tokens.weight.data_ptr()
    # Lossless round-trip: bit-identical to the qmodel.pt load.
    assert torch.equal(enc_pt, enc_st), "safetensors encoder output differs from qmodel.pt"
    assert torch.equal(gen_pt, gen_st), "safetensors generated ids differ from qmodel.pt"

    # P2-1 non-destructive invariant: if the qmodel.pt load fails, the existing
    # valid safetensors is preserved (the export does not delete it first).
    # Corrupt qmodel.pt, force the qmodel.pt source, and confirm the export
    # raises while the valid safetensors survives untouched.
    good_st_size = st_path.stat().st_size
    qm_path = out / "qmodel.pt"
    qm_path.write_bytes(b"not a qmodel file")
    monkeypatch.delenv("HQQ_FORMAT", raising=False)
    with pytest.raises(Exception):
        export_safetensors.export_safetensors(str(out))
    assert st_path.exists(), "export deleted the safetensors before the failed load"
    assert st_path.stat().st_size == good_st_size, "export clobbered the safetensors on a failed load"