"""Unit test for the HQQ three-tier linear categorization.

Builds a tiny fake module with named nn.Linear children that exercise all
three tiers (exempt / 8-bit tier / 4-bit tier), calls _patch_linears, and
asserts the counts and per-linear bit width. No real Whisper model is loaded;
this locks in the tier8_* vocabulary and the proj_out exemption. Fast unit
test (no model download, no network).
"""

import torch.nn as nn

import hqq_asr
from hqq.core.quantize import HQQLinear, BaseQuantizeConfig


class _Sub(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.out_proj = nn.Linear(64, 64)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Sub()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)


class _Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _Dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])
        self.embed_tokens = nn.Embedding(10, 64)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Enc()
        self.decoder = _Dec()


class _Fake(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_out = nn.Linear(64, 10)  # exempt (the tied lm_head)
        self.model = _Inner()


def _nbits(module):
    return module.quant_config["weight_quant_params"]["nbits"]


def test_patch_linears_three_tiers():
    fake = _Fake()
    default_cfg = BaseQuantizeConfig(nbits=4, group_size=32, axis=1)
    tier8_cfg = BaseQuantizeConfig(nbits=8, group_size=32, axis=1)

    n_default, n_tier8 = hqq_asr._patch_linears(
        fake, default_cfg, tier8_cfg, device="cpu",
        tier8_patterns=("encoder.layers", "fc1"),
    )

    # 8-bit tier (matched a pattern): encoder self_attn q/k/v/out (4) +
    # encoder fc1 (1) + encoder fc2 (1) + decoder fc1 (1) = 7.
    # 4-bit tier (no match): decoder self_attn q/k/v/out (4) + decoder fc2 (1)
    # = 5. proj_out is exempt (not counted).
    assert (n_default, n_tier8) == (5, 7)

    # proj_out is exempt: still a plain nn.Linear, not quantized.
    assert type(fake.proj_out) is nn.Linear

    # 8-bit tier: encoder self_attn q_proj matched "encoder.layers"; decoder
    # fc1 matched "fc1".
    assert isinstance(fake.model.encoder.layers[0].self_attn.q_proj, HQQLinear)
    assert _nbits(fake.model.encoder.layers[0].self_attn.q_proj) == 8
    assert isinstance(fake.model.decoder.layers[0].fc1, HQQLinear)
    assert _nbits(fake.model.decoder.layers[0].fc1) == 8

    # 4-bit tier: decoder self_attn q_proj matched no pattern.
    assert isinstance(fake.model.decoder.layers[0].self_attn.q_proj, HQQLinear)
    assert _nbits(fake.model.decoder.layers[0].self_attn.q_proj) == 4

    # Embedding is not a Linear, so _patch_linears leaves it untouched (the
    # fp16 cast happens elsewhere in quantize_whisper).
    assert type(fake.model.decoder.embed_tokens) is nn.Embedding