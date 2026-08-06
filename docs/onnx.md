# ONNX export of the HQQ Whisper models

This document is the spec for exporting the HQQ-quantized Whisper models to ONNX so they
run on CPU in ONNX Runtime and reproduce the HQQ transcription results.

Status: **all three flavors (tiny, base, small) pass both gates on the `onnx` branch**
(not merged; ONNX files published to the HQQ repos after user approval).

## Goal

ONNX Runtime on CPU reproduces the HQQ result. The canonical target for whisper-tiny is
`eval_hqq.json`: WER **0.1367** on `google/fleurs` `en_us` `test`, n=100, CPU, auto-detect
language, with the post-hoc gzip loop guard in `eval_wer.py`.

Two gates, kept separate:

1. **Packed-weight graph proof (structural)** — the raw `.onnx` protobuf, inspected before any ORT
   session, contains packed `uint8` `W_q` initializers and the dequant ops (`BitShift`/
   `BitwiseAnd` or `Div`/`Mod`, `Cast`/`Sub`/`Mul`/`Reshape`/`Transpose`/`MatMul`). If they are
   absent, the exporter folded the dequant to dense at export time and the file is a dense
   export — WER may still pass, but the weights are no longer packed.
2. **WER / text proof (functional)** — exact `output["text"]` match on all 100 samples vs a
   full 100-sample reference manifest (`eval_hqq.json` stores only 5, so it cannot gate this);
   WER == 0.1367 secondary. Logit/submodel tolerance `atol=1e-5, rtol=1e-5`; the text gate is
   exact.

Compact runtime RAM (the packed-weight export's reason to exist over a plain dense export) is a later
optimization, not part of these gates.

## Results (all flavors)

Both gates pass on the `onnx` branch. Settings: `google/fleurs` `en_us` `test`, n=100,
CPU ONNX Runtime, auto-detect language, post-hoc gzip loop guard, greedy
(`do_sample=False`). The ONNX run uses the same processor and generation config as the HQQ
run, so the texts are comparable.

| Flavor | HQQ repo | Gate 1 (packed-weight graph) | Gate 2 (exact text, 100 samples) | ONNX WER | HQQ WER |
|---|---|---|---|---|---|
| tiny  | `dkhokhlov/whisper-tiny-hqq-4bit`  | PASS | PASS, 0 mismatches | 0.1367 | 0.1367 |
| base  | `dkhokhlov/whisper-base-hqq-4bit`  | PASS | PASS, 0 mismatches | 0.0995 | 0.0995 |
| small | `dkhokhlov/whisper-small-hqq-4bit` | PASS | PASS, 0 mismatches | 0.0636 | 0.0636 |

For every flavor the ONNX WER is identical to the HQQ WER (the exact-text gate proves this:
zero mismatches implies WER equality). The HQQ math and the ONNX op chain are identical
across flavors; only the layer/linear counts differ.

### whisper-tiny (commit `dbc95c9`)

1. **Packed-weight graph proof (structural): PASS.** The raw `.onnx` protobuf, inspected before any
   ORT session, contains the packed `uint8` `W_q` initializers, the dequant chain
   (`Cast`/`Sub`/`Mul`/`Reshape`/`Transpose`/`MatMul`), and the 4-bit unpack ops
   (`BitwiseAnd`/`BitShift`) in every exported subgraph. The export did not fold the dequant to
   dense at export time (`do_constant_folding=False`). Shipped files: `encoder_model.onnx`
   and `decoder_model_merged.onnx`. optimum also emits the un-merged decoder pair
   (`decoder_model.onnx` + `decoder_with_past_model.onnx`); `export_onnx.py` trims them after
   the gate-1 check because `ORTModelForSpeechSeq2Seq` loads the merged decoder (the merged
   graph carries the no-past and with-past branches behind one control-flow switch), so the
   un-merged pair is redundant and roughly doubles the decoder disk footprint. Whisper is an
   encoder-decoder model, so two graphs (encoder + decoder) are the minimum; the
   autoregressive generation loop runs in Python in `ORTModelForSpeechSeq2Seq`.
2. **WER / text proof (functional): PASS.** The ONNX run reproduces the HQQ result exactly.
   - Exact `output["text"]` match on all 100 samples vs the full 100-sample HQQ reference
     manifest (`hqq_reference.json`, written by `make hqq-reference`). Zero mismatches.
   - WER **0.1367**, identical to the HQQ baseline (`eval_hqq.json`).

### whisper-base

1. **Packed-weight graph proof (structural): PASS.** Packed `uint8` `W_q` + dequant chain + 4-bit
   unpack ops present in every subgraph (encoder 763, decoder 1863, with_past 1563 nodes).
2. **WER / text proof (functional): PASS.** Exact `output["text"]` match on all 100 samples
   vs `hqq_reference_base.json`; 0 mismatches. WER **0.0995**, identical to HQQ.

### whisper-small

1. **Packed-weight graph proof (structural): PASS.** Packed `uint8` `W_q` + dequant chain + 4-bit
   unpack ops present in every subgraph (encoder 1501, decoder 3651, with_past 3062 nodes).
2. **WER / text proof (functional): PASS.** Exact `output["text"]` match on all 100 samples
   vs `hqq_reference_small.json`; 0 mismatches. WER **0.0636**, identical to HQQ.

Reproduce (per flavor; tiny shown):

```bash
make onnx-venv                       # .venv-onnx (pinned optimum/onnxruntime set)
make onnx HQQ_REPO=dkhokhlov/whisper-tiny-hqq-4bit ONNX_OUT=build/whisper-tiny-hqq-onnx
make hqq-reference HQQ_REPO=dkhokhlov/whisper-tiny-hqq-4bit EVAL_OUT=build/hqq_reference_tiny.json
make eval-onnx ONNX_OUT=build/whisper-tiny-hqq-onnx \
     HQQ_REFERENCE_MANIFEST=build/hqq_reference_tiny.json EVAL_OUT=build/eval_onnx_tiny.json
# eval_onnx_tiny.json -> exact_text_gate.exact_match == true, wer == 0.1367
```

### Known validation false alarm (cached decoder step)

optimum's built-in export validation flags the cached decoder (`decoder_with_past_model.onnx`
/ `decoder_model_merged.onnx`) with large KV-cache diffs (up to 13.4 in logits). This is a
false alarm, not a defect. The legacy exporter traces the `if sequence_length != 1` branch
(`transformers/models/whisper/modeling_whisper.py:100`) as a constant, and optimum's
validation feeds dummy shapes that break this assumption. `ORTModelForSpeechSeq2Seq` always
feeds `sequence_length == 1` for cached steps, so inference is exact. Gate 2 (the 100-sample
exact-text match) confirms it. The export validation `atol` is set to `1e-3` because
ORT-vs-torch fp32 reassociation over the 4-layer decoder is about `2e-4`, above the default
`1e-4`; the real gate is the exact-text match, not this sanity check.

## Why keep the packed weights

HQQ ships no ONNX exporter. A dense export (dequantize all weights to fp32, then export)
loses the quantization and the compact footprint. Re-quantizing inside ONNX Runtime is a
different, unmeasured quantization. This export instead keeps the packed `uint8` `W_q` +
per-group scale/zero as ONNX tensors and emits the unpack + dequant as ONNX ops, so the
exact HQQ weights and the measured WER carry over. Codex reviewed the bit math and
confirmed the approach is viable for WER correctness.

## Verified HQQ dequant math (hqq 0.2.8.post1)

Both tiers share the formula `W = (q - zero) * scale`, `Reshape` to `(O, I)`, then
`out = x @ W.T (+ bias)`. `axis=1, group_size=32`, `compute_dtype=fp32` (this repo overrides
hqq's fp16 default).

- **4-bit** (`nbits=4`): `W_q` uint8 `(O*I/64, 32)`; the pre-pack tensor `(O*I/32, 32)` is
  split into two halves along axis 0 — first half -> high nibble, second half -> low nibble.
- **8-bit** (`nbits=8`): `W_q` uint8 `(O*I/32, 32)`, 1:1, values `0..255`; no bit-packing.
- `scale`, `zero`: `(O*I/32, 1)`, broadcast across the 32 group columns.
- `proj_out` exempt (plain fp16 `nn.Linear`, tied to `decoder.embed_tokens`); embedding,
  convs, norms are fp16, upcast to fp32 at compute.

## Diagrams

### 1. Whisper-tiny three-tier architecture

```
whisper-tiny  (AutoModelForSpeechSeq2Seq; fp16 storage, fp32 compute)

encoder                                    : 8-BIT TIER (nbits=8, group=32, axis=1)
  conv1, conv2                             : exempt (fp16)
  layers x4:
    self_attn {q,k,v,out}_proj              : 8-bit
    fc1                                     : 8-bit
    fc2                                     : 8-bit

decoder
  embed_tokens = proj_out (tied)           : EXEMPT (one fp16 weight; re-tied on load)
  layers x4:
    self_attn {q,k,v,out}_proj              : 4-BIT TIER (nbits=4, group=32, axis=1)
    encoder_attn {q,k,v,out}_proj (cross)   : 4-bit
    fc1                                     : 8-bit  (matches "fc1")
    fc2                                     : 4-bit
  positional_embedding, layer_norms         : exempt (fp16)

proj_out (lm_head)                         : exempt (tied to embed_tokens, fp16)

tiny linear counts: 8-bit = 28, 4-bit = 36, exempt linear = 1 (proj_out)
selection: HQQ_8BIT_PATTERNS = "encoder.layers,fc1"  (substring match on module name)
```

### 2. 4-bit tier dequant ONNX op graph (per HQQLinearONNX forward)

New ONNX ops this path introduces: `BitwiseAnd`, `BitShift`, `Cast`, `Concat`, `Sub`, `Mul`,
`Reshape`, `Transpose`, `MatMul`, `Add`. `W_q` / `zero` / `scale` / `bias` are graph
initializers (constants).

```
W_q  (uint8 initializer, shape [O*I/64, 32])
  |-> BitwiseAnd(& 0xF0) -> BitShift(>> 4) -> Cast(u8 -> fp32)   = high
  |
  |-> BitwiseAnd(& 0x0F) ----------------------> Cast(u8 -> fp32) = low

Concat([high, low], axis=0) -> W_r  [O*I/32, 32]  (fp32)
  -> Sub( - zero [O*I/32, 1] )      broadcast over the 32 group cols
  -> Mul( * scale [O*I/32, 1] )     broadcast over the 32 group cols
  -> Reshape -> (O, I)
  -> Transpose(0, 1) -> W [I, O]
  -> MatMul(x, W)                   x: [..., in_features]
  -> Add bias [O,]  (if bias is not None)
  -> out
```

Fallback if torch 2.4.1's bitwise ops do not map to ONNX: replace `BitwiseAnd(& 0xF0)` +
`BitShift(>> 4)` with `Div(q, 16, floor)` and `BitwiseAnd(& 0x0F)` with `Remainder(q, 16)`.

### 3. 8-bit tier dequant ONNX op graph

No bit-packing (1:1, one weight per uint8 byte), so no `BitwiseAnd` / `BitShift`:

```
W_q  (uint8 initializer, shape [O*I/32, 32], values 0..255)
  -> Cast(u8 -> fp32)
  -> Sub( - zero [O*I/32, 1] )      broadcast over the 32 group cols
  -> Mul( * scale [O*I/32, 1] )     broadcast over the 32 group cols
  -> Reshape -> (O, I)
  -> Transpose(0, 1) -> W [I, O]
  -> MatMul(x, W)
  -> Add bias [O,]  (if bias is not None)
  -> out
```

### 4. Export and run pipeline

```
+------------------------------------------------------------------------+
| export (offline, .venv-onnx)                                           |
+------------------------------------------------------------------------+
|                                                                        |
| HQQ_OUT (qmodel.pt)                                                    |
|    v   load_whisper_hqq(): re-tie proj_out, fp32 compute, gen-config   |
| HQQ model (HQQLinear linears)                                          |
|    v   swap HQQLinear -> HQQLinearONNX (W_q/zero/scale/bias copied)    |
| swapped model                                                          |
|    v   onnx_export_from_model(fn_get_submodels, custom_onnx_configs)   |
|        dynamo=False, do_constant_folding=False, opset=18               |
| ONNX_OUT (shipped, 2 files):                                           |
|    encoder_model.onnx, decoder_model_merged.onnx,                      |
|    config.json, processor files, generation_config.json                |
| (optimum also emits decoder_model.onnx + decoder_with_past_model.onnx; |
|  export_onnx.py trims them: ORTModelForSpeechSeq2Seq loads the merged) |
+------------------------------------------------------------------------+
              HQQ_OUT is read-only (the exporter never writes it)
                                  v
+------------------------------------------------------------------------+
| run (CPU ONNX Runtime, .venv-onnx)                                     |
+------------------------------------------------------------------------+
|                                                                        |
| ONNX_OUT -> ORTModelForSpeechSeq2Seq -> ASR pipeline (quant=onnx)      |
|                                                                        |
| audio array -> pipe({array, sampling_rate:16000}) -> output[text]      |
+------------------------------------------------------------------------+
```

### 5. Two-gate validation flow

```
+-------------------------------------------------------+
| GATE 1 -- packed-weight graph proof (structural)      |
+-------------------------------------------------------+
| inspect raw .onnx protobuf BEFORE any ORT run         |
| require: uint8 W_q initializer (packed)               |
|          BitShift / BitwiseAnd (or Div/Mod)           |
|          Cast, Sub, Mul, Reshape, Transpose, MatMul   |
| if absent -> export folded to dense -> STOP           |
+-------------------------------------------------------+
            | pass
            v
+-------------------------------------------------------+
| GATE 2 -- WER / text proof (functional)               |
+-------------------------------------------------------+
| 100-sample fleurs en_us reference manifest            |
| exact output[text] match (post loop guard)            |
| WER == 0.1367  (secondary)                            |
+-------------------------------------------------------+
```

## Approach

1. Hand-write `HQQLinearONNX(nn.Module)` (do NOT trace hqq's `dequantize()`): buffers `W_q`
   (uint8), `zero`/`scale`/`bias` (fp32); nibble math via `torch.bitwise_and` /
   `torch.bitwise_right_shift` / `torch.cat`, `uint8` for bitwise, cast `fp32` before sub/mul,
   `Reshape` to `(O, I)`, `MatMul(x, W.T)` (rank-3 inputs), `+ bias`.
2. Swap every `HQQLinear` with `HQQLinearONNX` (copy `W_q`/`zero`/`scale`/`bias`); leave exempt
   modules. Keep an un-swapped HQQ reference for validation.
3. Export with optimum's `onnx_export_from_model(..., fn_get_submodels=...,
   custom_onnx_configs=...)` (one `export()` call is one graph; it does not auto-split). Legacy
   exporter: `dynamo=False, do_constant_folding=False, opset=18`. Canonical filenames
   `encoder_model.onnx` / `decoder_model.onnx` / `decoder_with_past_model.onnx`. Generation loop
   stays in the ORT wrapper.
4. Load via `ORTModelForSpeechSeq2Seq` + the transformers ASR pipeline, exposed through a new
   `quant="onnx"` branch in `build_pipeline`. `eval_wer.py` / `transcribe.py` work unchanged
   with `QUANT=onnx`.
5. Gate on exact text match over the 100-sample manifest; WER == 0.1367.

## Pinned dependencies (`.venv-onnx`)

```
optimum-onnx[onnxruntime]==0.0.3     # declares transformers>=4.36,<4.56, optimum~=2.0.0
optimum==2.0.0
transformers==4.44.2
torch==2.4.1            # CPU wheel index
torchaudio==2.4.1       # CPU wheel index
onnx==1.16.2
onnxruntime==1.19.2
hqq==0.2.8.post1
# + repo runtime deps: soundfile==0.12.1 "numpy<2" sentencepiece==0.2.0 jiwer datasets
```

Use `optimum-onnx==0.0.3`, not 0.1.0 (which moves to `optimum~=2.1.0` and adds version drift
for this legacy transformers). Do not leave `optimum-onnx` or `hqq` unpinned.

## Build order (tiny-first)

1. Create the `onnx` branch; `make onnx-venv` (the pinned set above).
2. Op probe: export `HQQLinearONNX` for a random `uint8` tensor, opset 18, run in ORT CPU.
   Confirm `BitShift`/`BitwiseAnd` on `uint8` map and run; else switch to `Div`/`Mod`.
3. Dequant equivalence unit test: `HQQLinearONNX` vs `hqq_linear.dequantize()` and
   `hqq_linear(x)`, rank-2 and rank-3, both tiers, real layers.
4. `export_onnx.py`: load tiny, swap, export the three graphs (canonical filenames, legacy
   exporter).
5. **Gate 1**: inspect the raw `.onnx` for packed `uint8` `W_q` + dequant ops; STOP if folded.
6. Submodel equivalence: each ONNX submodel in ORT vs the swapped PyTorch model (encoder, first
   decoder step, cached decoder step with `cache_position`), `atol=1e-5, rtol=1e-5`.
7. `quant="onnx"` branch via `ORTModelForSpeechSeq2Seq` + ASR pipeline.
8. `make hqq-reference` -> 100-sample manifest.
9. `make eval-onnx` -> exact text match vs manifest on 100 samples; WER == 0.1367.
10. Footprint measurement (informational): `.onnx` size, optimized size, RSS, with
    `ORT_ENABLE_ALL` vs `ORT_DISABLE_ALL`.

## Out of scope (for tiny-first)

- base and small ONNX exports.
- Uploading ONNX files to the HF repos (after tiny validates; the HQQ upload workflow —
  commit + tag + HF sha + push — applies then).
- Compact-runtime-RAM optimization.
- Any change to the canonical HQQ path (`load_whisper_hqq`, `quantize_whisper`, the three
  published models/cards, the `HQQ_OUT` directories).
---

# Zero-fp int8-compute ONNX (Phase B, in progress)

The published ONNX above is **fp16-storage / fp32-compute** — host-validation only. The
deployment HW has **no fp16 and no fp32 unit**, so compute must be **int8** (int8 weights +
int8 activations, int32/int64 accumulators, zero fp16/fp32 in the graph). This section
tracks the zero-fp int8-compute export. Work is on the `onnx-8b-compute` branch; the
storage target is **3-bit** (`build/whisper-tiny-hqq-3bit`): 2-bit is broken (WER 0.77),
3-bit == 4-bit WER (0.16) at ~half the storage.

## Reference: torch int8-compute (Phase A + A7) — DONE, gated

A torch reference (`int8_compute.py`) mirrors the int8-compute path so WER is validated
in torch before any ONNX. Each block was staged one family at a time (matmul -> +ln ->
+gelu -> +sm -> +conv), paired vs the same-model fp HQQ baseline, gate +0.02 abs.

- A1 per-group int8 matmul: dequant HQQ W_q -> int8 weight + per-group fixed-point scale
  (int mul + right-shift); int8 act x int8 weight -> int32 -> per-group fixed-point scale
  -> int32 bias -> requant. Fractional-bit retention + round-half-up `_rshift_round`;
  requant in int64 (int32 x 2^31 overflows).
- A2 int8 layernorm: fixed-point mean/var (Q16) + int rsqrt (Newton, Q20) + fixed-point
  gamma/beta (Q15). x_scale cancels in normalization (eps' = eps/x_scale^2).
- A3 int8 GELU: `x*Phi(x)` via an exact normal-CDF LUT (4096 entries Q16 over [-6,6]); Phi
  saturates outside so `x*LUT` keeps the tail exact. (The sigmoid(1.702x) approximation
  BROKE the staged gate; the exact-Phi LUT fixed it. The fix was the LUT content, not
  resolution.)
- A4 int8 softmax: subtract-max + exp LUT (Q15) + int reciprocal (Newton) + requant.
- A5 int8 conv: per-BATCH act quant (per-token not factorable across the conv window);
  per-channel int8 weight; int32 window matmul; conv2 stride 2. Integer attention scale
  1/sqrt(64) = 1/8 = exact >>3.
- A6 full int8-compute forward, staged + n=100 gate (fp act scale): fp 0.1542, int8 0.1495,
  delta -0.0047, 56/100 identical, bootstrap 95% CI [-0.0141, +0.004] -- upper bound
  +0.004 << +0.02. Phase A complete.
- **A7 integer-state reference + WER gate** (consensus-inserted before Phase B; codex+claude
  review). Phase A used RUNTIME fp per-token act scales; the zero-fp ONNX must use INTEGER
  act scales. A7 gates the fp->int act-scale conversion. Q1.16 fixed-point act scale (int64-
  overflow bound, not precision, sets Q; 2^-16 << int8's 2^-7 -> ~lossless). The boundary
  round uses the INTEGER scale (else A7 is optimistic vs the deployed graph's integer
  requant). The matmul dequant applies the int scale as int Mul + power-of-two shift,
  dequant without integer rounding loss (the round lives at the next op's input quant -- one
  round per boundary, matching ONNX). Firm n=100 gate (3-bit, int-act): fp 0.1542, int8
  0.1542, delta 0.0000, 55/100 identical, CI [-0.0091, +0.0106] -- upper bound +0.0106 <<
  +0.02. The int-act path is WER-equivalent to both the fp-act int8 path and the fp HQQ
  baseline. Remaining A7 items are NOT new WER gates: gamma/beta/bias are export-time
  fp->int bakes (low-risk, validated for weight scale in A1); the int KV cache reuses the
  same per-token act-quant pattern gated here (gate under Phase B Gate 2).

## Phase B emission — in progress (Q6 minimal-linear first)

Design (codex+claude consensus): per-group int8 matmul as standard ONNX int ops -- batched
`MatMulInteger` with G as the batch dim (mirror the reference `einsum("ogi,bgi->bog")`);
flatten-then-scale is impossible (the I contraction is inside MatMulInteger; the group axis
is gone before the per-group scale applies). Do NOT use MatMulInteger's scalar zero_point
(ORT 1.19.2 accepts only a SCALAR b_zp; HQQ zero is per-(o,g)) -- pass zp=0, emit the
`-x_zp*B` / `zero*(C-x_zp*D)` corrections explicitly. Requant in int64. 8-bit weights bake
to int8 (`w = qr - 128`) with a `+128*C` correction: ORT 1.19.2 implements batched
MatMulInteger only when BOTH inputs share a dtype (int8/uint8 is NOT_IMPLEMENTED), and
activations are int8, so the uint8 levels 0..255 cannot feed MatMulInteger directly. Nonlinear ops as int subgraphs (LUTs as
int initializers; avoid DynamicQuantizeLinear/QLinearMatMul/QLinearSoftmax/Round/standard
Conv/signed BitShift). int KV cache as loop-carried vars. Recursive zero-fp audit incl the
ORT-optimized artifact (ORT can inject fp Casts/fuse LN at load); validate under
ORT_DISABLE_ALL. New module `export_onnx_int8.py`.

**Build order (de-risk, do NOT build the merged-decoder Loop first):**
1. Q6 minimal ONNX of ONE real HQQ linear: packed W_q -> unpack -> batched MatMulInteger
   -> zp correction -> int per-group fixed-point -> int bias -> requant. Compare EVERY
   intermediate vs the torch int reference. Test batch 1&2, negatives, uint8 weights,
   extreme zps, all shift directions. onnx.checker + ORT CPU + raw zero-fp audit + optimized
   zero-fp audit. **Status: Stage 1a VERIFIED bit-exact (2026-08-05).** The 3-bit k_proj
   (`decoder.layers.0.self_attn.k_proj`, packing `3bit_32`, W_q int32 [461,32]) built by hand
   with `onnx.helper` (opset 18, ORT 1.19.2): 3-bit unpack via Div/Mod (W_q>=0); batched-over-G
   `MatMulInteger` (A=[G,B,gs] int8, B=[G,gs,O] int8 -> [G,B,O] -> [B,O,G]); the four terms
   A/Bt/C/D; the `-x_zp*B` / `zero*(C-x_zp*D)` zp corrections (zp=0 in MatMulInteger; HQQ zero
   is per-(o,g)); the int per-group fixed-point scale (precomputed int64 `half`/`p`
   initializers + Div/Mod signed-floor + round-half-up via `q - (neg & rem!=0)`); int act scale
   (Q1.16) as int Mul + power-of-two shift, dequant to fp (Stage 1a reference interface).
   **Every int intermediate is BIT-EXACT vs the torch `int8_matmul_fixed_point` int-act path**
   (qr_g, A, Bt, C, T1, T2, p1, p2, acc; 24..147456 elems). The fp output matches within
   float32 noise (rel ~1e-8). Robustness: B=1/2/3, all-positive (zp near -128), all-negative
   (zp near 127), extreme magnitudes -- `acc` bit-exact in every case. The signed rshift_round
   emulation is bit-exact incl the s=0 edge. The **8-bit path** (encoder+fc1 tier, packing
   `8bit_u8`) is also bit-exact: weights bake to int8 (`qr-128`) with a `+128*C` correction
   (ORT 1.19.2 batched MatMulInteger needs matching dtypes; int8/uint8 is NOT_IMPLEMENTED).
   `tests/test_int8_onnx.py` covers both tiers. **Next: Stage 1b -- the output requant (int8
   out + out scale, the true zero-fp boundary) and the raw + optimized zero-fp audit; then
   LN/GELU/softmax/conv/Loop.**
