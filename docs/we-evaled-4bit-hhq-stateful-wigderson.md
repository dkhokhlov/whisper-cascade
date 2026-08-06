# Plan: Zero-fp int8-compute Whisper-Tiny ONNX (HQQ 2/3-bit storage → int8 compute)

## Context

The deployment HW has **no fp16 and no fp32 unit**. The fp16 ONNX we just published
(v1.9.5) is therefore **host-validation-only, not deployable** on this HW — relabel, do
not delete. The real deploy target is an **int8-compute** graph (int8 weights + int8
activations, int32 accumulators, zero fp16/fp32).

User decision: build the **full zero-fp int8 graph now** (not the pragmatic RTL-split).

Feasibility findings (read-only, from ORT 1.19.2 quantizer source + ONNX op analysis):

- Stock ORT int8 quantization does NOT yield a zero-fp graph: LayerNormalization is only
  `QDQNormalization` (QDQ-wrapped → the op runs in **fp**), **GELU is not quantized at all**
  (left fp), and `QLinearSoftmax` is a `com.microsoft` custom-domain op (not standard
  ONNX). Whisper has 2 layernorms + 1 GELU per layer → stock tooling leaves these in fp.
- HQQ 2-bit/3-bit levels (4/8) fit **losslessly** into int8 (256 levels). The mismatch is
  scale granularity: HQQ is per-**group** (group_size=32) vs `QLinearMatMul` per-
  **channel**. Resolution: express the per-group int8 matmul with **standard ONNX int ops**
  (`MatMulInteger` over per-group reshapes + integer fixed-point scale), preserving HQQ
  exactly AND keeping the graph ORT-runnable (Gate 2 works without a custom kernel).
- The non-linear ops (GELU, layernorm, softmax) need **int8 fixed-point decompositions**:
  GELU = x·Φ(x) (exact normal-CDF LUT, Q16 over [-6,6]; the earlier sigmoid(1.702x)
  approximation was the dominant WER cost and BROKE the staged gate — replaced by the
  exact-Φ LUT, see [[int8-compute-nonlinearity-a2-a3]]); layernorm via fixed-point
  mean/var + integer sqrt-approx (Newton or LUT) + integer reciprocal; softmax via
  integer subtract-max + integer exp LUT + integer reciprocal. These add approximation
  error → WER risk, so the plan validates each **incrementally in torch** before any
  ONNX, isolating which block costs WER.
- HQQ has no int8-compute feature; this is a new export-time transformation we write.

Prerequisite (user instruction): **merge `onnx-fp16` to `main` first.**

Effort: multi-week, research-grade. The plan is staged with WER gates so we learn early
if a fixed-point block breaks WER and can stop or iterate before the ONNX work.

---

## Step 0 — Merge `onnx-fp16` → `main`; relabel fp16 release as host-validation-only

- `git checkout main && git merge --verify-signatures onnx-fp16` (ff, or `--no-ff -S` if
  a merge commit is needed; verify the signature; never disable gpgsign; agent does not
  push). User: `! git push origin main && git push origin v1.9.3 v1.9.4 v1.9.5`.
- **Relabel** the fp16 ONNX in the three cards: "host-side validation; the deployment HW
  has no fp16 unit — the int8-compute export is the deploy target." Upload relabeled
  cards to HF (outward-facing; confirm with user). Do not withdraw the fp16 files.

All remaining work on a new branch **`onnx-8b-compute`** off `main`.

## Step 1 — Branch + HQQ_NBITS exposure + quantize tiny at 2-bit and 3-bit (storage)

- `Makefile`: add `HQQ_NBITS ?= 4` (~line 55); forward in `quantize`/`push`
  (lines 170-176). (`quantize.py:29` already reads it; `int()` can't express 1.58 —
  document.)
- `git checkout main && git checkout -b onnx-8b-compute`.
- Quantize whisper-tiny at **2-bit** and **3-bit** (storage candidates; 3-bit is the
  viability probe, 2-bit the target): `make quantize MODEL_ASR=openai/whisper-tiny
  HQQ_NBITS=2 HQQ_OUT=build/whisper-tiny-hqq-2bit` (and `=3`). Verify per-layer
  `meta["nbits"]`/`meta["packing"]` via a heredoc load.

## Phase A — Torch int8-compute reference + STAGED WER validation (the de-risk)

Build a torch reference of the int8-compute path and validate WER **incrementally**, so
the first WER break isolates the offending fixed-point block. This runs in torch (fast)
before any ONNX emission. New module `int8_compute.py` (the reference + the spec the
ONNX emission mirrors).

- **A1 — Per-group int8 matmul block.** Dequant HQQ W_q (2/3-bit) → int8 weight +
  per-group fixed-point scale (int multiplier + right-shift). int8 act (per-token
  dynamic) × int8 weight → int32 (`MatMulInteger` math) → per-group fixed-point scale
  → int32 bias → requant to int8. Validate per-layer output vs the HQQ fp
  dequant+matmul (tolerance: integer-exact after the fixed-point scale, modulo the
  per-group scale's fixed-point rounding).
- **A2 — Int8 layernorm.** Fixed-point mean + variance + integer sqrt-approx (Newton
  iteration or small LUT) + integer reciprocal + per-channel γ/β (fixed-point) + requant.
  Validate vs fp layernorm.
- **A3 — Int8 GELU.** `x·Φ(x)` where Φ is the exact standard-normal CDF, via an integer
  Φ LUT (4096 entries, Q16 over [-6,6]) + integer Mul + requant. Φ saturates outside
  the grid so the `x·LUT` multiply keeps the tail exact. (The earlier `sigmoid(1.702x)`
  approximation was the dominant WER cost and was replaced by the exact-Φ LUT — see
  [[int8-compute-nonlinearity-a2-a3]].) Validate vs fp GELU.
- **A4 — Int8 softmax (attention).** Integer subtract-max + integer exp LUT + integer
  sum + integer reciprocal + Mul + requant. Validate vs fp softmax.
- **A5 — Quantized activations + int8 conv + integer attention scale.** Per-token
  dynamic int8 activation quantization between ops; `ConvInteger`/`QLinearConv` math for
  the feature conv; the eager-attention scale as an integer Mul (matches the fp16
  export's eager choice, so no Sqrt/Div).
- **A6 — Full int8-compute Whisper forward in torch; staged WER.** Run the same
  `fleurs en_us` n=100 + the 4-bit reference manifest; paired per-sample WER delta +
  bootstrap 95% CI (per the 2-bit plan's gate). **Enable one block at a time**
  (A1-only → +A2 → +A3 → +A4 → +A5), measuring the WER delta at each step → isolates
  which fixed-point block costs WER.
- **Decision gate:** if the staged WER stays within tolerance (first pass +0.02 absolute
  over the 4-bit reference, with the CI upper bound), proceed to Phase B. If a block
  breaks it, iterate that block's fixed-point approximation (more LUT entries, wider
  int32/int64 intermediates, per-token vs per-tensor activation scale) before proceeding.
  If no iteration keeps WER → stop, report (QAT/weight-recovery out of scope).
- **A7 — Integer-state reference + WER gate (consensus-inserted, prerequisite for Phase
  B).** The Phase A reference (A1-A6) uses RUNTIME fp activation scales (per-token
  `x_scale` fp). The zero-fp ONNX graph must use INTEGER (fixed-point) activation scales
  at runtime. The fp→int act-scale conversion is a NEW, unmeasured approximation; the
  A6 gate did not measure it. Rewrite the reference so every runtime scale is int
  fixed-point (`scale = mantissa_int * 2^exp`): integer per-token act scale (Q1.16 — Q1.30
  overflows because the act scale multiplies a 2^36-2^39 accumulator; apply as int Mul
  + power-of-two shift; dequant WITHOUT integer rounding loss to isolate the
  act-scale-VALUE approximation from the next op's output quantization), int γ/β bake,
  int bias bake (export-time fp→int via frexp is fine; only emitted initializers +
  runtime intermediates must be int), int KV cache (int8 K,V + int scale as loop-carried
  vars). Re-run the staged + n=100 WER gate on THIS integer-state reference. Only if A7
  stays within the +0.02 gate does Phase B emission measure the right WER. See
  [[phase-b-design-consensus]].

## Phase B — Emit the zero-fp ONNX graph (from the validated torch reference)

`export_onnx.py`: a new int8-compute export path alongside the fp16/fp32 path.

- **Per-group int8 matmul** as standard ONNX int ops: 2/3-bit unpack
  (BitwiseAnd/BitShift/Cast to int8) + `MatMulInteger` over per-group reshapes +
  integer fixed-point scale (int Mul + BitShift) + int32 bias Add + requant (Cast +
  Mul + Add for the per-tensor/per-token output scale). Standard ops → ORT-runnable →
  Gate 2 works without a custom kernel. (Custom-op fallback only if the node-count
  bloat is prohibitive on the import host.)
- **Int8 layernorm / GELU / softmax** as integer ONNX-op subgraphs mirroring the A2-A4
  math (LUTs as int initializers).
- **int8 conv** (`ConvInteger`/`QLinearConv`), **integer attention scale** (integer Mul).
- **int8 activations throughout; int32 accumulators; zero fp16/fp32.**
- `do_constant_folding=False` (keep W_q packed — same reason as 4-bit/fp16).
- Audit zero-fp recursively (encoder + merged-decoder nested `If`/`Loop` subgraphs) —
  extend the existing fp32-free audit to also assert no fp16.

## Phase C — Gates + footprint + publish (outward-facing, confirm)

- **Gate 1 (structural):** packed W_q + the int8 matmul (`MatMulInteger` + per-group
  scale) + int8 decomps; **zero fp16/fp32** (recursive, post-trim shipped files).
- **Gate 2 (functional):** ONNX WER (ORT runs it — standard ops) == the Phase A torch
  int8 WER == HQQ manifest WER (within the fixed-point tolerance). Report the
  torch-vs-ORT delta (validates the ONNX emission matches the torch reference).
- **Footprint gate:** int8 graph file size + load RSS + startup + RTF vs the fp16/fp32
  releases. The int8 decomp may be **larger as a file** (more nodes) but lower compute
  RAM — measure both and gate on the deployed-RAM win, not file size.
- **Publish:** new HF repo (e.g. `dkhokhlov/whisper-tiny-hqq-2bit-int8` or
  `...-3bit-int8`; confirm name + which storage bit-width with user). New card
  `hqq_report_<bits>bit_int8.md`. Upload `qmodel.pt` + the int8 ONNX files + config +
  README. HQQ upload workflow: commit + tag `v1.10.0` + HF sha + push; **regenerate
  `uv.lock`**; multilingual smoke test (≥1 non-`en_us` fleurs config). Branch
  `onnx-8b-compute` → merge to `main` (signed) + user pushes.

---

## Files

- `int8_compute.py` (new) — torch int8-compute reference blocks (A1-A5) + the full
  forward; the spec the ONNX emission mirrors; the staged-WER harness.
- `export_onnx.py` — new int8-compute export path: `HQQLinearInt8ONNX` (per-group int8
  matmul as standard int ops) + int8 layernorm/GELU/softmax decompositions + int8 conv;
  recursive zero-fp audit; reuse `do_constant_folding=False`.
- `hqq_asr.py` — no change (HQQ load path unchanged; int8 compute is export-time).
- `Makefile` — `HQQ_NBITS` exposure; int8-compute targets (`int8-onnx`, `eval-int8`).
- `tests/test_int8_compute.py` (new) — per-block equivalence (int8 block vs fp
  reference) + the staged-WER harness.
- `hqq_report_<bits>bit_int8.md` (new) — the int8 model card.
- `hqq_report.md`/`_base`/`_small` — relabel fp16 ONNX as host-validation-only (Step 0).
- `pyproject.toml` `1.9.5` → `1.10.0`; regenerate `uv.lock`.

## Verification

```bash
# Step 0: merge + relabel (agent does not push)
git checkout main && git merge --verify-signatures onnx-fp16
# user: ! git push origin main && git push origin v1.9.3 v1.9.4 v1.9.5
# relabel fp16 cards -> upload to HF (confirm first)

# Step 1: branch + quantize 2-bit + 3-bit storage
git checkout main && git checkout -b onnx-8b-compute
make quantize MODEL_ASR=openai/whisper-tiny HQQ_NBITS=2 HQQ_OUT=build/whisper-tiny-hqq-2bit
make quantize MODEL_ASR=openai/whisper-tiny HQQ_NBITS=3 HQQ_OUT=build/whisper-tiny-hqq-3bit

# Phase A: torch int8 reference, STAGED WER (enable one block at a time)
.venv/bin/python -m pytest tests/test_int8_compute.py -q   # per-block vs fp equivalence
.venv/bin/python int8_compute.py --staged-wer --ref build/hqq_reference_tiny_4bit.json
# expect: WER delta per stage; gate +0.02 abs over the 4-bit ref (bootstrap CI)

# Phase B: emit zero-fp int8 ONNX
make int8-onnx HQQ_REPO=build/whisper-tiny-hqq-2bit ONNX_OUT=build/whisper-tiny-hqq-2bit-int8

# Phase C: gates
# gate1: recursive zero-fp audit + packed W_q + MatMulInteger per-group scale (post-trim)
# gate2: ORT runs the standard-op graph; WER == Phase A torch WER == manifest (within tolerance)
# footprint: int8 file size + RSS + RTF vs fp16/fp32 releases; gate on deployed-RAM win
# publish: new repo (confirm name), card, v1.10.0, uv.lock regen, multilingual smoke test
```

## Out of scope

- Per-channel int8 requant (we preserve HQQ per-group via the MatMulInteger decomp).
- A custom ORT kernel (standard-op decomp keeps the graph ORT-runnable; custom op only
  if node-bloat forces it).
- base / small int8 — only after tiny int8 WER is acceptable.
- QAT / calibration-data weight recovery if pure-PTQ + fixed-point decomp fails the WER
  gate.
- True signed ternary {-1,0,1}; `nbits=1.58` (3-level, unsigned, dominated by 2-bit).
- Any change to the canonical 4-bit models or `load_whisper_hqq`/`quantize_whisper`.