"""HQQ 4-bit grouped quantization for Whisper ASR on CPU.

transformers' HqqConfig path needs a GPU for both quantization and loading
(it writes a quantization_config to config.json that triggers a GPU check on
every from_pretrained). This CPU-only project cannot use that path. Instead
this module uses the hqq library directly:

  - Quantize: replace every nn.Linear (except the tied proj_out head) with an
    HQQLinear (nbits=4, group_size=64) on CPU, then cast the remaining
    float modules to fp16 for compact storage, and save via hqq's
    save_quantized (a qmodel.pt of serialized weights + config.json).
  - Load: a thin subclass of AutoHQQHFModel overrides create_model to build a
    WhisperForConditionalGeneration (the base class picks AutoModel, which
    returns WhisperModel without generate), then from_quantized loads the
    HQQ weights on CPU. Stored fp16 weights are upcast to fp32 for compute.

proj_out (the lm_head) is kept tied to the embedding in fp16 (not quantized):
it shares its weight with the decoder embed_tokens, so quantizing it would
store the weight twice and add error to the vocab projection. Keeping it fp16
stores it once with no quantization error.

build_pipeline() returns a transformers ASR pipeline for either mode:
QUANT unset loads MODEL_ASR as fp32 (or an already-quantized model when
MODEL_ASR points at one); QUANT=hqq loads MODEL_ASR as a saved HQQ model.
"""

import os

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoConfig,
    pipeline,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
transformers.logging.set_verbosity_error()

from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
from hqq.models.hf.base import AutoHQQHFModel, init_empty_weights

# CPU compute dtype. Stored fp16 weights are upcast to this at load time so
# CPU matmuls run in fp32 (fast and dtype-consistent with the HQQ dequant).
COMPUTE_DTYPE = torch.float32
# Storage dtype for the non-quantized modules (embedding, convs, norms).
STORE_DTYPE = torch.float16


class WhisperHQQModel(AutoHQQHFModel):
    """AutoHQQHFModel that builds a SpeechSeq2Seq model for Whisper.

    The base create_model uses transformers.AutoModel for non-CausalLM
    architectures, which returns WhisperModel (no generate). Override it to
    use AutoModelForSpeechSeq2Seq so the loaded model can transcribe.
    """

    @classmethod
    def create_model(cls, save_dir, kwargs):
        config = AutoConfig.from_pretrained(os.path.join(save_dir, "config.json"))
        with init_empty_weights():
            return AutoModelForSpeechSeq2Seq.from_config(config)

    @classmethod
    def load_weights(cls, save_dir, map_location=None):
        """Load HQQ weights from qmodel.pt, or model.safetensors when HQQ_FORMAT=safetensors.

        The default qmodel.pt is a torch pickle holding the nested
        {module_name: {field: tensor}} dict from hqq serialize_weights. The
        optional model.safetensors is a flat dotted-key tensor map (one entry
        per leaf-module field); regroup it back into that nested dict so
        from_quantized._load_module, which indexes weights[module.name],
        works unchanged.
        """
        if os.environ.get("HQQ_FORMAT", "").strip().lower() == "safetensors":
            st = os.path.join(save_dir, "model.safetensors")
            if os.path.exists(st):
                from safetensors.torch import load_file

                flat = load_file(st)
                weights = {}
                for key, tensor in flat.items():
                    module_name, _, field = key.rpartition(".")
                    weights.setdefault(module_name, {})[field] = tensor
                return weights
        return super().load_weights(save_dir, map_location)


def _patch_linears(model, default_cfg, tier8_cfg, device, tier8_patterns):
    """Replace every nn.Linear (except proj_out) with an HQQLinear on device.

    A linear whose name contains any substring in tier8_patterns uses
    tier8_cfg (the 8-bit tier); the rest use default_cfg (the 4-bit tier).
    proj_out is exempt (tied to the embedding, kept fp16). Embeddings, convs,
    and norms are not nn.Linear, so they are not patched here; they are cast
    to fp16 separately. Return (n_default, n_tier8).
    """
    n_default = 0
    n_tier8 = 0
    for name, module in list(model.named_modules()):
        if type(module) is not nn.Linear:
            continue
        # proj_out is the tied lm_head (shares its weight with the decoder
        # embedding). Exempt it: quantizing it would store the weight twice
        # and add error to the vocab projection. It is kept in fp16 instead.
        if name == "proj_out":
            continue
        parent = model
        for part in name.split(".")[:-1]:
            parent = getattr(parent, part)
        leaf = name.split(".")[-1]
        if tier8_cfg is not None and any(p and p in name for p in tier8_patterns):
            setattr(parent, leaf, HQQLinear(
                module, tier8_cfg, compute_dtype=COMPUTE_DTYPE, device=device,
            ))
            n_tier8 += 1
        else:
            setattr(parent, leaf, HQQLinear(
                module, default_cfg, compute_dtype=COMPUTE_DTYPE, device=device,
            ))
            n_default += 1
    return n_default, n_tier8


def quantize_whisper(
    model_id,
    save_dir,
    nbits=4,
    group_size=32,
    axis=1,
    tier8_nbits=None,
    tier8_patterns=(),
    multilingual=True,
    device="cpu",
):
    """Quantize a Whisper model with HQQ and save it to save_dir.

    Load the model as fp32, replace its linears with HQQLinear, cast the
    non-quantized modules to fp16 for storage, and save via hqq's
    save_quantized. Also save the processor so the output dir is
    self-contained (AutoProcessor.from_pretrained(save_dir) works).

    axis=1 groups along the input/reduction dim and measures better than axis=0
    on whisper-tiny (0.1622 vs 0.2032 WER on fleurs en_us); axis=0 is meant for
    GPU-optimized inference. tier8_nbits/tier8_patterns assign sensitive linears
    (e.g. fc1, the encoder stack) to the 8-bit tier.

    multilingual: whisper-tiny's config.json ships with forced_decoder_ids that
    hardcode the output language to English (the English track). When True,
    clear those before saving and write a modern generation_config.json (with
    lang_to_id/task_to_id/is_multilingual) so the model auto-detects the
    language and transcribes, and accepts language=/task= overrides. This is
    the multilingual flavor; it is published under a separate name so the
    English track is preserved.
    """
    default_cfg = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=axis)
    tier8_cfg = None
    if tier8_nbits is not None and tier8_patterns:
        tier8_cfg = BaseQuantizeConfig(
            nbits=tier8_nbits, group_size=group_size, axis=axis
        )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=COMPUTE_DTYPE
    ).to(device)
    model.eval()

    n_default, n_tier8 = _patch_linears(
        model, default_cfg, tier8_cfg, device, tier8_patterns
    )
    # hqq bookkeeping so save_quantized serializes the patched model.
    AutoHQQHFModel.setup_model(model)
    model.hqq_quantized = True
    model.base_class = AutoHQQHFModel

    # Multilingual flavor: drop the English forced_decoder_ids baked into the
    # source config so the saved config.json does not hardcode the output
    # language. The modern generation_config.json written below drives
    # auto-detect + transcribe at load time.
    if multilingual:
        model.config.forced_decoder_ids = None

    # Compact storage: cast the non-quantized float modules to fp16. The tied
    # proj_out/embed_tokens Parameter object stays shared (nn.Module.to replaces
    # .data in place), so the weight is stored once. HQQLinear is skipped: it
    # overrides .float()/.to(dtype) as a no-op and must keep its packed uint8
    # weight and fp32 scale/zero.
    for module in model.modules():
        if not isinstance(module, HQQLinear):
            module.to(STORE_DTYPE)

    os.makedirs(save_dir, exist_ok=True)
    AutoHQQHFModel.save_quantized(model, save_dir)
    AutoProcessor.from_pretrained(model_id).save_pretrained(save_dir)
    if multilingual:
        from transformers import GenerationConfig
        gc = GenerationConfig.from_pretrained(model_id)
        gc.forced_decoder_ids = None
        gc.language = None
        gc.task = "transcribe"
        gc.save_pretrained(save_dir)
    return {"default": n_default, "tier8": n_tier8}


def load_whisper_hqq(model_id_or_dir, device="cpu"):
    """Load a saved HQQ Whisper model on CPU and return it.

    If the model ships a modern generation_config.json (the multilingual
    flavor), adopt it: it carries lang_to_id/task_to_id/is_multilingual so the
    model auto-detects the language and transcribes, and language=/task=
    overrides are accepted. Without it (the English track) the model keeps the
    forced_decoder_ids built from config.json (English) so that track is
    preserved unchanged.
    """
    model = WhisperHQQModel.from_quantized(
        model_id_or_dir, compute_dtype=COMPUTE_DTYPE, device=device, cache_dir=None,
    )
    # Re-tie proj_out to the decoder embedding. hqq's from_quantized._load_module
    # upcasts each stored tensor to fp32 with .to(), which copies storage and
    # breaks the proj_out/embed_tokens tie that exists on disk (torch.save dedups
    # the shared weight to one copy). Without this re-tie the embedding lives
    # twice in resident RAM, and for whisper-tiny HQQ RAM (181.8 MB) would exceed
    # the fp32 baseline (151.06 MB). The tie is value-identical (same weights),
    # so this is WER-neutral (verified 0.1367 == 0.1367 on fleurs en_us).
    model.proj_out.weight = model.model.decoder.embed_tokens.weight
    try:
        from transformers import GenerationConfig
        gc = GenerationConfig.from_pretrained(model_id_or_dir)
        # Only adopt the modern multilingual form; the English track's
        # config.json-based config has no lang_to_id, so it is left as-is.
        if getattr(gc, "lang_to_id", None):
            model.generation_config = gc
    except Exception:
        pass
    return model


def build_pipeline(model_id, quant, device="cpu"):
    """Build the ASR pipeline for the requested mode.

    quant == "hqq": load MODEL_ASR as a saved HQQ model and wrap it in the
    pipeline with its processor. Otherwise load MODEL_ASR with the pipeline
    defaults (fp32, or an already-quantized model when it points at one).

    device == "cpu" (default) keeps the original CPU behavior. Any other value
    (e.g. "cuda") moves the model to that device and runs the pipeline on GPU:
    the fp32 model is loaded and moved with .to(device), the HQQ model is
    loaded on device, and device=0 (cuda:0) is passed to the pipeline so the
    inputs are moved to the model. HQQ weights are dequantized in COMPUTE_DTYPE
    (fp32) on the chosen device.
    """
    if quant == "onnx":
        # Load the exported ONNX subgraphs (encoder/decoder/decoder_with_past
        # in model_id == ONNX_OUT) via ORTModelForSpeechSeq2Seq on CPU ONNX Runtime,
        # wrapped in the standard ASR pipeline. ORTModelForSpeechSeq2Seq runs the
        # generation loop outside ONNX. The processor (config/processor/tokenizer
        # files) was copied into ONNX_OUT by export_onnx.py. Import is lazy so the
        # CPU .venv (no onnxruntime) keeps working for make asr / make test.
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

        model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, provider="CPUExecutionProvider")
        processor = AutoProcessor.from_pretrained(model_id)
        return pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
    if quant == "hqq":
        model = load_whisper_hqq(model_id, device=device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)
        kwargs = dict(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
        if device != "cpu":
            kwargs["device"] = 0
        return pipeline(**kwargs)
    if device != "cpu":
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=COMPUTE_DTYPE
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)
        return pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0,
        )
    return pipeline(task="automatic-speech-recognition", model=model_id)