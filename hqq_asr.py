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


def _patch_linears(model, quant_config, device):
    """Replace every nn.Linear (except proj_out) with an HQQLinear on device.

    Return the count of patched linears. proj_out is skipped because it is
    tied to the embedding and is kept in fp16 (see module docstring).
    """
    patched = 0
    for name, module in list(model.named_modules()):
        if type(module) is not nn.Linear:
            continue
        # proj_out is the tied lm_head (shares its weight with the decoder
        # embedding). Skip it: quantizing it would store the weight twice and
        # add error to the vocab projection. It is kept in fp16 instead.
        if name == "proj_out":
            continue
        parent = model
        for part in name.split(".")[:-1]:
            parent = getattr(parent, part)
        leaf = name.split(".")[-1]
        setattr(parent, leaf, HQQLinear(
            module, quant_config, compute_dtype=COMPUTE_DTYPE, device=device,
        ))
        patched += 1
    return patched


def quantize_whisper(model_id, save_dir, nbits=4, group_size=64, device="cpu"):
    """Quantize a Whisper model with HQQ and save it to save_dir.

    Load the model as fp32, replace its linears with HQQLinear (nbits,
    group_size), cast the non-quantized modules to fp16 for storage, and save
    via hqq's save_quantized. Also save the processor so the output dir is
    self-contained (AutoProcessor.from_pretrained(save_dir) works).
    """
    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=COMPUTE_DTYPE
    ).to(device)
    model.eval()

    patched = _patch_linears(model, quant_config, device)
    # hqq bookkeeping so save_quantized serializes the patched model.
    AutoHQQHFModel.setup_model(model)
    model.hqq_quantized = True
    model.base_class = AutoHQQHFModel

    # Compact storage: cast the non-quantized float modules to fp16. The tied
    # lm_head/embed_tokens Parameter object stays shared (nn.Module.to replaces
    # .data in place), so the weight is stored once.
    for module in model.modules():
        if not isinstance(module, HQQLinear):
            module.to(STORE_DTYPE)

    os.makedirs(save_dir, exist_ok=True)
    AutoHQQHFModel.save_quantized(model, save_dir)
    AutoProcessor.from_pretrained(model_id).save_pretrained(save_dir)
    return patched


def load_whisper_hqq(model_id_or_dir, device="cpu"):
    """Load a saved HQQ Whisper model on CPU and return it."""
    return WhisperHQQModel.from_quantized(
        model_id_or_dir, compute_dtype=COMPUTE_DTYPE, device=device,
    )


def build_pipeline(model_id, quant, device="cpu"):
    """Build the ASR pipeline for the requested mode.

    quant == "hqq": load MODEL_ASR as a saved HQQ model and wrap it in the
    pipeline with its processor. Otherwise load MODEL_ASR with the pipeline
    defaults (fp32, or an already-quantized model when it points at one).
    """
    if quant == "hqq":
        model = load_whisper_hqq(model_id, device=device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)
        return pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
    return pipeline(task="automatic-speech-recognition", model=model_id)