"""Loading a VidVec model, its processor and its tokenizer.

Two callers, one loader:

* evaluation loads the released checkpoint (``iyttor/VidVec-7B``);
* training loads the base model ``DAMO-NLP-SG/VideoLLaMA3-7B`` and adds the
  ``<emb>`` token to it.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from models.videollama3_ret import VideoLLaMA3RetrievalWrapper

from .prompts import EMB_TOKEN

__all__ = ["BASE_MODEL", "ATTN_IMPLEMENTATION", "VidVecBundle", "load_vidvec", "ensure_emb_token"]

BASE_MODEL = "DAMO-NLP-SG/VideoLLaMA3-7B"

#: FlashAttention-2 when installed (the setting the paper's numbers were produced
#: with), otherwise PyTorch's SDPA.
ATTN_IMPLEMENTATION = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"


@dataclass
class VidVecBundle:
    model: VideoLLaMA3RetrievalWrapper
    tokenizer: Any
    processor: Any
    emb_token_id: int
    model_path: str

    @property
    def label(self) -> str:
        return os.path.basename(str(self.model_path).rstrip("/")) or self.model_path


def _load_processor(path: str, **kwargs):
    """Load a processor and refuse to return the wrong thing.

    When a checkpoint's ``processor_config.json`` lacks ``auto_map``,
    ``AutoProcessor`` silently degrades to a bare tokenizer instead of raising.
    Downstream that shows up much later as a missing ``conversation=`` kwarg, so
    catch it here where the cause is obvious.
    """
    proc = AutoProcessor.from_pretrained(path, trust_remote_code=True, **kwargs)
    if not hasattr(proc, "tokenizer"):
        raise RuntimeError(
            f"AutoProcessor returned {type(proc).__name__} rather than a VideoLLaMA3 "
            f"processor for {path!r}: its processor_config.json has no auto_map entry."
        )
    return proc


def ensure_emb_token(processor, model, emb_token: str = EMB_TOKEN) -> int:
    """Make sure ``<emb>`` exists and ``config.emb_token_ids`` agrees with it.

    Deliberately does **not** resize unconditionally. ``<emb>`` is id 151668 and the
    embedding matrix has 152064 rows, so the row already exists; resizing to
    ``len(tokenizer)`` (151669) would only truncate unreachable padding rows.
    Resizing is kept solely as a genuine out-of-range guard.
    """
    tokenizer = processor.tokenizer

    if emb_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [emb_token]},
            replace_additional_special_tokens=False,
        )

    emb_id = tokenizer.convert_tokens_to_ids(emb_token)
    if emb_id is None or emb_id < 0:
        raise RuntimeError(f"Could not resolve {emb_token!r} to a token id")

    n_rows = model.get_input_embeddings().weight.shape[0]
    if emb_id >= n_rows:
        model.resize_token_embeddings(len(tokenizer))

    configured = list(getattr(model.config, "emb_token_ids", None) or [])
    if configured and configured[0] != emb_id:
        raise RuntimeError(
            f"Checkpoint config says emb_token_ids={configured} but the tokenizer maps "
            f"{emb_token!r} to {emb_id}. The weights and tokenizer do not belong together."
        )
    model.config.emb_token_ids = configured or [emb_id]
    return emb_id


def load_vidvec(
    model_path: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[Union[str, Dict]] = "cuda:0",
    attn_implementation: Optional[str] = ATTN_IMPLEMENTATION,
    cache_dir: Optional[str] = None,
    low_cpu_mem_usage: bool = True,
) -> VidVecBundle:
    """Load ``model_path`` (a Hub id) wrapped for embedding extraction, in eval mode."""
    load_kwargs: Dict[str, Any] = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    processor = _load_processor(model_path, **({"cache_dir": cache_dir} if cache_dir else {}))
    model = VideoLLaMA3RetrievalWrapper(AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs))
    emb_id = ensure_emb_token(processor, model)
    model.eval()

    return VidVecBundle(
        model=model,
        tokenizer=processor.tokenizer,
        processor=processor,
        emb_token_id=emb_id,
        model_path=model_path,
    )
