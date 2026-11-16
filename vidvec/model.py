"""The VidVec model: VideoLLaMA3 read out as an embedding model, its training loss, and
its loader.

Two callers, one loader:

* evaluation loads the released checkpoint (``iyttor/VidVec-7B``);
* training loads the base model ``DAMO-NLP-SG/VideoLLaMA3-7B`` and adds the
  ``<emb>`` token to it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import SequenceClassifierOutput

from .prompts import EMB_TOKEN

__all__ = [
    "BASE_MODEL", "ATTN_IMPLEMENTATION", "DSL_TEMPERATURE", "dsl_loss",
    "VideoLLaMA3RetrievalWrapper", "VidVecBundle", "load_vidvec", "ensure_emb_token",
]

BASE_MODEL = "DAMO-NLP-SG/VideoLLaMA3-7B"

ATTN_IMPLEMENTATION = "flash_attention_2"

#: Dual Softmax Loss temperature of the released recipe.
DSL_TEMPERATURE = 0.07


# ---------------------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------------------


def dsl_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float = DSL_TEMPERATURE,
             eps: float = 1e-8) -> torch.Tensor:
    """Symmetric Dual Softmax Loss over the in-batch similarity matrix.

    Row ``i`` of ``emb_a`` and row ``i`` of ``emb_b`` are a positive pair; every other
    row is a negative. Each logit is reweighted by the joint probability of the
    row-wise and column-wise softmaxes (in log space) before the cross-entropy.
    """
    a = F.normalize(emb_a, dim=-1)
    b = F.normalize(emb_b, dim=-1)
    logits = (a @ b.t()) / temperature  # (N, N)

    # Dual softmax prior weights; the softmaxes run in fp32 for stability.
    logits_fp32 = logits.float()
    p_row = F.softmax(logits_fp32, dim=1).to(logits.dtype)
    p_col = F.softmax(logits_fp32, dim=0).to(logits.dtype)
    logits_tilde = logits + torch.log(p_row * p_col + eps)

    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits_tilde, targets) + F.cross_entropy(logits_tilde.t(), targets))


# ---------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------


class VideoLLaMA3RetrievalWrapper(nn.Module):
    """VideoLLaMA3 causal LM read out as an embedding model.

    The embedding is the last-layer hidden state at the position before the
    ``<emb>`` token, L2-normalised. With ``inference=True`` the forward returns the
    embeddings; otherwise the batch holds anchors then positives, and it returns the
    Dual Softmax Loss with negatives gathered across data-parallel ranks.
    """

    def __init__(self, base_model):
        super().__init__()
        # Register as submodule for proper .to(), .parameters(), etc.
        self.base_model = base_model

        # Hook storage for capturing LLM input after visual encoder processing
        self._llm_input_ids = None
        self._hook_handle = None
        self._setup_llm_input_hook()

    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.base_model.resize_token_embeddings(*args, **kwargs)

    def tie_weights(self):
        if hasattr(self.base_model, "tie_weights"):
            self.base_model.tie_weights()

    def __getattr__(self, name):
        # Fallback: delegate to base_model after ours
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def _setup_llm_input_hook(self):
        """Capture the input_ids the LLM actually sees, after visual token compression."""

        def embed_tokens_hook(module, args):
            if len(args) > 0 and hasattr(args[0], 'shape'):
                self._llm_input_ids = args[0]

        self._hook_handle = self.base_model.model.embed_tokens.register_forward_pre_hook(embed_tokens_hook)

    def __del__(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
        pixel_values: torch.Tensor = None,
        inference: bool = False,
        ids=None,
        **kwargs
    ):
        # Sanitize kwargs to avoid duplicate arguments
        sanitized = dict(kwargs)
        for dup_key in ["labels", "output_hidden_states", "use_cache"]:
            sanitized.pop(dup_key, None)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            **sanitized,
        )

        hidden_states = outputs.hidden_states[-1]  # (B, L, D)
        batch_total = hidden_states.size(0)

        emb_token_ids = getattr(self.base_model.config, "emb_token_ids", None)
        assert emb_token_ids, "emb_token_ids missing in config."
        emb_id = emb_token_ids[0]

        if pixel_values is not None and self._llm_input_ids is not None:
            # Video: token compression drops visual tokens before the LLM runs, so
            # <emb> is located in the ids the LLM received. One video per batch.
            search_input_ids = self._llm_input_ids
            if search_input_ids.dim() == 1:
                search_input_ids = search_input_ids.unsqueeze(0)
            assert batch_total == 1, f"VideoLLaMA3 expects batch_size=1 for video inputs, got {batch_total}"
        else:
            search_input_ids = input_ids

        # The embedding is read at the position before <emb>.
        embed_indices = torch.argmax((search_input_ids == emb_id).int(), dim=1) - 1
        embed_features = hidden_states[torch.arange(batch_total, device=hidden_states.device), embed_indices]

        if inference:
            embed_features = F.normalize(embed_features, dim=-1)
            if ids is not None:
                return embed_features, ids
            return embed_features

        batch_size = batch_total // 2
        emb_a = embed_features[:batch_size]
        emb_p = embed_features[batch_size:]

        # Gather across ranks so every pair in the global batch is a negative.
        if dist.is_initialized():
            emb_a_list = [torch.zeros_like(emb_a) for _ in range(dist.get_world_size())]
            emb_p_list = [torch.zeros_like(emb_p) for _ in range(dist.get_world_size())]
            dist.all_gather(emb_a_list, emb_a.contiguous())
            dist.all_gather(emb_p_list, emb_p.contiguous())
            emb_a_list[dist.get_rank()] = emb_a
            emb_p_list[dist.get_rank()] = emb_p
            emb_a = torch.cat(emb_a_list, 0)
            emb_p = torch.cat(emb_p_list, 0)

        emb_a = F.normalize(emb_a, dim=-1)
        emb_p = F.normalize(emb_p, dim=-1)
        return SequenceClassifierOutput(loss=dsl_loss(emb_a, emb_p))


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------


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
