import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

#: Dual Softmax Loss temperature of the released recipe.
DSL_TEMPERATURE = 0.07


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
