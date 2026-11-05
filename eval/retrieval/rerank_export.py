"""Write the text-to-video candidate lists that the reranking stage reads."""

from __future__ import annotations

import json
import os
from typing import Sequence, Tuple

import numpy as np
import torch

from .scoring import T2V_SCALE, apply_score_scaling

__all__ = ["candidate_files", "export_t2v_candidates"]


def candidate_files(out_dir: str, dataset: str) -> Tuple[str, str]:
    """``(queries, candidates)``: caption ids, and each caption's videos ranked best first."""
    stem = os.path.join(out_dir, f"{dataset}_t2v")
    return f"{stem}_query_names.json", f"{stem}_cand_names.json"


def export_t2v_candidates(
    *,
    dataset: str,
    query_names: Sequence[str],
    candidate_names: Sequence[str],
    scores: torch.Tensor,  # (texts, videos), unscaled
    out_dir: str,
) -> None:
    """Rank every video for every caption, using the same scaled scores the metrics use."""
    os.makedirs(out_dir, exist_ok=True)
    query_path, cand_path = candidate_files(out_dir, dataset)
    order = torch.argsort(apply_score_scaling(scores, T2V_SCALE), dim=1, descending=True)
    ranked = np.asarray(list(candidate_names))[order.cpu().numpy()]
    with open(query_path, "w", encoding="utf-8") as fh:
        json.dump(list(query_names), fh, indent=2)
    with open(cand_path, "w", encoding="utf-8") as fh:
        json.dump(ranked.tolist(), fh, indent=2)
