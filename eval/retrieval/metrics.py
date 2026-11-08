"""Retrieval metrics: R@K, median rank, mean rank.

``recall_at_k`` and ``batchify`` are ported verbatim from the legacy eval scripts
so the reported numbers stay comparable. ``rank_metrics`` is a vectorised rewrite
of the per-row Python loop that previously existed only in the DiDeMo / VATEX /
ActivityNet scripts and only for t2v; it is exercised against the original loop
in ``code/eval/tests/test_metrics.py``.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

__all__ = [
    "recall_at_k",
    "batchify",
    "clip_style_recall",
    "rank_metrics",
    "evaluate_direction",
]


def recall_at_k(scores: torch.Tensor, positive_pairs: torch.Tensor, k: int) -> torch.Tensor:
    """Fraction of a query's positives that land in its top-k. Verbatim port."""
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    return nb_true_positive / nb_positive


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    """Apply a row-wise matrix metric in chunks. Verbatim port."""
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        results.append(func(x, y, *args, **kwargs).cpu())
    return torch.cat(results)


def clip_style_recall(
    scores: torch.Tensor,
    positive_pairs: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, float]:
    """R@K the way retrieval papers report it: 1 if *any* positive is in the top-k.

    Recovered from the true recall by thresholding at >0, which is exactly what
    the legacy scripts did.
    """
    out: Dict[str, float] = {}
    for k in ks:
        k_eff = min(k, scores.shape[1])
        hits = batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=k_eff)
        out[f"R@{k}"] = (hits > 0).float().mean().item()
    return out


def rank_metrics(
    scores: torch.Tensor,
    positive_pairs: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Median and mean rank of each query's best-ranked positive (1-indexed).

    Ties are counted optimistically -- ``rank = #{strictly better} + 1`` -- which
    matches the legacy implementation.
    """
    ranks = []
    for start in range(0, scores.shape[0], batch_size):
        chunk = scores[start : start + batch_size].to(device)
        pos = positive_pairs[start : start + batch_size].to(device)

        # Best positive score per row; rows with no positive are skipped.
        neg_inf = torch.finfo(chunk.dtype).min
        best_pos = torch.where(pos, chunk, torch.full_like(chunk, neg_inf)).max(dim=1).values
        has_pos = pos.any(dim=1)

        better = (chunk > best_pos.unsqueeze(1)).sum(dim=1) + 1
        ranks.append(better[has_pos].float().cpu())

    if not ranks:
        return {"median_rank": float("nan"), "mean_rank": float("nan")}

    all_ranks = torch.cat(ranks).numpy()
    if all_ranks.size == 0:
        return {"median_rank": float("nan"), "mean_rank": float("nan")}
    return {
        "median_rank": float(np.median(all_ranks)),
        "mean_rank": float(np.mean(all_ranks)),
    }


def evaluate_direction(
    scores: torch.Tensor,
    positive_pairs: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
    device: str = "cpu",
    want_rank_metrics: bool = True,
    batch_size: int = 64,
) -> Dict[str, float]:
    """R@K plus, optionally, rank statistics for one retrieval direction."""
    out = clip_style_recall(scores, positive_pairs, ks, device, batch_size)
    out["Rsum"] = sum(out[f"R@{k}"] for k in ks)
    if want_rank_metrics:
        out.update(rank_metrics(scores, positive_pairs, device, batch_size))
    return out
