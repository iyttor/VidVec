"""Ground-truth pair construction, unified across every dataset.

The five legacy scripts each rolled their own. Two of them
(``eval_didemo.py``, ``eval_activitynet.py``) indexed ``positive_pairs`` with
*dataset* indices into rows that came back in *gathered* order -- correct only at
world size 1, silently wrong otherwise.

Here a dataset contributes exactly two callables (see ``registry.PairSpec``):
a key per video and, for each caption, the key of the video it belongs to.
Everything is then resolved through a ``{key: row}`` dict built from the ids the
encoder actually returned, so the result is invariant to gather order, world
size, and dropped items. ``pos_v2t`` is just ``pos.t()``.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

__all__ = ["build_positive_pairs", "pair_stats", "assert_pairs_sane"]


def build_positive_pairs(spec, video_ds, text_ds, video_ids, text_ids) -> torch.Tensor:
    """Boolean ``(num_texts, num_videos)`` matrix aligned to the encoded rows.

    Args:
        spec: the ``DatasetSpec`` whose ``.pairs`` supplies the two key functions.
        video_ds / text_ds: the dataset objects (keys are read off them).
        video_ids / text_ids: dataset indices, in the row order of the feature
            matrices they label.
    """
    video_keys = list(spec.pairs.video_keys(video_ds))
    text_keys = list(spec.pairs.text_keys(text_ds))

    key_to_row: Dict[object, int] = {}
    for row, dataset_index in enumerate(video_ids):
        key_to_row[video_keys[int(dataset_index)]] = row

    pos = torch.zeros((len(text_ids), len(video_ids)), dtype=torch.bool)
    orphaned = 0
    for row, dataset_index in enumerate(text_ids):
        col = key_to_row.get(text_keys[int(dataset_index)])
        if col is None:
            orphaned += 1  # its video has no embedding
            continue
        pos[row, col] = True

    if orphaned:
        print(f"[pairs] {orphaned} caption(s) have no surviving video and cannot be matched")
    return pos


def pair_stats(pos: torch.Tensor) -> Dict[str, float]:
    per_text = pos.sum(dim=1)
    per_video = pos.sum(dim=0)
    return {
        "num_pairs": int(pos.sum().item()),
        "texts_without_positive": int((per_text == 0).sum().item()),
        "videos_without_positive": int((per_video == 0).sum().item()),
        "captions_per_video_mean": float(per_video.float().mean().item()),
        "captions_per_video_max": int(per_video.max().item()) if per_video.numel() else 0,
    }


def assert_pairs_sane(pos: torch.Tensor, dataset: str, kind: str) -> None:
    """Fail loudly rather than reporting NaN or a silently wrong number."""
    if pos.numel() == 0 or not bool(pos.any()):
        raise ValueError(f"[{dataset}] no positive pairs at all -- the key functions disagree")

    # recall_at_k divides by the per-row positive count, so a zero row means NaN.
    empty_t2v = int((pos.sum(dim=1) == 0).sum().item())
    empty_v2t = int((pos.sum(dim=0) == 0).sum().item())
    if empty_t2v:
        raise ValueError(
            f"[{dataset}] {empty_t2v} caption(s) have no positive video; "
            f"recall_at_k would divide by zero."
        )
    if empty_v2t:
        raise ValueError(f"[{dataset}] {empty_v2t} video(s) have no positive caption")

    if kind == "diagonal":
        expected = torch.eye(pos.shape[0], pos.shape[1], dtype=torch.bool)
        if pos.shape[0] != pos.shape[1] or not torch.equal(pos, expected):
            raise ValueError(
                f"[{dataset}] declares 1:1 pairing but the matrix is not the identity "
                f"(shape {tuple(pos.shape)}, {int(pos.sum())} pairs)"
            )
