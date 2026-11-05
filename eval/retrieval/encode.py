"""Turning a dataset split into a feature matrix.

Two things here are load-bearing and were wrong (or fragile) before:

1. **Rows are rebuilt from an ``{id: row}`` dict, not ``np.argsort``.** Accelerate
   pads the final uneven batch by repeating samples. ``gather_for_metrics``
   truncates, ``gather_object`` does not -- hence the legacy ``[:len(embed)]``
   slice. Keying by dataset id makes a repeat overwrite itself harmlessly;
   argsort would instead yield ``N+k`` rows and misalign every subsequent metric.
2. **Batch size is 1.** ``VideoLLaMA3RetrievalWrapper`` asserts it for video,
   and so does the upstream token-compression path.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Set

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

__all__ = ["tensors_to_device", "encode_split"]


def tensors_to_device(batch: Dict[str, Any], device, dtype) -> Dict[str, Any]:
    """Move to device, casting only the pixel tensors. Verbatim port."""
    for key in list(batch.keys()):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            if key in ("pixel_values", "pixel_values_videos"):
                batch[key] = value.to(device).to(dtype)
            else:
                batch[key] = value.to(device)
    return batch


def _rows_by_id(feature_chunks, id_chunks, expected_ids: Set[int]):
    """Collapse gathered chunks into one matrix ordered by ascending dataset id."""
    by_id: Dict[int, torch.Tensor] = {}
    for feats, ids in zip(feature_chunks, id_chunks):
        for row, item_id in enumerate(ids):
            by_id[int(item_id)] = feats[row]

    ids_sorted = sorted(by_id)
    missing = set(expected_ids) - set(ids_sorted)
    if missing:
        raise RuntimeError(
            f"encode produced no embedding for {len(missing)} item(s), e.g. {sorted(missing)[:8]}"
        )
    features = torch.stack([by_id[i] for i in ids_sorted]) if ids_sorted else torch.empty(0)
    return np.asarray(ids_sorted, dtype=np.int64), features


def encode_split(
    model,
    dataset,
    collator,
    *,
    accelerator,
    desc: str = "encode",
) -> Dict[str, Any]:
    """Encode every item of ``dataset`` into an L2-normalised feature matrix.

    Returns ``{"ids", "features", "seconds"}`` with ``ids`` ascending and
    ``features[i]`` belonging to ``ids[i]``.
    """
    started = time.time()

    # Batch size 1 and no workers: one decoded video is up to 180 frames of
    # pixel_values, and worker->parent tensors go through /dev/shm, which containers
    # commonly cap at 1 GB ("Bus error ... insufficient shared memory").
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn=collator)
    loader = accelerator.prepare(loader)

    device = accelerator.device
    dtype = next(model.parameters()).dtype

    feature_chunks: List[torch.Tensor] = []
    id_chunks: List[List[int]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, disable=not accelerator.is_main_process):
            batch = tensors_to_device(batch, device, dtype)
            embeds, batch_ids = model(**batch, inference=True)

            embeds = F.normalize(embeds, dim=-1)
            embeds = accelerator.gather_for_metrics(embeds)
            # gather_object does not truncate accelerate's last-batch padding the
            # way gather_for_metrics does, hence the slice.
            batch_ids = accelerate.utils.gather_object(batch_ids)[: len(embeds)]

            feature_chunks.append(embeds.detach().float().cpu())
            id_chunks.append([int(i) for i in batch_ids])

    ids, features = _rows_by_id(feature_chunks, id_chunks, set(range(len(dataset))))
    return {"ids": ids, "features": features, "seconds": time.time() - started}
