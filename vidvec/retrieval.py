"""First stage: embed both sides of a benchmark, score every caption against every
video, and measure text-to-video and video-to-text recall.

Encoded rows are rebuilt in dataset order from an ``{id: row}`` dict. Accelerate pads
the final uneven batch by repeating samples, and ``gather_object`` does not truncate
that padding the way ``gather_for_metrics`` does; keying by dataset id makes a repeat
overwrite itself harmlessly. Row ``i`` is therefore item ``i`` on any world size.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "tensors_to_device", "encode_split", "prepare_scores",
    "positive_pairs",
    "recall_at_k", "batchify", "recall",
    "candidate_files", "export_t2v_candidates",
]


# ---------------------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------------------


def tensors_to_device(batch: Dict[str, Any], device, dtype) -> Dict[str, Any]:
    """Move to device, casting only the pixel tensors."""
    for key in list(batch.keys()):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            if key in ("pixel_values", "pixel_values_videos"):
                batch[key] = value.to(device).to(dtype)
            else:
                batch[key] = value.to(device)
    return batch


def _rows_by_id(feature_chunks, id_chunks, num_items: int) -> torch.Tensor:
    """Collapse gathered chunks into one matrix in dataset order."""
    by_id: Dict[int, torch.Tensor] = {}
    for feats, ids in zip(feature_chunks, id_chunks):
        for row, item_id in enumerate(ids):
            by_id[int(item_id)] = feats[row]

    missing = sorted(set(range(num_items)) - set(by_id))
    if missing:
        raise RuntimeError(f"encode produced no embedding for {len(missing)} item(s), e.g. {missing[:8]}")
    return torch.stack([by_id[i] for i in range(num_items)])


def encode_split(model, dataset, collator, *, accelerator, desc: str = "encode") -> Dict[str, Any]:
    """Encode every item of ``dataset`` into an L2-normalised feature matrix.

    Returns ``{"features", "seconds"}``; ``features[i]`` belongs to item ``i``.
    """
    import accelerate
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    started = time.time()

    # Batch size 1 and no workers: VideoLLaMA3RetrievalWrapper asserts batch size 1
    # for video, one decoded video is up to 180 frames of pixel_values, and
    # worker->parent tensors go through /dev/shm, which containers commonly cap at
    # 1 GB ("Bus error ... insufficient shared memory").
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

    features = _rows_by_id(feature_chunks, id_chunks, len(dataset))
    return {"features": features, "seconds": time.time() - started}


def prepare_scores(cfg, bench) -> Dict[str, Any]:
    """Encode both sides of ``bench`` and score them in both directions.

    ``t2v`` is ``(texts, videos)`` and ``v2t`` is ``(videos, texts)``; rows and columns
    follow the benchmark's order.
    """
    from accelerate import Accelerator

    from .benchmarks import EvalDataCollator, RetrievalDataset
    from .model import load_vidvec

    print(f"[retrieval] loading model from {cfg.ckpt}")
    bundle = load_vidvec(cfg.ckpt)
    collator = EvalDataCollator(bundle.processor, video_fps=cfg.video_fps,
                                video_max_frames=cfg.video_max_frames)

    encoded = {}
    for split in ("video", "text"):
        ds = RetrievalDataset(bench, split, cfg.video_prompt, cfg.text_prompt)
        encoded[split] = encode_split(
            bundle.model, ds, collator,
            accelerator=Accelerator(mixed_precision="bf16"),
            desc=f"{cfg.dataset}/{split}",
        )

    # Embeddings are L2-normalised, so this is cosine similarity, followed by
    # dual-softmax inference per direction.
    sim = encoded["text"]["features"] @ encoded["video"]["features"].t()
    t2v = sim * F.softmax(cfg.t2v_temp * sim, dim=0)
    v2t = sim.t() * F.softmax(cfg.v2t_temp * sim.t(), dim=0)

    return {
        "t2v": t2v,
        "v2t": v2t,
        "seconds": {s: round(encoded[s]["seconds"], 1) for s in encoded},
        "model_label": bundle.label,
    }


def positive_pairs(bench) -> torch.Tensor:
    """Boolean ``(num_texts, num_videos)`` matrix: each caption's own video."""
    pos = torch.zeros((len(bench.texts), len(bench.videos)), dtype=torch.bool)
    pos[torch.arange(len(bench.texts)), torch.tensor(bench.text2video)] = True
    return pos


# ---------------------------------------------------------------------------------------
# Metrics: R@K
# ---------------------------------------------------------------------------------------


def recall_at_k(scores: torch.Tensor, positive_pairs: torch.Tensor, k: int) -> torch.Tensor:
    """Fraction of a query's positives that land in its top-k."""
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    return nb_true_positive / nb_positive


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    """Apply a row-wise matrix metric in chunks."""
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        results.append(func(x, y, *args, **kwargs).cpu())
    return torch.cat(results)


def recall(
    scores: torch.Tensor,
    positive_pairs: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, float]:
    """R@K: the share of queries with at least one positive in their top-k."""
    out: Dict[str, float] = {}
    for k in ks:
        k_eff = min(k, scores.shape[1])
        hits = batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=k_eff)
        out[f"R@{k}"] = (hits > 0).float().mean().item()
    return out


# ---------------------------------------------------------------------------------------
# Hand-off to reranking
# ---------------------------------------------------------------------------------------


def candidate_files(out_dir: str, dataset: str) -> Tuple[str, str]:
    """``(queries, candidates)``: caption ids, and each caption's videos ranked best first."""
    stem = os.path.join(out_dir, f"{dataset}_t2v")
    return f"{stem}_query_names.json", f"{stem}_cand_names.json"


def export_t2v_candidates(dataset: str, scores: torch.Tensor, out_dir: str) -> None:
    """Rank every video for every caption of the ``(texts, videos)`` scores, best first."""
    os.makedirs(out_dir, exist_ok=True)
    query_path, cand_path = candidate_files(out_dir, dataset)
    with open(query_path, "w", encoding="utf-8") as fh:
        json.dump(list(range(scores.shape[0])), fh)
    with open(cand_path, "w", encoding="utf-8") as fh:
        json.dump(torch.argsort(scores, dim=1, descending=True).tolist(), fh)
