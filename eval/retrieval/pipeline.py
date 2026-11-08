"""Encode both sides of a benchmark and score every caption against every video."""

from __future__ import annotations

from typing import Any, Dict

from .registry import DatasetSpec, build_datasets
from .scoring import similarity_matrix

__all__ = ["prepare_scores"]


def prepare_scores(cfg, spec: DatasetSpec) -> Dict[str, Any]:
    """Build the datasets, encode both splits, and return the (texts x videos) scores."""
    from accelerate import Accelerator

    from collators.eval_collator import EvalDataCollator
    from vidvec.loading import load_vidvec

    from .encode import encode_split

    video_ds, text_ds = build_datasets(spec, cfg.video_prompt, cfg.text_prompt)

    print(f"[pipeline] loading model from {cfg.ckpt}")
    bundle = load_vidvec(cfg.ckpt)
    collator = EvalDataCollator(
        bundle.tokenizer, bundle.processor,
        video_fps=cfg.video_fps, video_max_frames=cfg.video_max_frames,
    )

    encoded = {}
    for split, ds in (("video", video_ds), ("text", text_ds)):
        encoded[split] = encode_split(
            bundle.model, ds, collator,
            accelerator=Accelerator(mixed_precision="bf16"),
            desc=f"{cfg.dataset}/{split}",
        )

    return {
        "scores": similarity_matrix(encoded["text"]["features"], encoded["video"]["features"]),
        "video_ds": video_ds,
        "text_ds": text_ds,
        "video_ids": encoded["video"]["ids"],
        "text_ids": encoded["text"]["ids"],
        "seconds": {s: round(encoded[s]["seconds"], 1) for s in encoded},
        "model_label": bundle.label,
    }
