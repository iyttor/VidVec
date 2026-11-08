"""Formatting results: the legacy stdout lines plus a structured JSON payload."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

__all__ = ["DirectionResult", "print_report", "build_payload", "write_json"]


@dataclass
class DirectionResult:
    direction: str  # "t2v" | "v2t"
    num_queries: int
    num_candidates: int
    scaled: Dict[str, float]
    raw: Dict[str, float]
    scale_factor: float
    seconds: Dict[str, float] = field(default_factory=dict)


# Legacy stdout keys, kept verbatim so existing scrapers keep matching.
_LEGACY_KEY = {"t2v": "video_retrieval_recall", "v2t": "text_retrieval_recall"}
_LEGACY_TITLE = {"t2v": "Text-to-Video Retrieval", "v2t": "Video-to-Text Retrieval"}


def _fmt(metrics: Dict[str, float]) -> str:
    bits = []
    for key, value in metrics.items():
        bits.append(f"{key}={value:.1f}" if "rank" in key else f"{key}={value * 100:.1f}")
    return "  ".join(bits)


def print_report(results: Dict[str, DirectionResult], dataset: str, model_label: str) -> None:
    print()
    print(f"===== {dataset}  |  {model_label} =====")
    for direction in ("t2v", "v2t"):
        res = results.get(direction)
        if res is None:
            continue
        print(f"\n  {_LEGACY_TITLE[direction]}  ({res.num_queries} queries -> {res.num_candidates} candidates)")
        print(f"    raw               {_fmt(res.raw)}")
        print(f"    scaled x{res.scale_factor:<8g} {_fmt(res.scaled)}")

    # The one machine-readable line per direction, in the historical format.
    # Emitted exactly once each so `grep -oP` (which returns all matches) is safe.
    print()
    for direction in ("t2v", "v2t"):
        res = results.get(direction)
        if res is None:
            continue
        key = _LEGACY_KEY[direction]
        payload = {f"{key}@{k}": res.scaled[f"R@{k}"] for k in (1, 5, 10) if f"R@{k}" in res.scaled}
        print(f"{_LEGACY_TITLE[direction]}: {payload}")


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def build_payload(
    *,
    results: Dict[str, DirectionResult],
    dataset: str,
    spec,
    cfg,
    model_label: str,
    counts: Dict[str, Any],
    timestamp: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ok",
        "dataset": dataset,
        "pairing": spec.pairs.kind,
        "run": {
            "timestamp": timestamp,
            "git_sha": _git_sha(),
        },
        "model": {"ckpt": cfg.ckpt, "label": model_label},
        "config": {
            "video_prompt": cfg.video_prompt,
            "text_prompt": cfg.text_prompt,
            "video_fps": cfg.video_fps,
            "video_max_frames": cfg.video_max_frames,
        },
        "counts": counts,
        "results": {d: asdict(r) for d, r in results.items()},
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"\n[report] wrote {path}")
