"""Reporting: the per-benchmark printout and JSON, and the merged summary across
benchmarks.

Stdlib only, so ``evaluate.py merge`` (run after every stage by ``scripts/eval.sh``)
starts instantly.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

__all__ = ["DirectionResult", "print_report", "build_payload", "write_json", "merge_results"]

#: Order of the benchmarks in the merged results.
ORDER = ["msrvtt", "msvd", "didemo", "activitynet", "vatex"]


# ---------------------------------------------------------------------------------------
# One benchmark
# ---------------------------------------------------------------------------------------


@dataclass
class DirectionResult:
    direction: str  # "t2v" | "v2t"
    num_queries: int
    num_candidates: int
    metrics: Dict[str, float]
    seconds: Dict[str, float] = field(default_factory=dict)


# Stdout keys kept verbatim so existing scrapers keep matching.
_SCRAPE_KEY = {"t2v": "video_retrieval_recall", "v2t": "text_retrieval_recall"}
_TITLE = {"t2v": "Text-to-Video Retrieval", "v2t": "Video-to-Text Retrieval"}


def _fmt(metrics: Dict[str, float]) -> str:
    return "  ".join(f"{k}={metrics[k] * 100:.1f}" for k in ("R@1", "R@5", "R@10"))


def print_report(results: Dict[str, DirectionResult], dataset: str, model_label: str) -> None:
    print()
    print(f"===== {dataset}  |  {model_label} =====")
    for direction in ("t2v", "v2t"):
        res = results.get(direction)
        if res is None:
            continue
        print(f"\n  {_TITLE[direction]}  ({res.num_queries} queries -> {res.num_candidates} candidates)")
        print(f"    {_fmt(res.metrics)}")

    # One machine-readable line per direction, emitted exactly once each so
    # `grep -oP` (which returns all matches) is safe.
    print()
    for direction in ("t2v", "v2t"):
        res = results.get(direction)
        if res is None:
            continue
        key = _SCRAPE_KEY[direction]
        payload = {f"{key}@{k}": res.metrics[f"R@{k}"] for k in (1, 5, 10) if f"R@{k}" in res.metrics}
        print(f"{_TITLE[direction]}: {payload}")


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
    cfg,
    model_label: str,
    counts: Dict[str, Any],
    timestamp: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ok",
        "dataset": dataset,
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


# ---------------------------------------------------------------------------------------
# All benchmarks
# ---------------------------------------------------------------------------------------


def merge_results(in_dir: str, out: str, rerank_k: int = 5,
                  print_only: Optional[str] = None, print_vidvec_o: bool = False) -> None:
    """Merge the per-benchmark retrieval and rerank JSONs in ``in_dir`` into ``out``,
    and print a summary table.

    ``rerank_k`` selects the reranking results computed with that K (0 ignores
    them). ``print_only`` restricts the printout, not the file, to one benchmark;
    ``print_vidvec_o`` prints only the VidVec-O rows.
    """
    datasets = {}
    for path in sorted(glob.glob(os.path.join(in_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.endswith(("_rerank", "_rerank_scores")):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                datasets[name] = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            datasets[name] = {"status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}

    if not datasets:
        raise SystemExit(f"No result JSONs found in {in_dir}")

    for name, entry in datasets.items():
        path = os.path.join(in_dir, f"{name}_rerank.json")
        if rerank_k <= 0 or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        if m.get("rank_num") == rerank_k and entry.get("status") == "ok":
            entry["rerank"] = {"direction": "t2v", "K": m["rank_num"],
                               "before": m["before_rerank"], "after": m["after_rerank"]}

    first_ok = next((d for d in datasets.values() if d.get("status") == "ok"), {})
    payload = {
        "schema_version": 1,
        "model": first_ok.get("model"),
        "config": first_ok.get("config"),
        "datasets": {k: datasets[k] for k in
                     sorted(datasets, key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))},
    }

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(f"\n{'':<14}{'':<10}{'T2V R@1':>8}{'R@5':>6}{'R@10':>6}{'V2T R@1':>10}{'R@5':>6}{'R@10':>6}")
    for name, entry in payload["datasets"].items():
        if print_only and name != print_only:
            continue
        if entry.get("status") != "ok":
            print(f"{name:<14}{entry.get('status', '?')}")
            continue
        t2v = {k: entry["results"]["t2v"]["metrics"][k] * 100 for k in ("R@1", "R@5", "R@10")}
        v2t = {k: entry["results"]["v2t"]["metrics"][k] * 100 for k in ("R@1", "R@5", "R@10")}
        rows = [("VidVec-O", t2v)]
        if "rerank" in entry and not print_vidvec_o:
            rows.append(("VidVec", entry["rerank"]["after"]))
        for model, t in rows:
            print(f"{name:<14}{model:<10}{t['R@1']:>8.1f}{t['R@5']:>6.1f}{t['R@10']:>6.1f}"
                  f"{v2t['R@1']:>10.1f}{v2t['R@5']:>6.1f}{v2t['R@10']:>6.1f}")
    if not print_only:
        print(f"\nwrote {out}")
