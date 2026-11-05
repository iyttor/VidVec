#!/usr/bin/env python
"""Merge the per-dataset retrieval and rerank JSONs into one file, and print a summary
table. Run by ``scripts/eval.sh``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ORDER = ["msrvtt", "msvd", "didemo", "activitynet", "vatex"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", required=True, help="dir of per-dataset JSONs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rerank-k", type=int, default=5,
                    help="include reranking results computed with this K; 0 ignores them")
    ap.add_argument("--print-only", help="print only this benchmark's rows (all are saved)")
    ap.add_argument("--print-vidvec-o", action="store_true", help="print only the VidVec-O rows")
    args = ap.parse_args()

    datasets = {}
    for path in sorted(glob.glob(os.path.join(args.in_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.endswith(("_rerank", "_rerank_scores")):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                datasets[name] = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            datasets[name] = {"status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}

    if not datasets:
        raise SystemExit(f"No result JSONs found in {args.in_dir}")

    for name, entry in datasets.items():
        path = os.path.join(args.in_dir, f"{name}_rerank.json")
        if args.rerank_k <= 0 or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        if m.get("rank_num") == args.rerank_k and entry.get("status") == "ok":
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

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(f"\n{'':<14}{'':<10}{'T2V R@1':>8}{'R@5':>6}{'R@10':>6}{'V2T R@1':>10}{'R@5':>6}{'R@10':>6}")
    for name, entry in payload["datasets"].items():
        if args.print_only and name != args.print_only:
            continue
        if entry.get("status") != "ok":
            print(f"{name:<14}{entry.get('status', '?')}")
            continue
        t2v = {k: entry["results"]["t2v"]["scaled"][k] * 100 for k in ("R@1", "R@5", "R@10")}
        v2t = {k: entry["results"]["v2t"]["scaled"][k] * 100 for k in ("R@1", "R@5", "R@10")}
        rows = [("VidVec-O", t2v)]
        if "rerank" in entry and not args.print_vidvec_o:
            rows.append(("VidVec", entry["rerank"]["after"]))
        for model, t in rows:
            print(f"{name:<14}{model:<10}{t['R@1']:>8.1f}{t['R@5']:>6.1f}{t['R@10']:>6.1f}"
                  f"{v2t['R@1']:>10.1f}{v2t['R@5']:>6.1f}{v2t['R@10']:>6.1f}")
    if not args.print_only:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
