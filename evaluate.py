#!/usr/bin/env python
"""VidVec evaluation on one benchmark, in two stages, plus the merged summary.

    python evaluate.py retrieve --dataset msrvtt --results-json out/msrvtt.json --rerank-out-dir out/cands
    accelerate launch --num_processes 1 evaluate.py rerank --dataset msrvtt --rank-num 5 \\
        --cands-dir out/cands --out-dir out
    python evaluate.py merge --in out --out out/results.json

``scripts/eval.sh`` runs all three over every benchmark; use that to reproduce the paper.

retrieve  Text-to-video and video-to-text retrieval with VidVec embeddings (VidVec-O).
rerank    Rerank each caption's text-to-video top-K with zero-shot VideoLLaMA3-7B
          (VidVec).
merge     Merge the per-benchmark JSONs into one file and print a summary table.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

from vidvec.prompts import TEXT_PROMPT, VIDEO_FPS, VIDEO_MAX_FRAMES, VIDEO_PROMPT

DEFAULT_CKPT = "iyttor/VidVec-7B"
BENCHMARKS = ["msrvtt", "msvd", "didemo", "activitynet", "vatex"]


@dataclass
class RetrievalConfig:
    dataset: str
    ckpt: str
    video_prompt: str
    text_prompt: str
    video_fps: int
    video_max_frames: int
    results_json: Optional[str]
    rerank_out_dir: Optional[str]
    # Dual-softmax inference temperature.
    t2v_temp: float = 20
    v2t_temp: float = 35


def retrieve(args: argparse.Namespace) -> int:
    from vidvec.benchmarks import load_benchmark
    from vidvec.report import DirectionResult, build_payload, print_report, write_json
    from vidvec.retrieval import export_t2v_candidates, positive_pairs, prepare_scores, recall

    cfg = RetrievalConfig(
        dataset=args.dataset, ckpt=args.ckpt,
        video_prompt=args.video_prompt, text_prompt=args.text_prompt,
        video_fps=args.video_fps, video_max_frames=args.video_max_frames,
        results_json=args.results_json, rerank_out_dir=args.rerank_out_dir,
    )
    bench = load_benchmark(cfg.dataset)

    bundle = prepare_scores(cfg, bench)
    pos_t2v = positive_pairs(bench)

    results = {}
    for direction in ("t2v", "v2t"):
        mat = bundle[direction]
        pos = pos_t2v if direction == "t2v" else pos_t2v.t()
        results[direction] = DirectionResult(
            direction=direction,
            num_queries=mat.shape[0],
            num_candidates=mat.shape[1],
            metrics=recall(mat, pos),
            seconds=bundle["seconds"],
        )

    print_report(results, cfg.dataset, bundle["model_label"])

    if cfg.rerank_out_dir:
        export_t2v_candidates(cfg.dataset, bundle["t2v"], cfg.rerank_out_dir)
        print(f"[rerank] wrote candidate lists to {cfg.rerank_out_dir}")

    if cfg.results_json:
        num_texts, num_videos = bundle["t2v"].shape
        counts = {"num_videos": int(num_videos), "num_texts": int(num_texts)}
        write_json(cfg.results_json, build_payload(
            results=results, dataset=cfg.dataset, cfg=cfg,
            model_label=bundle["model_label"], counts=counts,
            timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
        ))
    return 0


def rerank(args: argparse.Namespace) -> int:
    from vidvec.benchmarks import RerankDataset, load_benchmark
    from vidvec.rerank import rerank_metrics, score_pairs
    from vidvec.retrieval import candidate_files

    query_file, cand_file = candidate_files(args.cands_dir, args.dataset)
    for path in (query_file, cand_file):
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}; run `evaluate.py retrieve --rerank-out-dir` first.")
    dataset = RerankDataset(load_benchmark(args.dataset), query_file, cand_file, args.rank_num)
    print(f"{args.dataset}: reranking the top-{args.rank_num} of {len(dataset.queries)} captions")

    pair_scores = score_pairs(dataset)
    summary = rerank_metrics(dataset, pair_scores)

    print("\nBefore reranking: " + "  ".join(f"{k}={v:.2f}" for k, v in summary["before_rerank"].items()))
    print("After reranking:  " + "  ".join(f"{k}={v:.2f}" for k, v in summary["after_rerank"].items()))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{args.dataset}_rerank")
    per_query = {str(q): pair_scores[i * args.rank_num:(i + 1) * args.rank_num]
                 for i, q in enumerate(dataset.queries)}
    with open(f"{stem}_scores.json", "w") as fh:
        json.dump(per_query, fh, indent=2)
    with open(f"{stem}.json", "w") as fh:
        json.dump({**summary, "dataset": args.dataset, "rank_num": args.rank_num}, fh, indent=2)
    print(f"\nwrote {stem}.json")
    return 0


def merge(args: argparse.Namespace) -> int:
    from vidvec.report import merge_results

    merge_results(args.in_dir, args.out, rerank_k=args.rerank_k,
                  print_only=args.print_only, print_vidvec_o=args.print_vidvec_o)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("retrieve", help="first stage: embedding retrieval (VidVec-O)")
    p.add_argument("--dataset", required=True, choices=BENCHMARKS)
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Hugging Face model id")
    p.add_argument("--video-prompt", default=VIDEO_PROMPT)
    p.add_argument("--text-prompt", default=TEXT_PROMPT)
    p.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    p.add_argument("--video-max-frames", type=int, default=VIDEO_MAX_FRAMES)
    p.add_argument("--results-json", default=None)
    p.add_argument("--rerank-out-dir", default=None,
                   help="write the text-to-video candidate lists here for reranking")
    p.set_defaults(func=retrieve)

    p = sub.add_parser("rerank", help="second stage: rerank the text-to-video top-K (VidVec)")
    p.add_argument("--dataset", required=True, choices=BENCHMARKS)
    p.add_argument("--rank-num", type=int, required=True, help="K: candidates reranked per caption")
    p.add_argument("--cands-dir", required=True, help="--rerank-out-dir of the retrieve stage")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=rerank)

    p = sub.add_parser("merge", help="merge per-benchmark results and print a summary")
    p.add_argument("--in", dest="in_dir", required=True, help="dir of per-benchmark JSONs")
    p.add_argument("--out", required=True)
    p.add_argument("--rerank-k", type=int, default=5,
                   help="include reranking results computed with this K; 0 ignores them")
    p.add_argument("--print-only", help="print only this benchmark's rows (all are saved)")
    p.add_argument("--print-vidvec-o", action="store_true", help="print only the VidVec-O rows")
    p.set_defaults(func=merge)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
