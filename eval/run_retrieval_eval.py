#!/usr/bin/env python
"""Text-to-video and video-to-text retrieval on one benchmark. Run by ``scripts/eval.sh``.

Score scaling is applied unconditionally using the constants in
``retrieval/scoring.py``; raw (unscaled) metrics are reported alongside, for
reference.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from eval.retrieval.config import add_arguments, build_config  # noqa: E402
from eval.retrieval.metrics import evaluate_direction  # noqa: E402
from eval.retrieval.pairs import assert_pairs_sane, build_positive_pairs, pair_stats  # noqa: E402
from eval.retrieval.pipeline import prepare_scores  # noqa: E402
from eval.retrieval.registry import get_spec, list_datasets  # noqa: E402
from eval.retrieval.report import (  # noqa: E402
    DirectionResult, build_payload, print_report, write_json,
)
from eval.retrieval.scoring import T2V_SCALE, V2T_SCALE, apply_score_scaling  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=list_datasets())
    add_arguments(parser)
    cfg = build_config(parser.parse_args())
    spec = get_spec(cfg.dataset)

    bundle = prepare_scores(cfg, spec)
    scores = bundle["scores"]

    pos_t2v = build_positive_pairs(
        spec, bundle["video_ds"], bundle["text_ds"], bundle["video_ids"], bundle["text_ids"]
    )
    assert_pairs_sane(pos_t2v, cfg.dataset, spec.pairs.kind)

    results = {}
    for direction in ("t2v", "v2t"):
        mat = scores if direction == "t2v" else scores.t()
        pos = pos_t2v if direction == "t2v" else pos_t2v.t()
        factor = T2V_SCALE if direction == "t2v" else V2T_SCALE
        results[direction] = DirectionResult(
            direction=direction,
            num_queries=mat.shape[0],
            num_candidates=mat.shape[1],
            raw=evaluate_direction(mat, pos, device="cpu"),
            scaled=evaluate_direction(apply_score_scaling(mat, factor), pos, device="cpu"),
            scale_factor=factor,
            seconds=bundle["seconds"],
        )

    print_report(results, cfg.dataset, bundle["model_label"])

    if cfg.rerank_out_dir:
        from eval.retrieval.rerank_export import export_t2v_candidates

        export_t2v_candidates(
            dataset=cfg.dataset,
            query_names=[str(t) for t in bundle["text_ids"]],
            candidate_names=[str(v) for v in bundle["video_ids"]],
            scores=scores,
            out_dir=cfg.rerank_out_dir,
        )
        print(f"[rerank] wrote candidate lists to {cfg.rerank_out_dir}")

    if cfg.results_json:
        counts = {"num_videos": int(scores.shape[1]), "num_texts": int(scores.shape[0]),
                  **pair_stats(pos_t2v)}
        write_json(cfg.results_json, build_payload(
            results=results, dataset=cfg.dataset, spec=spec, cfg=cfg,
            model_label=bundle["model_label"], counts=counts,
            timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
