#!/usr/bin/env python
"""Rerank the text-to-video top-K of one benchmark with zero-shot VideoLLaMA3-7B.
Run by ``scripts/eval.sh --rerank-k K``.

For each caption, each of its top-K videos (as ranked by ``run_retrieval_eval.py``)
is shown to the base model with the question whether the video matches the caption.
The probability of "Yes" becomes the new score. Candidates below K keep their
first-stage order.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: E402

from collators.eval_collator import EvalDataCollator  # noqa: E402
from eval.retrieval.registry import dataset_paths, get_spec, list_datasets  # noqa: E402
from eval.retrieval.rerank_export import candidate_files  # noqa: E402
from vidvec.loading import ATTN_IMPLEMENTATION, BASE_MODEL  # noqa: E402

RECALL_KS = (1, 5, 10)


def result_stem(out_dir: str, dataset: str) -> str:
    return os.path.join(out_dir, f"{dataset}_rerank")


def score_pairs(dataset) -> list:
    """P(Yes) for every (caption, candidate video) item of ``dataset``, in item order."""
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model.eval()

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    dtype = model.dtype
    model = model.to(device)

    # Batch size 1 and no workers: one decoded clip is up to 180 frames of
    # pixel_values, and worker->parent tensors go through /dev/shm.
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False,
                            collate_fn=EvalDataCollator(tokenizer=tokenizer, processor=processor))
    dataloader, model = accelerator.prepare(dataloader, model)

    # "Yes"/"yes" and "No"/"no" are pooled before the two-way softmax.
    yes_ids = [tokenizer(w, add_special_tokens=False).input_ids[0] for w in ("Yes", "yes")]
    no_ids = [tokenizer(w, add_special_tokens=False).input_ids[0] for w in ("No", "no")]

    scores = {}
    for batch in tqdm(dataloader, desc="Reranking"):
        indexes = batch.pop("ids")
        batch.pop("labels", None)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device).to(dtype) if key in ("pixel_values", "pixel_values_videos") \
                    else value.to(device)

        outputs = model.generate(**batch, max_new_tokens=128, output_scores=True,
                                 return_dict_in_generate=True, do_sample=False)
        logits = outputs.scores[0]  # first generated token
        for row, index in enumerate(indexes):
            yes = torch.logsumexp(torch.FloatTensor([logits[row][i] for i in yes_ids]), dim=0)
            no = torch.logsumexp(torch.FloatTensor([logits[row][i] for i in no_ids]), dim=0)
            p_yes = torch.softmax(torch.FloatTensor([yes, no]), dim=0)[0].item()
            first = outputs.sequences[row][0].item() if len(outputs.sequences[row]) else None
            if first not in yes_ids + no_ids:
                text = processor.batch_decode(outputs.sequences[row:row + 1], skip_special_tokens=True)[0]
                print(f"Warning: no clear yes/no in output {text[:100]!r}; using P(Yes)={p_yes:.4f}")
            scores.setdefault(int(index), p_yes)
    return [scores[i] for i in sorted(scores)]


def recall_at_k(scores: torch.Tensor, gt_positions: list, ks=RECALL_KS) -> dict:
    """Recall@K (percent) when query ``i``'s positive is column ``gt_positions[i]``."""
    # stable=True is load-bearing, not a detail. P(Yes) comes from bf16 logits, so it
    # is coarsely quantised and exact ties are common (~2% of MSR-VTT queries have one
    # at the top). Column j is the candidate the first stage ranked j-th, so a stable
    # sort breaks ties by retrieval rank -- the sensible prior when the reranker has no
    # opinion -- and makes R@K reproducible. An unstable sort left R@1 ambiguous by
    # +/-0.35 pp, the same size as the effect being reported.
    _, order = torch.sort(scores, dim=1, descending=True, stable=True)
    out = {}
    for k in ks:
        hits = sum(1 for i, gt in enumerate(gt_positions) if gt in order[i, :k].tolist())
        out[f"R@{k}"] = hits / len(gt_positions) * 100
    return out


def rerank_metrics(dataset, pair_scores: list) -> dict:
    """Recall before and after reranking, over the queries whose positive was retrieved."""
    rank_num = dataset.rank_num
    queries = dataset.ret_query_data
    candidates = dataset.ret_cand_data
    max_candidates = max(len(c) for c in candidates)

    gt_positions, extended = [], []
    for q, (query_id, cands) in enumerate(zip(queries, candidates)):
        gt = dataset.text2video_gt_index[query_id]
        gt_positions.append(cands.index(gt) if gt in cands else -1)
        # Reranked candidates get P(Yes); the rest keep their order, below all of them.
        row = pair_scores[q * rank_num:(q + 1) * rank_num][:len(cands)]
        floor = min(row)
        row = row + [floor - 0.01 * (j + 1) for j in range(len(cands) - len(row))]
        extended.append(row + [-float("inf")] * (max_candidates - len(row)))

    valid = [i for i, gt in enumerate(gt_positions) if gt >= 0]
    before = {f"R@{k}": sum(1 for i in valid if gt_positions[i] < k) / len(valid) * 100 for k in RECALL_KS}
    after = recall_at_k(torch.tensor([extended[i] for i in valid]), [gt_positions[i] for i in valid])
    return {"before_rerank": before, "after_rerank": after, "num_queries": len(queries),
            "num_valid_queries": len(valid), "max_candidates": max_candidates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list_datasets())
    parser.add_argument("--rank-num", type=int, required=True, help="K: candidates reranked per caption")
    parser.add_argument("--cands-dir", required=True, help="--rerank-out-dir of run_retrieval_eval.py")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    query_file, cand_file = candidate_files(args.cands_dir, args.dataset)
    for path in (query_file, cand_file):
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}; run eval/run_retrieval_eval.py with --rerank-out-dir first.")
    dataset = spec.rerank_cls(ret_query_data_path=query_file, ret_cand_data_path=cand_file,
                              rank_num=args.rank_num, **dataset_paths(spec))
    print(f"{args.dataset}: reranking the top-{args.rank_num} of {len(dataset.ret_query_data)} captions")

    pair_scores = score_pairs(dataset)
    summary = rerank_metrics(dataset, pair_scores)

    print("\nBefore reranking: " + "  ".join(f"{k}={v:.2f}" for k, v in summary["before_rerank"].items()))
    print("After reranking:  " + "  ".join(f"{k}={v:.2f}" for k, v in summary["after_rerank"].items()))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = result_stem(args.out_dir, args.dataset)
    per_query = {str(q): pair_scores[i * args.rank_num:(i + 1) * args.rank_num]
                 for i, q in enumerate(dataset.ret_query_data)}
    with open(f"{stem}_scores.json", "w") as fh:
        json.dump(per_query, fh, indent=2)
    with open(f"{stem}.json", "w") as fh:
        json.dump({**summary, "dataset": args.dataset, "rank_num": args.rank_num}, fh, indent=2)
    print(f"\nwrote {stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
