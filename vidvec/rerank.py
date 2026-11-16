"""Second stage: rerank the text-to-video top-K with zero-shot VideoLLaMA3-7B.

For each caption, each of its top-K videos (as ranked by the first stage) is shown to
the base model with the question whether the video matches the caption. The
probability of "Yes" becomes the new score. Candidates below K keep their first-stage
order.
"""

from __future__ import annotations

import torch

__all__ = ["RECALL_KS", "score_pairs", "recall_at_k", "rerank_metrics"]

RECALL_KS = (1, 5, 10)


def score_pairs(dataset) -> list:
    """P(Yes) for every (caption, candidate video) item of ``dataset``, in item order."""
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoProcessor

    from .benchmarks import EvalDataCollator
    from .model import ATTN_IMPLEMENTATION, BASE_MODEL

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
                            collate_fn=EvalDataCollator(processor))
    dataloader, model = accelerator.prepare(dataloader, model)

    # "Yes"/"yes" and "No"/"no" are pooled before the two-way softmax.
    yes_ids = [tokenizer(w, add_special_tokens=False).input_ids[0] for w in ("Yes", "yes")]
    no_ids = [tokenizer(w, add_special_tokens=False).input_ids[0] for w in ("No", "no")]

    scores = {}
    for batch in tqdm(dataloader, desc="Reranking"):
        indexes = batch.pop("ids")
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
    queries = dataset.queries
    candidates = dataset.candidates
    max_candidates = max(len(c) for c in candidates)

    gt_positions, extended = [], []
    for q, (query_id, cands) in enumerate(zip(queries, candidates)):
        gt = dataset.bench.text2video[query_id]
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
