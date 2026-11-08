"""CPU-only tests for the retrieval core: pairs, metrics, scoring, registry.

No GPU, no model, no encoding -- these run in seconds and cover the parts most
likely to be silently wrong (per-dataset kwargs, pair ordering, the scaling
operator's asymmetry).

    python -m pytest code/eval/tests/test_retrieval.py -q
    python code/eval/tests/test_retrieval.py
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import torch

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from eval.retrieval import registry  # noqa: E402
from eval.retrieval.metrics import clip_style_recall, rank_metrics, recall_at_k  # noqa: E402
from eval.retrieval.pairs import assert_pairs_sane, build_positive_pairs  # noqa: E402
from eval.retrieval.scoring import apply_score_scaling  # noqa: E402


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def test_scaling_matches_the_reference_expression():
    torch.manual_seed(0)
    scores = torch.randn(7, 11)
    for factor in (0.5, 20.0, 40.0, 1000.0):
        expected = scores * torch.nn.functional.softmax(factor * scores, dim=0)
        assert torch.equal(apply_score_scaling(scores, factor), expected)


def test_scaling_is_disabled_for_nonpositive_factors():
    scores = torch.randn(4, 4)
    for factor in (0, 0.0, -1, None):
        assert torch.equal(apply_score_scaling(scores, factor), scores)


def test_scaling_is_not_transpose_symmetric():
    """Each direction must pass its OWN matrix; transposing the result differs."""
    torch.manual_seed(1)
    scores = torch.randn(5, 8)
    assert not torch.allclose(
        apply_score_scaling(scores.t(), 20.0), apply_score_scaling(scores, 20.0).t()
    )


def test_scaling_preserves_shape_and_finiteness():
    scores = torch.randn(6, 9) * 100
    out = apply_score_scaling(scores, 40.0)
    assert out.shape == scores.shape and torch.isfinite(out).all()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_recall_at_k_on_a_hand_checked_matrix():
    # 3 queries x 4 candidates; positives on the diagonal.
    scores = torch.tensor([
        [9.0, 1.0, 0.0, 0.0],   # positive col 0 ranks 1st -> hit at every k
        [1.0, 0.5, 9.0, 0.0],   # positive col 1 ranks 3rd (9.0, 1.0, 0.5) -> hit only at k>=3
        [1.0, 2.0, 3.0, 0.5],   # positive col 2 ranks 1st -> hit at every k
    ])
    pos = torch.zeros(3, 4, dtype=torch.bool)
    pos[0, 0] = pos[1, 1] = pos[2, 2] = True

    assert recall_at_k(scores, pos, 1).tolist() == [1.0, 0.0, 1.0]
    got = clip_style_recall(scores, pos, ks=(1, 2, 3), device="cpu")
    assert abs(got["R@1"] - 2 / 3) < 1e-6
    assert abs(got["R@2"] - 2 / 3) < 1e-6
    assert abs(got["R@3"] - 1.0) < 1e-6

    ranks = rank_metrics(scores, pos, device="cpu")
    assert ranks["median_rank"] == 1.0
    assert abs(ranks["mean_rank"] - (1 + 3 + 1) / 3) < 1e-6


def test_clip_style_recall_counts_a_query_once_with_many_positives():
    scores = torch.tensor([[5.0, 4.0, 0.0, 0.0]])
    pos = torch.tensor([[True, True, False, False]])
    # Both positives are in the top-2, but R@k is still capped at 1 per query.
    assert clip_style_recall(scores, pos, ks=(2,), device="cpu")["R@2"] == 1.0


def _reference_rank_metrics(scores, pos):
    """The original per-row loop from eval_vatex.py, kept as the oracle."""
    ranks = []
    for i in range(scores.shape[0]):
        row = scores[i]
        idx = torch.nonzero(pos[i]).flatten().tolist()
        if not idx:
            continue
        best = max(row[j].item() for j in idx)
        ranks.append(int((row > best).sum().item()) + 1)
    return float(np.median(ranks)), float(np.mean(ranks))


def test_vectorised_rank_metrics_match_the_original_loop():
    torch.manual_seed(7)
    for _ in range(5):
        scores = torch.randn(23, 31)
        pos = torch.rand(23, 31) > 0.85
        for i in range(23):  # guarantee every row has a positive
            pos[i, int(torch.randint(31, (1,)))] = True
        want_med, want_mean = _reference_rank_metrics(scores, pos)
        got = rank_metrics(scores, pos, device="cpu", batch_size=7)
        assert abs(got["median_rank"] - want_med) < 1e-6
        assert abs(got["mean_rank"] - want_mean) < 1e-6


def test_rank_metrics_count_ties_optimistically():
    scores = torch.tensor([[1.0, 1.0, 1.0]])
    pos = torch.tensor([[False, True, False]])
    # Nothing is strictly better, so the positive ranks 1st.
    assert rank_metrics(scores, pos, device="cpu")["median_rank"] == 1.0


# --------------------------------------------------------------------------
# pairs
# --------------------------------------------------------------------------


class _FakeDS:
    def __init__(self, n, **attrs):
        self._n = n
        for k, v in attrs.items():
            setattr(self, k, v)

    def __len__(self):
        return self._n


def _spec(pairs):
    return registry.DatasetSpec(
        name="fake", retrieval_cls=object, rerank_cls=object, paths={}, pairs=pairs,
        expected_sizes=(0, 0),
    )


def test_diagonal_pairing_is_the_identity():
    spec = _spec(registry.DIAGONAL)
    v, t = _FakeDS(4), _FakeDS(4, texts=list(range(4)))
    pos = build_positive_pairs(spec, v, t, np.arange(4), np.arange(4))
    assert torch.equal(pos, torch.eye(4, dtype=torch.bool))
    assert_pairs_sane(pos, "fake", "diagonal")


def test_gt_index_pairing_handles_many_captions_per_video():
    spec = _spec(registry.PairSpec("gt_index", lambda ds: range(len(ds)),
                                   lambda ds: list(ds.text2video_gt_index)))
    v = _FakeDS(3)
    t = _FakeDS(6, texts=list(range(6)), text2video_gt_index=[0, 0, 1, 1, 2, 2])
    pos = build_positive_pairs(spec, v, t, np.arange(3), np.arange(6))
    assert pos.sum().item() == 6
    assert pos[0, 0] and pos[2, 1] and pos[5, 2]
    assert_pairs_sane(pos, "fake", "gt_index")


def test_caption_map_pairing_with_string_keys():
    """The VATEX shape: video keys are names, not indices."""
    spec = _spec(registry.PairSpec(
        "caption_map",
        lambda ds: list(ds.video_to_captions.keys()),
        lambda ds: [ds.caption_to_video[i] for i in range(len(ds.texts))],
    ))
    v = _FakeDS(2, video_to_captions={"vidA": [], "vidB": []})
    t = _FakeDS(4, texts=list(range(4)),
                caption_to_video={0: "vidA", 1: "vidA", 2: "vidB", 3: "vidB"})
    pos = build_positive_pairs(spec, v, t, np.arange(2), np.arange(4))
    assert pos[0, 0] and pos[1, 0] and pos[2, 1] and pos[3, 1]


def test_pairing_is_invariant_to_gather_order():
    """The regression test for the multi-GPU bug in the legacy didemo/anet code.

    Row order comes from accelerate's gather, which is not the dataset order.
    Permuting it must permute the matrix, never scramble the correspondence.
    """
    spec = _spec(registry.PairSpec("gt_index", lambda ds: range(len(ds)),
                                   lambda ds: list(ds.text2video_gt_index)))
    v = _FakeDS(4)
    t = _FakeDS(8, texts=list(range(8)), text2video_gt_index=[0, 1, 2, 3, 0, 1, 2, 3])

    base = build_positive_pairs(spec, v, t, np.arange(4), np.arange(8))

    rng = np.random.default_rng(0)
    vperm, tperm = rng.permutation(4), rng.permutation(8)
    shuffled = build_positive_pairs(spec, v, t, vperm, tperm)

    assert torch.equal(shuffled, base[np.ix_(tperm, vperm)])


def test_v2t_is_just_the_transpose():
    spec = _spec(registry.PairSpec("gt_index", lambda ds: range(len(ds)),
                                   lambda ds: list(ds.text2video_gt_index)))
    v, t = _FakeDS(2), _FakeDS(4, texts=list(range(4)), text2video_gt_index=[0, 0, 1, 1])
    pos = build_positive_pairs(spec, v, t, np.arange(2), np.arange(4))
    manual = torch.zeros(2, 4, dtype=torch.bool)
    for text_idx, video_idx in enumerate([0, 0, 1, 1]):
        manual[video_idx, text_idx] = True
    assert torch.equal(pos.t(), manual)


def test_empty_row_is_rejected_not_returned_as_nan():
    pos = torch.zeros(3, 3, dtype=torch.bool)
    pos[0, 0] = pos[1, 1] = True  # row 2 has no positive
    try:
        assert_pairs_sane(pos, "fake", "gt_index")
    except ValueError as exc:
        assert "no positive" in str(exc)
    else:
        raise AssertionError("expected a ValueError for the empty row")


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_registry_lists_exactly_the_documented_benchmarks():
    assert sorted(registry.list_datasets()) == ["activitynet", "didemo", "msrvtt", "msvd", "vatex"]


def test_registry_paths_all_exist():
    """Only meaningful where the benchmarks are present."""
    if not os.environ.get("VIDVEC_DATA_ROOT"):
        return
    for name in registry.list_datasets():
        registry.dataset_paths(registry.get_spec(name))


# --------------------------------------------------------------------------
# reranking
# --------------------------------------------------------------------------


def test_rerank_ties_keep_first_stage_order():
    from eval.run_rerank_eval import recall_at_k as rerank_recall

    # Both candidates scored equally: the stable sort keeps the first-stage order,
    # so the positive in column 1 is not at rank 1.
    scores = torch.tensor([[0.5, 0.5], [0.2, 0.9]])
    out = rerank_recall(scores, [1, 1], ks=(1, 2))
    assert out == {"R@1": 50.0, "R@2": 100.0}


# --------------------------------------------------------------------------
# training loss
# --------------------------------------------------------------------------


def test_dsl_loss_matches_the_reference_expression():
    import torch.nn.functional as F

    from models.videollama3_ret import dsl_loss

    torch.manual_seed(0)
    a, b = torch.randn(6, 8), torch.randn(6, 8)
    logits = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).t() / 0.07
    w = F.softmax(logits, dim=1) * F.softmax(logits, dim=0)
    tilde = logits + torch.log(w + 1e-8)
    target = torch.arange(6)
    expected = 0.5 * (F.cross_entropy(tilde, target) + F.cross_entropy(tilde.t(), target))
    assert torch.allclose(dsl_loss(a, b), expected)
