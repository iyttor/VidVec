#!/usr/bin/env bash
# Reproduce the paper's results in one command. For each benchmark in turn: retrieval in
# both directions (T2V + V2T), whose VidVec-O results are printed and saved; then
# reranking of the T2V top-K (default 5), whose VidVec results are printed and saved.
# Options are documented in the README.
#
#   VIDVEC_DATA_ROOT=/path/to/benchmarks ./scripts/eval.sh
#
# A benchmark that fails is recorded and skipped rather than taking the whole sweep
# down with it. --resume skips any stage that already finished successfully (reranking
# only if it used the same --rerank-k).
set -uo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT="iyttor/VidVec-7B"
OUT="$CODE_DIR/results/vidvec"
DATASETS="msrvtt"
RERANK_K=5
RESUME=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)       CKPT="$2"; shift 2 ;;
        --out)        OUT="$2"; shift 2 ;;
        --datasets)   DATASETS="$2"; shift 2 ;;
        --rerank-k)   RERANK_K="$2"; shift 2 ;;
        --resume)     RESUME=1; shift ;;
        *)            EXTRA+=("$1"); shift ;;
    esac
done

if [[ -z "${VIDVEC_DATA_ROOT:-}" ]]; then
    echo "Set VIDVEC_DATA_ROOT to the directory holding the benchmarks (see the README)." >&2
    exit 1
fi

mkdir -p "$OUT/per_dataset" "$OUT/logs" "$OUT/cands"
OUT="$(cd "$OUT" && pwd)"
cd "$CODE_DIR"

echo "=================================================="
echo " model    : $CKPT"
echo " datasets : $DATASETS"
echo " rerank   : $([[ $RERANK_K -gt 0 ]] && echo "T2V top-$RERANK_K" || echo off)"
echo " output   : $OUT"
echo "=================================================="

# retrieval_done FILE: a per-dataset result from a successful run.
retrieval_done() {
    [[ -s "$1" ]] && python -c "import json,sys; sys.exit(json.load(open(sys.argv[1])).get('status') != 'ok')" "$1" 2>/dev/null
}
# rerank_done FILE K: a rerank result computed with the same K.
rerank_done() {
    [[ -s "$1" ]] && python -c "import json,sys; sys.exit(json.load(open(sys.argv[1])).get('rank_num') != int(sys.argv[2]))" "$1" "$2" 2>/dev/null
}

# merge [DS [ARGS]]: save results.json with everything finished so far; print DS's rows
# (or all rows).
merge() {
    python eval/merge_results.py --in "$OUT/per_dataset" --out "$OUT/results.json" \
        --rerank-k "$RERANK_K" ${1:+--print-only "$1"} "${@:2}"
}

FAILED=()
for DS in $DATASETS; do
    RESULT="$OUT/per_dataset/${DS}.json"
    METRICS="$OUT/per_dataset/${DS}_rerank.json"
    LOG="$OUT/logs/${DS}.log"
    RLOG="$OUT/logs/${DS}_rerank.log"

    # -- retrieval (VidVec-O) ------------------------------------------------------
    if [[ $RESUME -eq 1 ]] && retrieval_done "$RESULT"; then
        echo "[skip] $DS retrieval (already done)"
    else
        # A rerank result belongs to the retrieval run that produced its candidates.
        rm -f "$METRICS" "$OUT/per_dataset/${DS}_rerank_scores.json"
        echo "[run ] $DS retrieval -> $LOG"
        TOKENIZERS_PARALLELISM=false python eval/run_retrieval_eval.py \
                --dataset "$DS" --ckpt "$CKPT" --results-json "$RESULT" \
                --rerank-out-dir "$OUT/cands" \
                "${EXTRA[@]+"${EXTRA[@]}"}" > "$LOG" 2>&1
        if [[ $? -ne 0 ]]; then
            echo "[fail] $DS retrieval -- see $LOG"
            FAILED+=("$DS")
            python - "$RESULT" "$DS" "$LOG" <<'PY'
import json, sys
path, dataset, log = sys.argv[1:4]
json.dump({"status": "failed", "dataset": dataset, "log": log}, open(path, "w"), indent=2)
PY
            continue
        fi
    fi
    # VidVec-O is printed and saved before reranking starts.
    merge "$DS" --print-vidvec-o

    # -- reranking (VidVec) --------------------------------------------------------
    [[ $RERANK_K -gt 0 ]] || continue
    if [[ $RESUME -eq 1 ]] && rerank_done "$METRICS" "$RERANK_K"; then
        echo "[skip] $DS rerank (already done)"
        continue
    fi
    rm -f "$METRICS" "$OUT/per_dataset/${DS}_rerank_scores.json"
    echo "[run ] $DS rerank top-$RERANK_K -> $RLOG"
    accelerate launch --num_processes 1 --main_process_port 29508 \
        eval/run_rerank_eval.py --dataset "$DS" --rank-num "$RERANK_K" \
            --cands-dir "$OUT/cands" --out-dir "$OUT/per_dataset" > "$RLOG" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "[fail] $DS rerank -- see $RLOG"
        FAILED+=("${DS}_rerank")
        continue
    fi
    merge "$DS"
done

echo
echo "All benchmarks:"
merge

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi
