#!/bin/bash
# Train VidVec: LoRA + Dual Softmax Loss on VideoUFO caption pairs.
#
# This is the exact recipe behind the released checkpoint. Run it from the repository
# root:
#
#     DATA_PATH=/data/VideoUFO.csv \
#     OUTPUT_DIR=/checkpoints/vidvec-7b \
#     bash scripts/train_vidvec.sh
#
# Hardware: 4 x B200 (180 GB), DeepSpeed ZeRO-3, bf16. The global batch of 288 is the
# number that matters -- the contrastive loss gathers embeddings across ranks, so the
# in-batch negatives are the whole 288. Reproducing the checkpoint on a different GPU
# count means keeping NUM_GPUS x PER_DEVICE_BATCH_SIZE x GRAD_ACCUM = 288, and even
# then only NUM_GPUS=4 gives bit-comparable results, because gradient accumulation
# splits the negative pool where data parallelism does not.
#
# Data: VideoUFO.csv from the WenhaoWang/VideoUFO dataset on the Hugging Face Hub,
# unmodified (only the "Brief Caption" and "Detailed Caption" columns are read; no
# videos are needed).

set -euo pipefail

[ -f ./ds_configs/zero3.json ] || {
    echo "Error: ./ds_configs/zero3.json not found. Run this script from the repository root." >&2
    exit 1
}

DATA_PATH="${DATA_PATH:?set DATA_PATH to VideoUFO.csv (huggingface.co/datasets/WenhaoWang/VideoUFO)}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the directory to write checkpoints to}"
NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-72}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
# The released checkpoint is step 210, and 210 x 288 = 60,480 examples. STOP_AT_STEP
# truncates the run there; it deliberately does NOT shorten the LR schedule, which is
# still the 7582 steps implied by two epochs over the full CSV (warmup 0.03 = 227
# steps). Set STOP_AT_STEP=0 to train the full two epochs.
STOP_AT_STEP="${STOP_AT_STEP:-210}"
# Optional HuggingFace cache for the base weights. Unset uses the default HF cache.
CACHE_DIR="${CACHE_DIR:-}"
# Logging backend of the Hugging Face Trainer; "none" disables logging.
REPORT_TO="${REPORT_TO:-none}"

[ -f "$DATA_PATH" ] || { echo "Error: DATA_PATH not found: $DATA_PATH" >&2; exit 1; }
if [ -d "$OUTPUT_DIR" ]; then
    echo "Error: OUTPUT_DIR already exists: $OUTPUT_DIR" >&2
    echo "Refusing to write into it -- move it aside or pick another directory." >&2
    exit 1
fi

GLOBAL_BATCH=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE * GRAD_ACCUM))
echo "data          $DATA_PATH"
echo "output        $OUTPUT_DIR"
echo "global batch  $NUM_GPUS x $PER_DEVICE_BATCH_SIZE x $GRAD_ACCUM = $GLOBAL_BATCH"
if [ "$STOP_AT_STEP" != "0" ]; then
    echo "stopping at   step $STOP_AT_STEP ($((STOP_AT_STEP * GLOBAL_BATCH)) examples)"
fi

TRAIN_ARGS=(
    --data_path "$DATA_PATH"
    --output_dir "$OUTPUT_DIR"
    --run_name "$(basename "$OUTPUT_DIR")"
    --report_to "$REPORT_TO"

    --deepspeed ./ds_configs/zero3.json
    --bf16 True
    --tf32 True
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --seed 42

    --num_train_epochs 2
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate 2e-4
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --max_grad_norm 2.0
    --logging_steps 1

    --save_strategy steps
    --save_steps 210
    --save_total_limit 1
)
if [ -n "$CACHE_DIR" ]; then
    TRAIN_ARGS+=(--cache_dir "$CACHE_DIR")
fi
if [ "$STOP_AT_STEP" != "0" ]; then
    TRAIN_ARGS+=(--stop_at_step "$STOP_AT_STEP")
fi

torchrun --nnodes=1 --nproc_per_node "$NUM_GPUS" --rdzv_backend c10d --rdzv_endpoint localhost:0 \
    train/train_captions_videollama3.py "${TRAIN_ARGS[@]}"

echo
echo "Training finished. Checkpoints in $OUTPUT_DIR:"
ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null || echo "  (none -- check the log above)"
