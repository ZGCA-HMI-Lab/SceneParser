#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENEPARSER_ROOT="$(cd "$EVAL_DIR/.." && pwd)"
cd "$SCENEPARSER_ROOT"

MODEL_PATH=${MODEL_PATH:-"$SCENEPARSER_ROOT/finetuning/work_dirs/sceneparser_curriculum_stage3_mixed50pseudo50_when_available_3ep_from_stage2"}
TEST_JSONL=${TEST_JSONL:-"$SCENEPARSER_ROOT/datasets/val.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"$EVAL_DIR/results/curriculum_stage3_eval"}
NUM_SHARDS=${NUM_SHARDS:-8}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:--1}
MAX_TOKENS=${MAX_TOKENS:-2048}
MIN_PIXELS=${MIN_PIXELS:-$((16 * 28 * 28))}
MAX_PIXELS=${MAX_PIXELS:-$((2560 * 28 * 28))}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
CROSS_IMAGE_BATCH_SIZE=${CROSS_IMAGE_BATCH_SIZE:-16}
SKIP_INFERENCE_IF_EXISTS=${SKIP_INFERENCE_IF_EXISTS:-1}
QUESTION=${QUESTION:-$'Return a single JSON object only.\nParse [OBJ] in this image and return object bbox, part bboxes, and affordance points when available.'}

mkdir -p "$OUTPUT_DIR"
SHARD_DIR="$OUTPUT_DIR/shards"
mkdir -p "$SHARD_DIR"

MERGED_PRED_JSONL="$OUTPUT_DIR/answer.jsonl"
EVAL_JSON_FULL="$OUTPUT_DIR/eval_results_full.json"
EVAL_JSON_FILTERED="$OUTPUT_DIR/eval_results_filtered.json"
FINAL_METRICS_JSON="$OUTPUT_DIR/final_metrics.json"
EVAL_LOG_FULL="$OUTPUT_DIR/eval_results_full.log"
EVAL_LOG_FILTERED="$OUTPUT_DIR/eval_results_filtered.log"

TOTAL_RECORDS=$(python3 - "$TEST_JSONL" <<'PY'
import sys
count = 0
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            count += 1
print(count)
PY
)

if [[ "$END_IDX" == "-1" ]]; then
  EFFECTIVE_END=$TOTAL_RECORDS
else
  EFFECTIVE_END=$END_IDX
fi

TOTAL_RANGE=$((EFFECTIVE_END - START_IDX))
if (( TOTAL_RANGE <= 0 )); then
  echo "Invalid range: START_IDX=$START_IDX END_IDX=$END_IDX TOTAL_RECORDS=$TOTAL_RECORDS"
  exit 1
fi

SHARD_SIZE=$(((TOTAL_RANGE + NUM_SHARDS - 1) / NUM_SHARDS))

echo "MODEL_PATH=$MODEL_PATH"
echo "TEST_JSONL=$TEST_JSONL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "TOTAL_RECORDS=$TOTAL_RECORDS"
echo "SHARD_SIZE=$SHARD_SIZE"
echo "CROSS_IMAGE_BATCH_SIZE=$CROSS_IMAGE_BATCH_SIZE"

declare -a PIDS=()
declare -a SHARD_FILES=()

for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  SHARD_START=$((START_IDX + shard * SHARD_SIZE))
  SHARD_END=$((SHARD_START + SHARD_SIZE))
  if (( SHARD_START >= EFFECTIVE_END )); then
    break
  fi
  if (( SHARD_END > EFFECTIVE_END )); then
    SHARD_END=$EFFECTIVE_END
  fi

  SHARD_PRED="$SHARD_DIR/answer_shard_${shard}.jsonl"
  SHARD_FILES+=("$SHARD_PRED")

  if [[ "$SKIP_INFERENCE_IF_EXISTS" == "1" && -f "$SHARD_PRED" ]]; then
    echo "Shard $shard skipped: $SHARD_PRED already exists"
    continue
  fi

  GPU_ID=$shard
  echo "Launching shard $shard on GPU $GPU_ID: [$SHARD_START, $SHARD_END)"
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluation/inference_sceneparser.py \
    --model_path "$MODEL_PATH" \
    --test_jsonl_path "$TEST_JSONL" \
    --save_path "$SHARD_PRED" \
    --start_idx "$SHARD_START" \
    --end_idx "$SHARD_END" \
    --max_new_tokens "$MAX_TOKENS" \
    --min_pixels "$MIN_PIXELS" \
    --max_pixels "$MAX_PIXELS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor_parallel_size 1 \
    --cross_image_batch_size "$CROSS_IMAGE_BATCH_SIZE" \
    --question "$QUESTION" &
  PIDS+=("$!")
done

if (( ${#PIDS[@]} > 0 )); then
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
fi

python3 - "$MERGED_PRED_JSONL" "${SHARD_FILES[@]}" <<'PY'
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
shard_files = [Path(p) for p in sys.argv[2:]]
with out_path.open("w", encoding="utf-8") as fout:
    for path in shard_files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    fout.write(line)
print(out_path)
PY

python3 evaluation/metrics/sceneparser_hierarchical_metric.py \
  --data_path "$MERGED_PRED_JSONL" \
  --output_path "$EVAL_JSON_FULL" \
  > "$EVAL_LOG_FULL"

python3 evaluation/metrics/sceneparser_hierarchical_metric_with_filters.py \
  --data_path "$MERGED_PRED_JSONL" \
  --output_path "$EVAL_JSON_FILTERED" \
  --ignore_pseudo_part \
  --ignore_none_action \
  > "$EVAL_LOG_FILTERED"

python3 evaluation/tools/extract_final_metrics.py \
  --eval_json "$EVAL_JSON_FILTERED" \
  --output_path "$FINAL_METRICS_JSON"

echo "Predictions: $MERGED_PRED_JSONL"
echo "Filtered eval: $EVAL_JSON_FILTERED"
echo "Final metrics: $FINAL_METRICS_JSON"
