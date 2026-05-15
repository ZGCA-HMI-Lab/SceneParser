SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$FINETUNE_DIR"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
LOGGING_STEPS=${LOGGING_STEPS:-10}
DISABLE_TQDM=${DISABLE_TQDM:-False}
SAVE_STEPS=${SAVE_STEPS:-500}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-$FINETUNE_DIR/work_dirs/sceneparser_curriculum_stage2_mixed70pseudo30_when_available_4ep_from_stage1_3ep}
export MODEL_NAME_OR_PATH

run_name=${RUN_NAME:-"sceneparser_curriculum_stage3_mixed50pseudo50_when_available_3ep_from_stage2"}
output_dir=${OUTPUT_DIR:-"work_dirs/sceneparser_curriculum_stage3_mixed50pseudo50_when_available_3ep_from_stage2"}

if [ "$(realpath -m "$output_dir")" = "$(realpath -m "$MODEL_NAME_OR_PATH")" ]; then
    echo "ERROR: OUTPUT_DIR must be different from MODEL_NAME_OR_PATH to protect Stage-2 outputs."
    exit 1
fi

echo "torchrun config:"
echo "NNODES=$NNODES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
echo "RUN_NAME=$run_name"
echo "OUTPUT_DIR=$output_dir"
echo "SAVE_STEPS=$SAVE_STEPS"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --config configs/sft_sceneparser_curriculum_stage3_mixed50pseudo50_when_available.py \
    --deepspeed scripts/zero2.json \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 5 \
    --learning_rate 6e-6 \
    --mm_projector_lr 6e-6 \
    --vision_tower_lr 6e-7 \
    --optim adamw_torch \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps ${LOGGING_STEPS} \
    --disable_tqdm ${DISABLE_TQDM} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --gradient_checkpointing True \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --run_name ${run_name} \
    --report_to tensorboard
