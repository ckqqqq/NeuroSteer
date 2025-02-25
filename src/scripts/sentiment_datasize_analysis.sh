#!/bin/bash

#  获取根目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
cd $ROOT_DIR/src
echo "ROOT_DIR: $ROOT_DIR"
echo "SRC_DIR: $ROOT_DIR/src"
echo "RES_DIR: $ROOT_DIR/results"
echo "ENV: $ROOT_DIR/.env"
# Root path of project
cd $ROOT_DIR/src
# GPU id
export CUDA_VISIBLE_DEVICES=1

# Important config
TASK="sentiment"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k"
ALPHA=100
LLM="gpt2-small"
DEBUG=0  # 不是布尔值
TOPK=100
DATASIZES=(300 600 900 1200 1500)

# Enable command trace for debugging
set -x

for DATASIZE in ${DATASIZES[@]}; do
/home/ckqsudo/miniconda3/envs/SAE/bin/python main_control_LLM_v2.py \
    --task $TASK \
    --layer 6 \
    --LLM $LLM \
    --seed 42 \
    --data_size $DATASIZE \
    --device "cuda" \
    --alpha $ALPHA \
    --method "val_mul" \
    --topk_cnt $TOPK \
    --batch_size 32 \
    --source "pos" \
    --target "neg" \
    --prompt_source "pos" \
    --prompt_data_size 500 \
    --mean_type 'dif_mean' \
    --steer_type 'last' \
    --output_dir $ROOT_DIR"/results/sentiment_analysis/sentiment_datasize_analysis/alpha100" \
    --dataset_path $DATASET_PATH \
    --prompt_path $PROMPT_PATH \
    --env_path $ROOT_DIR"/.env" \
    --save_no_steer 0 \
    --debug $DEBUG \
    --use_cache 0 \
    --repeat_num 1 \
    --gen_batch_size 16 \
    --example_prompt "But the lack of financial aid would| I feel " \

echo "Finished processing layer $LAYER with TOPK $TOPK"
done


# Disable command trace
set +x
