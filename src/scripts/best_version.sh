#!/bin/bash
#  获取根目录
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
export CUDA_VISIBLE_DEVICES=3

# Important config
TASK="sentiment"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k"
ALPHA=100
LLM="gpt2-small"
DEBUG=0  # 不是布尔值

# Enable command trace for debugging
set -x

# 修正TOPK循环：使用空格分隔的列表替代逗号分隔字符串
TOPK_VALUES=(100)

for TOPK in "${TOPK_VALUES[@]}"; do
  for LAYER in {6..6}; do  # 共12层（0-11）
    /home/ckqsudo/miniconda3/envs/SAE/bin/python main_control_LLM_v2.py \
      --task $TASK \
      --layer $LAYER \
      --LLM $LLM \
      --seed 42 \
      --data_size -1 \
      --device "cuda" \
      --alpha $ALPHA \
      --method "val_mul" \
      --topk_cnt $TOPK \
      --batch_size 32 \
      --source "neg" \
      --target "pos" \
      --prompt_source "neg" \
      --prompt_data_size -1 \
      --mean_type 'dif_mean' \
      --steer_type 'all' \
      --output_dir "./results/waste_/" \
      --dataset_path $DATASET_PATH \
      --prompt_path $PROMPT_PATH \
      --env_path $ROOT_DIR/.env \
      --save_no_steer 0 \
      --debug $DEBUG \
      --use_cache 0 \
      --repeat_num 2 \
      --gen_batch_size 16 \
      --example_prompt "But the lack of financial aid would| I feel " \

    echo "Finished processing layer $LAYER with TOPK $TOPK"
  done
done

# Disable command trace
set +x
