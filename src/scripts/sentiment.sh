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
export CUDA_VISIBLE_DEVICES=3
# important config
TASK="sentiment"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k"
ALPHA=50
TOPK=100
LLM="gpt2-small"
LAYER=6

DEBUG=1  # 不是布尔值


# Enable command trace for debugging
set -x

# Run the Python script
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
  --data_size -1 \
  --batch_size 32 \
  --source "neg" \
  --target "pos" \
  --prompt_source "neg" \
  --mean_type 'dif_mean' \
  --steer_type 'all' \
  --output_dir $ROOT_DIR"/results/sentiment/" \
  --dataset_path $DATASET_PATH \
  --prompt_path $PROMPT_PATH \
  --env_path $ROOT_DIR"/.env" \
  --save_no_steer 1 \
  --debug $DEBUG \
  --use_cache 0 \
  --gen_batch_size 16 \
  --example_prompt "But the lack of financial aid would| The passage of the AI Act will" \
  --is_norm_delta_matrix 0 \
  # --max_new_tokens 30\
  # --is_norm_delta_matrix 0\ 
  


## TO 孙泽凯 
# data_size改为-1 跑全量
#   --source "neu" \ 实现中性到正向的扭转
  # --target "pos" \ 

# 这是COT的测试prompt,COT不能用gpt """ Q: Cody goes to the store and buys $40 worth of stuff.  The taxes were 5%.  After taxes, he got an $8 discount.  Cody and his friend split the final price equally. How much did Cody pay?
# # A:"""
# Disable command trace

set +x
