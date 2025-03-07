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
TASK="polite"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/style_transfer/politeness-corpus"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/style_transfer/politeness-corpus"
ALPHA=10
TOPK=150
LLM="gpt2-small"
LAYER=6

DEBUG=0

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
  --data_size -1 \
  --topk_cnt $TOPK \
  --batch_size  16\
  --source "pos" \
  --target "neg" \
  --prompt_source "pos" \
  --prompt_data_size 100 \
  --mean_type 'dif_mean' \
  --steer_type 'all' \
  --output_dir $ROOT_DIR"/results/polite/" \
  --dataset_path $DATASET_PATH \
  --prompt_path $PROMPT_PATH \
  --env_path $ROOT_DIR"/.env" \
  --save_no_steer 1 \
  --debug $DEBUG \
  --use_cache 0 \
  --gen_batch_size 16 \
  --example_prompt "But the lack of financial aid would| The passage of the AI Act wil" \
  --is_norm_delta_matrix 0 \
  # --max_new_tokens 30\
  # --is_norm_delta_matrix 0\ 
  

set +x
