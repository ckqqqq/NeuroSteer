#!/bin/bash

# Root path of project
cd /home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src

# GPU id
export CUDA_VISIBLE_DEVICES=3
# important config
TASK="sentiment"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k"
ALPHA=400
LLM="gpt2-small"
LAYER=6
DEBUG=1  # 不是布尔值


# Enable command trace for debugging
set -x

# Run the Python script
/home/ckqsudo/miniconda3/envs/SAE/bin/python main_control_LLM.py \
  --task $TASK \
  --layer $LAYER \
  --LLM $LLM \
  --seed 42 \
  --data_size 1000 \
  --device "cuda" \
  --alpha $ALPHA \
  --method "val_mul" \
  --topk_mean 100 \
  --topk_cnt 100 \
  --batch_size 16 \
  --source "neg" \
  --target "pos" \
  --mean_type 'dif_mean' \
  --steer_type 'last' \
  --output_dir "./results/sentiment/" \
  --dataset_path $DATASET_PATH \
  --prompt_path $PROMPT_PATH \
  --env_path "/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env" \
  --save_no_steer 1 \
  --debug $DEBUG \
  # --is_norm_delta_matrix 0\ 
  

# Disable command trace
set +x
