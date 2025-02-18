#!/bin/bash

# Root path of project
cd /home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src

# GPU id
export CUDA_VISIBLE_DEVICES=2
# important config
DATASET_PATH="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/data/toxicity/jigsaw-unintended-bias-in-toxicity-classification"
PROMPT_PATH="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/data/toxicity/prompts"
ALPHA=101
LLM="gpt2-small"
LAYER=6
DEBUG=0  # 不是布尔值

# Enable command trace for debugging
set -x

# Run the Python script
/home/ckqsudo/miniconda3/envs/SAE/bin/python main_control_LLM_v2.py \
  --task "toxicity" \
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
  --output_dir "./results/toxicity" \
  --dataset_path $DATASET_PATH \
  --prompt_path $PROMPT_PATH \
  --env_path "/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env" \
  --save_no_steer 1 \
  --debug $DEBUG \

# Disable command trace
set +x
