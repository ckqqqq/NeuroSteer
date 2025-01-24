#!/bin/bash

# Root path of project
cd /home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src

# GPU id
export CUDA_VISIBLE_DEVICES=3
# important config
TASK="cot"
# DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/math/COT_GSM8k"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/math/COT_GSM8k"
ALPHA=300
# LLM="meta-llama/Llama-3.1-8B"
LLM="gemma-2-2b"
LAYER=0
DEBUG=1  # 不是布尔值
DEVICE="cpu"
BATCH_SIZE=8
# Enable command trace for debugging
set -x

# Run the Python script
/home/ckqsudo/miniconda3/envs/SAE/bin/python test_llama.py \
  --task $TASK \
  --layer $LAYER \
  --LLM $LLM \
  --seed 42 \
  --data_size 1000 \
  --device $DEVICE \
  --alpha $ALPHA \
  --method "val_mul" \
  --topk_mean 100 \
  --topk_cnt 100 \
  --batch_size $BATCH_SIZE \
  --source "pos" \
  --target "neg" \
  --mean_type 'dif_mean' \
  --steer_type 'last' \
  --output_dir "./results/cot/" \
  --dataset_path $DATASET_PATH \
  --prompt_path $PROMPT_PATH \
  --env_path "/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env" \
  --save_no_steer 1 \
  --debug $DEBUG \
  # --is_norm_delta_matrix 0\ 
  

# Disable command trace
set +x
