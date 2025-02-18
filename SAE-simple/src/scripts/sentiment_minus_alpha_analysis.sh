#!/bin/bash

# Root path of project
cd /home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src

# GPU id
export CUDA_VISIBLE_DEVICES=2

# Important config
TASK="sentiment"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k"
ALPHA=203
LLM="gpt2-small"
DEBUG=1  # 不是布尔值

# Enable command trace for debugging
set -x

# Loop over layers from 0 to 12
for LAYER in {5..6}; do
  /home/ckqsudo/miniconda3/envs/SAE/bin/python main_control_LLM_v2.py \
    --task $TASK \
    --layer $LAYER \
    --LLM $LLM \
    --seed 42 \
    --data_size -1 \
    --device "cuda" \
    --alpha $ALPHA \
    --method "val_mul" \
    --topk_cnt 150 \
    --batch_size 32 \
    --source "neg" \
    --target "pos" \
    --prompt_source "neu" \
    --mean_type 'dif_mean' \
    --steer_type 'last' \
    --output_dir "./results/sentiment_alpha_analysis/" \
    --dataset_path $DATASET_PATH \
    --prompt_path $PROMPT_PATH \
    --env_path "/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env" \
    --save_no_steer 1 \
    --debug $DEBUG \
    --use_cache 0 \
    --example_prompt "But the lack of financial aid would"

  echo "Finished processing layer $LAYER"
done

# Disable command trace
set +x