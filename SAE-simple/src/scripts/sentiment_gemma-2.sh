#!/bin/bash

# Root path of project
cd /home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src

# GPU id
export CUDA_VISIBLE_DEVICES=0
# important config
TASK="sentiment"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k"
ALPHAS=(500 700 1000)
TOPK=100
LLM="gemma-2-2b"
LAYER=12

DEBUG=0  # 不是布尔值


# Enable command trace for debugging
set -x

# Run the Python script
for ALPHA in ${ALPHAS[@]}; do
/home/ckqsudo/miniconda3/envs/SAE/bin/python main_control_LLM_v2.py \
  --task $TASK \
  --layer $LAYER \
  --LLM $LLM \
  --seed 42 \
  --data_size 1500 \
  --device "cpu" \
  --alpha $ALPHA \
  --method "val_mul" \
  --topk_cnt $TOPK \
  --batch_size 32 \
  --source "neg" \
  --target "pos" \
  --prompt_source "neg" \
  --prompt_data_size 500 \
  --mean_type 'dif_mean' \
  --steer_type 'all' \
  --output_dir "./results/sentiment_gemma/" \
  --dataset_path $DATASET_PATH \
  --prompt_path $PROMPT_PATH \
  --env_path "/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env" \
  --save_no_steer 0 \
  --debug $DEBUG \
  --use_cache 0 \
  --gen_batch_size 16 \
  --example_prompt "But the lack of financial aid would| The passage of the AI Act will" \
  --is_norm_delta_matrix 0 \
  # --max_new_tokens 30\
  # --is_norm_delta_matrix 0\ 
  
done

## TO 孙泽凯 
# data_size改为-1 跑全量
#   --source "neu" \ 实现中性到正向的扭转
  # --target "pos" \ 

# 这是COT的测试prompt,COT不能用gpt """ Q: Cody goes to the store and buys $40 worth of stuff.  The taxes were 5%.  After taxes, he got an $8 discount.  Cody and his friend split the final price equally. How much did Cody pay?
# # A:"""
# Disable command trace

set +x
