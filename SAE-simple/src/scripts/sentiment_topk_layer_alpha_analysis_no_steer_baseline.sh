#!/bin/bash
# 获取脚本所在文件父文件夹的路径
root_dir=$(dirname "$(dirname "$(realpath "$0")")")
echo "root_dir: $root_dir"
# $root_dir/scripts
# Root path of project
cd $root_dir
# GPU id
export CUDA_VISIBLE_DEVICES=2

# Important config
TASK="sentiment"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k"
ALPHA=203
LLM="gpt2-small"
DEBUG=0  # 不是布尔值

# Enable command trace for debugging
set -x

# 修正TOPK循环：使用空格分隔的列表替代逗号分隔字符串
# TOPK_VALUES=(1000)

for TOPK in {1000..1000}; do
  for LAYER in {0..0}; do  # 共12层（0-11）
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
      --source "pos" \
      --target "neg" \
      --prompt_source "pos" \
      --prompt_data_size 500 \
      --mean_type 'dif_mean' \
      --steer_type 'last' \
      --output_dir "./results/sentiment_analysis/sentiment_grid_analysis_baseline" \
      --dataset_path $DATASET_PATH \
      --prompt_path $PROMPT_PATH \
      --env_path "/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env" \
      --save_no_steer 1 \
      --debug $DEBUG \
      --use_cache 0 \
      --repeat_num 1 \
      --gen_batch_size 16 \
      --example_prompt "But the lack of financial aid would| I feel " \

    echo "Finished processing layer $LAYER with TOPK $TOPK"
  done
done

# Disable command trace
set +x
