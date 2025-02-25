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

# Important config
TASK="sentiment"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k"
ALPHA=100
LLM="gpt2-small"
DEBUG=1
OUTPUT_DIR="./results/demo/demo_v1"


# 修正TOPK循环
TOPK_VALUES=(100)

# Enable command trace for debugging
set -x


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
        --source "pos" \
        --target "neg" \
        --prompt_source "pos" \
        --prompt_data_size -1 \
        --mean_type 'dif_mean' \
        --steer_type 'all' \
        --output_dir $OUTPUT_DIR \
        --dataset_path $DATASET_PATH \
        --prompt_path $PROMPT_PATH \
        --env_path $ROOT_DIR"/.env" \
        --save_no_steer 1 \
        --debug $DEBUG \
        --use_cache 0 \
        --repeat_num 2 \
        --gen_batch_size 16 \
        --example_prompt "I am so happy | I feel " 

        echo "Finished processing: source=$SOURCE target=$TARGET layer=$LAYER topk=$TOPK"
    done
done
TASK="toxicity"
DATASET_PATH=$ROOT_DIR"/data/toxicity/jigsaw-unintended-bias-in-toxicity-classification"
PROMPT_PATH=$ROOT_DIR"/data/toxicity/prompts"
ALPHA=100
LLM="gpt2-small"
LAYER=6

for TOPK in "${TOPK_VALUES[@]}"; do
    for LAYER in {6..6}; do  # 共12层（0-11）
            /home/ckqsudo/miniconda3/envs/SAE/bin/python main_control_LLM_v2.py \
        --task $TASK \
        --layer $LAYER \
        --LLM $LLM \
        --seed 42 \
        --data_size 5000 \
        --device "cuda" \
        --alpha $ALPHA \
        --method "val_mul" \
        --topk_cnt $TOPK \
        --batch_size 8 \
        --source "pos" \
        --target "neg" \
        --prompt_source "pos" \
        --prompt_data_size -1 \
        --mean_type 'dif_mean' \
        --steer_type 'all' \
        --output_dir $OUTPUT_DIR \
        --dataset_path $DATASET_PATH \
        --prompt_path $PROMPT_PATH \
        --env_path $ROOT_DIR"/.env" \
        --save_no_steer 0 \
        --debug $DEBUG \
        --use_cache 0 \
        --repeat_num 2 \
        --gen_batch_size 16 \
        --example_prompt "He is fucking | The Act of AI is " 
        echo "Finished processing: source=$SOURCE target=$TARGET layer=$LAYER topk=$TOPK"
    done
done


# important config
TASK="polite"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/style_transfer/politeness-corpus"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/style_transfer/politeness-corpus"
ALPHA=10
TOPK=120 # ps polite 好像是广而浅
LLM="gpt2-small"
LAYER=6


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
        --batch_size 16 \
        --source "pos" \
        --target "neg" \
        --prompt_source "pos" \
        --prompt_data_size -1 \
        --mean_type 'dif_mean' \
        --steer_type 'all' \
        --output_dir $OUTPUT_DIR \
        --dataset_path $DATASET_PATH \
        --prompt_path $PROMPT_PATH \
        --env_path $ROOT_DIR"/.env" \
        --save_no_steer 0 \
        --debug $DEBUG \
        --use_cache 0 \
        --repeat_num 2 \
        --gen_batch_size 16 \
        --example_prompt "He is fucking | The Act of AI is " 
        echo "Finished processing: source=$SOURCE target=$TARGET layer=$LAYER topk=$TOPK"
    done
done
# Important config
TASK="debate"
# 哎，泽凯，注意是debate
DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/debate/StanceSentences"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/debate/StanceSentences" 
# prompt直接用的stancesententce里面的test集
ALPHA=15
LLM="gpt2-small"


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
        --source "pos" \
        --target "neg" \
        --prompt_source "pos" \
        --prompt_data_size -1 \
        --mean_type 'dif_mean' \
        --steer_type 'all' \
        --output_dir $OUTPUT_DIR \
        --dataset_path $DATASET_PATH \
        --prompt_path $PROMPT_PATH \
        --env_path $ROOT_DIR"/.env" \
        --save_no_steer 0 \
        --debug $DEBUG \
        --use_cache 0 \
        --repeat_num 2 \
        --gen_batch_size 16 \
        --example_prompt "He is fucking | The Act of AI is " 
        echo "Finished processing: source=$SOURCE target=$TARGET layer=$LAYER topk=$TOPK"
    done
done

# Disable command trace
set +x