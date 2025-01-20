#!/bin/bash

# Default values
TASK="sentiment"
LAYER=6
LLM="gpt2-small"
SEED=42
DATA_SIZE="ALL"
DEVICE="cuda"
ALPHA=500
STEER="neg-pos"
METHOD="val_mul"
TOPK_MEAN=100
TOPK_CNT=100
BATCH_SIZE=32
SOURCE="pos"
TARGET="neg"
MEAN_TYPE="dif_mean"
STEER_TYPE="last"
DEBUG=False
OUTPUT_DIR="./results"
DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5"
PROMPT_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k"
ENV_PATH="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env"

# Parse arguments
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    --task)
      TASK="$2"
      shift 2
      ;;
    --layer)
      LAYER="$2"
      shift 2
      ;;
    --LLM)
      LLM="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --data_size)
      DATA_SIZE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    --steer)
      STEER="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --topk_mean)
      TOPK_MEAN="$2"
      shift 2
      ;;
    --topk_cnt)
      TOPK_CNT="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --mean_type)
      MEAN_TYPE="$2"
      shift 2
      ;;
    --steer_type)
      STEER_TYPE="$2"
      shift 2
      ;;
    --debug)
      DEBUG="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --prompt_path)
      PROMPT_PATH="$2"
      shift 2
      ;;
    --env_path)
      ENV_PATH="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Select task-specific arguments
if [[ "$TASK" == "sentiment" ]]; then
  DATASET_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5"
  PROMPT_PATH="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k"
elif [[ "$TASK" == "cot" ]]; then
  DATASET_PATH="/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/math/COT_GSM8k"
  PROMPT_PATH=""
  DEVICE="cpu"
  ALPHA=100
  STEER="cot-direct"
elif [[ "$TASK" == "polite" ]]; then
  DATASET_PATH="/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/style_transfer/politeness-corpus"
  PROMPT_PATH=""
  ALPHA=100
  STEER="polite-impolite"
else
  echo "Unsupported task: $TASK"
  exit 1
fi

# Print selected settings
echo "Selected Settings:"
echo "TASK: $TASK"
echo "LAYER: $LAYER"
echo "LLM: $LLM"
echo "SEED: $SEED"
echo "DATA_SIZE: $DATA_SIZE"
echo "DEVICE: $DEVICE"
echo "ALPHA: $ALPHA"
echo "STEER: $STEER"
echo "METHOD: $METHOD"
echo "TOPK_MEAN: $TOPK_MEAN"
echo "TOPK_CNT: $TOPK_CNT"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "SOURCE: $SOURCE"
echo "TARGET: $TARGET"
echo "MEAN_TYPE: $MEAN_TYPE"
echo "STEER_TYPE: $STEER_TYPE"
echo "DEBUG: $DEBUG"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "DATASET_PATH: $DATASET_PATH"
echo "PROMPT_PATH: $PROMPT_PATH"
echo "ENV_PATH: $ENV_PATH"

# Execute Python script with parsed arguments
/home/ckqsudo/miniconda3/envs/SAE/bin/python main_control_LLM.py --task $TASK --layer $LAYER --LLM $LLM --seed $SEED --data_size $DATA_SIZE --device $DEVICE --alpha $ALPHA --steer $STEER --method $METHOD --topk_mean $TOPK_MEAN --topk_cnt $TOPK_CNT --batch_size $BATCH_SIZE --source $SOURCE --target $TARGET --mean_type $MEAN_TYPE --steer_type $STEER_TYPE --debug $DEBUG --output_dir $OUTPUT_DIR --dataset_path $DATASET_PATH --prompt_path $PROMPT_PATH --env_path $ENV_PATH
