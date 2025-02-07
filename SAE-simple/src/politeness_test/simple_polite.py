from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import os

# Load the pre-trained model and tokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/style_transfer/politeness-corpus")

# Configure LoRA:
# LoRA Configuration
total_samples = len(dataset['train'])
dataset_val = dataset['train'].select(range(total_samples//2, total_samples - total_samples//4))
dataset_train = dataset['train'].select(range(total_samples//2))

# tokenizer.pad_token =tokenizer.eos_token
tokenizer.pad_token='[PAD]'
# config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,  # Causal Language Model Task
#     r=128,  # Low-rank parameter
#     lora_alpha=128,
#     lora_dropout=0.1,
#     # target_modules=["q_proj", "v_proj"]  # Inject LoRA in specific layers
# )
# model = get_peft_model(model, config)
# Fine-Tune the Model:
from transformers import Trainer, TrainingArguments
# Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples["text"], truncation=True, padding=True)
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=200)
tokenized_dataset_train = dataset_train.map(tokenize_function, batched=True)
tokenized_dataset_val = dataset_val.map(tokenize_function, batched=True)
# Add labels column (using input_ids as labels for causal LM)
# Add 'labels' column to both train and validation datasets

def filter_support(example):
    return example['label'] == 2
    # return example['label'] == 0


# 过滤训练集和验证集
tokenized_dataset_train = tokenized_dataset_train.filter(filter_support)
tokenized_dataset_val = tokenized_dataset_val.filter(filter_support)
print(tokenized_dataset_train)
print(tokenized_dataset_val)
# 
tokenized_dataset_train = tokenized_dataset_train.remove_columns(["label"])
tokenized_dataset_val = tokenized_dataset_val.remove_columns(["label"])
# label名称是关键
tokenized_dataset_train = tokenized_dataset_train.add_column("labels", tokenized_dataset_train["input_ids"])
tokenized_dataset_val = tokenized_dataset_val.add_column("labels", tokenized_dataset_val["input_ids"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results/polite/",
    evaluation_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=16,
    num_train_epochs=2
)
print(len(tokenized_dataset_train),len(tokenized_dataset_val))
# Define Trainer and start fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_val
)
trainer.train()

save_path="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/politeness_test/polite_tune_ckpt/fulltune_gpt2_small_checkpoint_2polite"
# 保存微调后的模型
model.save_pretrained(save_path)
print(save_path)