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
# Load a dataset
data_files={"train":"stance_sentences_train.jsonl","validation":"stance_sentences_validation.jsonl"}
dataset = load_dataset("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/data/stance/StanceSentences")
# Configure LoRA:
# LoRA Configuration
dataset["train"]=dataset["train"].shuffle(seed=42)
print(dataset["train"][579])
# tokenizer.pad_token =tokenizer.eos_token
tokenizer.pad_token='[PAD]'
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Causal Language Model Task
    r=128,  # Low-rank parameter
    lora_alpha=128,
    lora_dropout=0.1,
    # target_modules=["q_proj", "v_proj"]  # Inject LoRA in specific layers
)
model = get_peft_model(model, config)
# Fine-Tune the Model:
from transformers import Trainer, TrainingArguments
# Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples["text"], truncation=True, padding=True)
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=200)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(tokenized_dataset)
# Add labels column (using input_ids as labels for causal LM)
# Add 'labels' column to both train and validation datasets

def filter_support(example):
    return example['label'] == 'oppose'
    # return example['label'] == 'support'


# 过滤训练集和验证集
tokenized_dataset['train'] = tokenized_dataset['train'].filter(filter_support)
tokenized_dataset['validation'] = tokenized_dataset['validation'].filter(filter_support)
# 
tokenized_dataset['train'] = tokenized_dataset['train'].remove_columns(["label"])
tokenized_dataset['validation'] = tokenized_dataset['validation'].remove_columns(["label"])
# label名称是关键
tokenized_dataset['train'] = tokenized_dataset['train'].add_column("labels", tokenized_dataset['train']["input_ids"])
tokenized_dataset['validation'] = tokenized_dataset['validation'].add_column("labels", tokenized_dataset['validation']["input_ids"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=16,
    num_train_epochs=2
)
print(len(tokenized_dataset["train"]),len(tokenized_dataset["validation"]))
# Define Trainer and start fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)
trainer.train()

save_path="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/debate_test/debate_gpt2_ckpt/lora_gpt2_small_checkpoint_ckq"
# 保存微调后的模型
model.save_pretrained(save_path)
print(save_path)