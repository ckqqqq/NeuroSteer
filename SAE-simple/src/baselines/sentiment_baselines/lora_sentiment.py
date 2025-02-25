from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import os
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import json

# Load the pre-trained model and tokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt_path = "/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k"

# tokenizer.pad_token =tokenizer.eos_token
tokenizer.pad_token='[PAD]'
DATASIZE = 2400



def load_and_prepare_triple_dataset(dataset_path: str, seed: int, dataset_name:str):
    assert dataset_name in ["sst5","multiclass","polite"],"错误数据集名字"
    if dataset_name in ["sst5"]:
        neu_label=2 # 中性情感对应的label
        assert "sst5" in dataset_path
    elif  dataset_name in ["polite","multiclass"]:
        neu_label=1
    dataset = load_dataset(dataset_path)
    dataset["train"] = dataset['train'].shuffle(seed=seed)
    
    neg_train_set = dataset['train'].filter(lambda example: example['label'] < neu_label)
    pos_train_set = dataset['train'].filter(lambda example: example['label'] == neu_label)
    neu_train_set = dataset['train'].filter(lambda example: example['label'] > neu_label)

    print(f"检查数据量 Selected {len(neg_train_set)} negative, {len(pos_train_set)} positive, and {len(neu_train_set)} neutral samples")
    assert 'validation' in dataset.keys() and "test" in dataset.keys(),"数据集不兼容"
    
    val_set=dataset['validation']
    test_set=dataset["test"]
    return neg_train_set, pos_train_set, neu_train_set,val_set,test_set

def load_and_prepare_sentiment_prompts(prompt_path:str):
    prompt_files = {"neg": "negative_prompts.jsonl", "pos": "positive_prompts.jsonl","neu":"neutral_prompts.jsonl"}
    prompts= load_dataset(prompt_path,data_files=prompt_files)
    print(prompts)
    return prompts
    
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=100)
def filter_neg(example):
    return example['label'] == 0 or example['label'] == 1
def filter_pos(example):
    return example['label'] == 3 or example['label'] == 4

neg_train_set, pos_train_set, neu_train_set,val_set,test_set = load_and_prepare_triple_dataset(
    dataset_path="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5", 
    seed=42, 
    dataset_name="sst5",
)

neg_dataset_train = neg_train_set.select(range(DATASIZE)).map(tokenize_function, batched=True)
neg_dataset_eval = val_set.filter(filter_neg).map(tokenize_function, batched=True)

neg_dataset_train = neg_dataset_train.remove_columns(["label"])
neg_dataset_eval = neg_dataset_eval.remove_columns(["label"])
# label名称是关键
neg_dataset_train = neg_dataset_train.add_column("labels", neg_dataset_train["input_ids"])
neg_dataset_eval = neg_dataset_eval.add_column("labels", neg_dataset_eval["input_ids"])

train_time_start=time.time()
# Define training arguments
training_args = TrainingArguments(
    output_dir="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/sentiment_lora_baseline/lora_ckpt",
    eval_strategy="epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    num_train_epochs=2
)

# Define Trainer and start fine-tuning
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Causal Language Model Task
    r=128,  # Low-rank parameter
    lora_alpha=128,
    lora_dropout=0.1,
    # target_modules=["q_proj", "v_proj"]  # Inject LoRA in specific layers
)
model = get_peft_model(model, config)
# Fine-Tune the Model:
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=neg_dataset_train,
    eval_dataset=neg_dataset_eval
)
trainer.train()
train_time_end=time.time()
save_path=f"/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/sentiment_lora_baseline/lora_ckpt/sentiment_pos2neg_lora_baseline_{DATASIZE}"
# 保存微调后的模型
model.save_pretrained(save_path)
print(save_path)
print("训练时间开销",train_time_end-train_time_start)


prompts = load_and_prepare_sentiment_prompts(prompt_path=prompt_path)
pos_prompts = prompts['pos'].select(range(500))

gpt2_path = "/home/ckqsudo/code2024/0models/gpt-2-openai/gpt-2-openai"
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
tokenizer.pad_token='[PAD]'
fine_tuned_model = GPT2LMHeadModel.from_pretrained(f"/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/sentiment_lora_baseline/lora_ckpt/sentiment_pos2neg_lora_baseline_{DATASIZE}")
fine_tuned_model.to("cuda")

lora_generations = []
for text in tqdm(pos_prompts):
    inputs = tokenizer(text['prompt']['text'], return_tensors="pt").to(fine_tuned_model.device)
    generations = []
    for i in range(2):
        generated_ids = fine_tuned_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],  # Pass the attention_mask to generation
            max_length=200,
            num_return_sequences=1,
            do_sample=True,
            top_k=100,
            top_p=0.1,
            temperature=1.0
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generations.append({'text': generated_text})
    text['generations'] = generations
    lora_generations.append(text)

with open(f'/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/sentiment_lora_baseline/lora_pos2neg_baseline_generations_{DATASIZE}.jsonl','w') as f:
    for item in lora_generations:
        f.write(json.dumps(item)+'\n')
