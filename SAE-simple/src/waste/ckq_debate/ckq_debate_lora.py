from numpy import pad
import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, EarlyStoppingCallback, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# 设置路径和其他参数
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

gpt2_path = "/home/ckqsudo/code2024/0models/gpt-2-openai/gpt-2-openai"
dataset_path = "/home/ckqsudo/code2024/0dataset/baseline-acl/data/debate/StanceSentences"

# 加载数据集
dataset = load_dataset(dataset_path)
dataset=dataset.shuffle(seed=42)

# 2. 加载 tokenizer 和模型
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
tokenizer.pad_token = '<|pad|>'
print(tokenizer)
# tokenizer.pad_token_id = tokenizer.eos_token_id
model = GPT2LMHeadModel.from_pretrained(gpt2_path)

# 3. 使用 LoRA 配置进行微调
lora_config = LoraConfig(
    r=128,                  # LoRA rank, 控制低秩矩阵的维度
    lora_alpha=128,        # LoRA alpha，控制低秩矩阵与原模型权重的融合比例
    target_modules="all-linear",  # 目标模块指定要进行LoRA微调的模块
    lora_dropout=0.1,  # 增加dropout
    bias="none",          # 设置LoRA中的偏置选项
    task_type='CAUSAL_LM',  # 因为是生成任务
)

# 获取 LoRA 微调后的模型
model = get_peft_model(model, lora_config)

# 4. 加载数据集并过滤掉 'support' 的数据
def filter_support(examples):
    return examples['label'] != 'support'

train_dataset = dataset['train'].filter(filter_support).shuffle(seed=42)
validation_dataset = dataset['validation'].filter(filter_support).shuffle(seed=42)

# 5. 按照句子拆分 'text' 为前半句和后半句
def split_text(examples):
    texts = examples['text']
    inputs = []
    labels = []
    
    for text in texts:
        tokens = tokenizer.tokenize(text+'<|endoftext|>')
        input_tokens = tokens
        # label_tokens = tokens[1:]
        # label_tokens.append('<|endoftext|>')
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        # label_ids = tokenizer.convert_tokens_to_ids(label_tokens)?
        
        inputs.append(input_ids)
        # labels.append(label_ids)
    
    return {'input_ids': inputs}

train_dataset = train_dataset.map(split_text, batched=True)
validation_dataset = validation_dataset.map(split_text, batched=True)

# 6. Padding操作：对input_ids和labels进行padding，确保它们的长度一致
# def pad_data(examples):
#     # 对 input_ids 和 labels 进行padding，确保它们的长度一致
#     padded_input_ids = tokenizer.pad(
#         {'input_ids': examples['input_ids']}, 
#         padding=True, 
#         max_length=200,
#         return_tensors='pt'
#     )['input_ids']
    
    # padded_labels = tokenizer.pad(
    #     {'input_ids': examples['labels']}, 
    #     padding=True, 
    #     max_length=200, 
    #     return_tensors='pt'
    # )['input_ids']
    
    # Attention mask是根据input_ids来生成的，填充部分用0表示
    # attention_mask = (padded_input_ids != tokenizer.pad_token_id).long()
    
    # # 返回input_ids, labels和attention_mask
    # return {
    #     'input_ids': padded_input_ids.tolist(), 
    #     # 'labels': padded_labels.tolist(), 
    #     'attention_mask': attention_mask.tolist()
    # }

# train_dataset = train_dataset.map(p, batched=True)
# validation_dataset = validation_dataset.map(pad_data, batched=True)

# print(train_dataset[0])
train_dataset = train_dataset["input_ids"]
validation_dataset = validation_dataset["input_ids"]

# 8. 设置训练参数
training_args = TrainingArguments(
    output_dir="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/debate_test/debate_gpt2_ckpt",
    eval_strategy="epoch",
    learning_rate=5e-6,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.1,
    logging_dir="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/debate_test/debate_logs",
    logging_steps=200,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,
)
print("训练集",len(train_dataset),"验证集",len(validation_dataset))
# 传递模型和数据集给Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator
)

trainer.train()

# 保存微调后的模型
model.save_pretrained("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/debate_test/debate_gpt2_ckpt/lora_gpt2_small_checkpoint")

# 9. 加载微调后的模型进行推理生成
fine_tuned_model = GPT2LMHeadModel.from_pretrained("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/debate_test/debate_gpt2_ckpt/lora_gpt2_small_checkpoint")
fine_tuned_model.to("cuda" if torch.cuda.is_available() else "cpu")

# 10. 推理生成
prompt = "Trump is about to take office as the President of the United States. What do you think of this new president?"
inputs = tokenizer(prompt, return_tensors="pt").to(fine_tuned_model.device)
attention_mask = (inputs['input_ids'] != tokenizer.pad_token_id).long()  # Manually create the attention mask

generated_ids = fine_tuned_model.generate(
    inputs['input_ids'],
    attention_mask=attention_mask,  # Pass the attention_mask to generation
    max_length=200,
    do_sample=True,
    top_k=100,
    top_p=0.6,
    temperature=0.4,
    eos_token_id=tokenizer.eos_token_id,  # 设置生成结束的token
    early_stopping=True  # 一旦生成eos_token就停止
)

# 解码生成的文本
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated text: ", generated_text)
# 屎山