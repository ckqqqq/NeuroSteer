import os
import random
import transformers
import torch
import torch.nn as nn
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
# from utils import *
# utils中有基础模型位置 需要手动改
from datasets import load_dataset

import os
# 设置路径和其他参数
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
def get_parameters():

    time = "???"
    # 模型输出位置(checkpoint)
    output_dir = os.environ.get('output_dir',f'./')
    # 测试集位置 只取前500
    # 模型位置
    model_name = os.environ.get('model_name','')
    # 是否lora训练
    lora = bool(os.environ.get('lora', False))
    # lora rank
    num_epochs = int(os.environ.get('epoch', 1))
    # 优化器
    optim = os.environ.get('optim', "adamw_torch")
    learning_rate = float(os.environ.get('lr', 1e-5))
    # bs
    micro_batch_size = int(os.environ.get('micro_batch_size', 20))
    # 存多少个checkpoint
    save_total_limit = int(os.environ.get('save_total_limit', 1))
    # 训练全过程中eval多少次
    neval = int(os.environ.get('neval', 3))
    # wandb任务名

    return None, None, num_epochs, optim,learning_rate,lora,model_name,None,time,output_dir,micro_batch_size,save_total_limit,neval,None

_, _, num_epochs, optim,learning_rate,lora,_,wandb_run_name,\
                                time,output_dir,micro_batch_size,save_total_limit,neval,_ = get_parameters()
# 限制显卡使用时设置
os.environ["CUDA_VISIBLE_DEVICES"]="2" 
# 去除一个harmless warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 硬性规定 gradient_accumulation_steps
# gradient_accumulation_steps = 1

# 导入
model = AutoModelForCausalLM.from_pretrained(
    "/home/ckqsudo/code2024/0models/gpt-2-openai/gpt-2-openai", 
    load_in_8bit=False, 
    device_map='auto')
# 包装
lora_r=128
lora_alpha=128
if lora:
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
# 数据集
wandb_run_name = f'{time}_lora_weight{str(bool(lora))}_loraR{lora_r}_optim_{optim}_epoch{num_epochs}_lr{learning_rate}'
output_dir = output_dir + wandb_run_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
def pre_process():
    dataset_path = "/home/ckqsudo/code2024/0dataset/baseline-acl/data/debate/StanceSentences"
    dataset = load_dataset(dataset_path)
    dataset=dataset.shuffle(seed=42)
    return dataset
tokenizer = AutoTokenizer.from_pretrained("/home/ckqsudo/code2024/0models/gpt-2-openai/gpt-2-openai")
tokenizer.pad_token = '<|pad|>'
import datasets
def text_to_token_ids(data):
    # data_list = list(map())
    # data = Dataset.from_list(data_list[:k])
    
    # data=data.map(lambda x: x['text']+"'<|endoftext|>'")
    # print(data)
    # data = data['text']
    # data=
    # print(data[:100])
    tokenized_output=tokenizer(data['text'],truncation=True,
                            max_length=500,
                            padding=True,
                            return_tensors='pt',)
    # 将 input_ids 添加到数据集中
    # data = data.add_column("input_ids", tokenized_output["input_ids"].tolist())
    print(tokenized_output)
    return tokenized_output
    print(text_ids)
    """
        data = data.map(lambda x: tokenizer(x['ins'],
                            truncation=True,
                            max_length=300,
                            padding=True,
                            return_tensors='pt',), batched=True)

    Returns:
        _type_: _description_
    """

# 加载数据集
# dataset = load_dataset(dataset_path)
# dataset=dataset.shuffle(seed=42)
def filter_support(examples):
    return examples['label'] != 'support'
dataset=pre_process()

train_dataset = dataset['train'].filter(filter_support).shuffle(seed=42)
validation_dataset = dataset['validation'].filter(filter_support).shuffle(seed=42)

train_token_ids= text_to_token_ids(train_dataset)
validation_token_ids = text_to_token_ids(validation_dataset)
# data_test,data_test_list = getdata(tokenizer,path_test,500)#前500个

group_by_length = False
ddp = True
use_wandb = True

# 使save_steps是eval_steps的整数倍 同时也设定了steps数目
nstep = num_epochs*len(train_token_ids)/micro_batch_size
eval_steps = int(nstep/neval)
save_steps = 8

# 常规参数以及lr的cosine规划 不使用wandb在此处改
args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        # gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=0,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type='linear',
        fp16=True,
        logging_steps=20,
        optim=optim,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        auto_find_batch_size=True,
        output_dir=output_dir,
        # save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if ddp else None,
        # group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name = wandb_run_name if use_wandb else None,
    )


trainer = transformers.Trainer(
    model=model, 
    train_dataset=train_token_ids,
    eval_dataset=validation_token_ids,
    args=args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()