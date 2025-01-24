import os
import transformers
import torch
import torch.nn as nn
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
# from utils import *
# utils中有基础模型位置 需要手动改
from datetime import datetime
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def get_current_datetime_string():
    now = datetime.now()
    # Format the date and time as a string
    datetime_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    return datetime_string
def pre_process():
    from datasets import load_dataset
    
    dataset_path = "/home/ckqsudo/code2024/0dataset/baseline-acl/data/debate/StanceSentences"
    dataset = load_dataset(dataset_path)
    dataset=dataset.shuffle(seed=42)
    return dataset
from datasets import Dataset
dataset=pre_process()
def filter_support(examples):
    return examples['label'] != 'support'
dataset=pre_process()

train_dataset = dataset['train'].filter(filter_support).shuffle(seed=42)
validation_dataset = dataset['validation'].filter(filter_support).shuffle(seed=42)
tokenizer = AutoTokenizer.from_pretrained("/home/ckqsudo/code2024/0models/gpt-2-openai/gpt-2-openai")
tokenizer.pad_token = '<|pad|>'
def getdata(tokenizer,data):

    # data_list = data["text"]
    # data = Dataset.from_list(data_list[:])
    data = data.map(lambda x: tokenizer(x['text'],
                            truncation=True,
                            max_length=300,
                            padding=True,
                            return_tensors='pt',), batched=True)
    print(data[0],data[1])
    return data

data = getdata(tokenizer,train_dataset)
data_test = getdata(tokenizer,validation_dataset)#前500个
def get_parameters():

    time = get_current_datetime_string()
    # 模型输出位置(checkpoint)
    output_dir = os.environ.get('output_dir',f'./')
    # 测试集位置 只取前500
    path_test = os.environ.get('path_test', '/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/src/sft_test.json')
    # 训练集位置 只取前500
    path_train = os.environ.get('path_train', '/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/src/sft_train.json') 
    # 模型位置
    model_name = os.environ.get('model_name','/home/ckqsudo/code2024/0models/gpt-2-openai/gpt-2-openai')
    # 是否lora训练
    lora = bool(os.environ.get('lora', False))
    # lora rank
    lora_r = int(os.environ.get('lora_r', 32))
    num_epochs = int(os.environ.get('epoch', 5))
    # 优化器
    optim = os.environ.get('optim', "adamw_torch")
    learning_rate = float(os.environ.get('lr', 1e-5))
    # bs
    micro_batch_size = int(os.environ.get('micro_batch_size', 20))
    # 存多少个checkpoint
    save_total_limit = int(os.environ.get('save_total_limit', 10))
    # 训练全过程中eval多少次
    neval = int(os.environ.get('neval', 80))
    # wandb任务名
    wandb_run_name = f'{time}_lora_weight{str(bool(lora))}_loraR{lora_r}_optim_{optim}_epoch{num_epochs}_lr{learning_rate}'
    output_dir = output_dir + wandb_run_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return path_train, path_test, num_epochs, optim,learning_rate,lora,model_name,wandb_run_name,time,output_dir,micro_batch_size,save_total_limit,neval,lora_r

path_train, path_test, num_epochs, optim,learning_rate,lora,model_name,wandb_run_name,\
                                time,output_dir,micro_batch_size,save_total_limit,neval,lora_r = get_parameters()
# 限制显卡使用时设置
# os.environ["CUDA_VISIBLE_DEVICES"]="0" 
# 去除一个harmless warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 硬性规定 gradient_accumulation_steps
# gradient_accumulation_steps = 1

# 导入
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    load_in_8bit=False, 
    device_map='auto')
# 包装
if lora:
    config = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
# 数据集


group_by_length = True
ddp = True
use_wandb = False

# 使save_steps是eval_steps的整数倍 同时也设定了steps数目
nstep = num_epochs*len(data)/micro_batch_size
eval_steps = int(nstep/neval)
save_steps = (int(nstep/save_total_limit)//eval_steps)*eval_steps

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
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name = wandb_run_name if use_wandb else None,
    )


trainer = transformers.Trainer(
    model=model, 
    train_dataset=data,
    eval_dataset=data_test,
    args=args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
# 不保存最后模型 只存前面checkpoint而已


# 保存微调后的模型
model.save_pretrained("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/debate_test/debate_lora_model/lora_gpt2_small")

# 9. 加载微调后的模型进行推理生成
fine_tuned_model = GPT2LMHeadModel.from_pretrained("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/debate_test/debate_lora_model/lora_gpt2_small")
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