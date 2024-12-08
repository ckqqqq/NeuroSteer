"""所有的常见公用函数"""
from config_file import model_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import tiktoken
from config_file import model_dict 
import random
import json
import os

def shuffle_data(dataset,seed_number):
    """打乱列表数据集"""
    print("随机种子为",seed_number)
    random.seed(seed_number)  # 你可以使用任何整数作为种子
    random.shuffle(dataset)# 打乱列表元素
    return dataset

def save_dataset(dataset,task_id,msg):
    """缓存数据"""
    with open(f"cache/cache2_{task_id}_{msg}.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
def get_model_and_tokenizer(model_id):
    """以transformer获得开源模型的model和tokenizer"""
    if model_dict[model_id]["is_open"]==False:
        raise ValueError(f"没有这个开源模型{model_id}")
    tokenizer=AutoTokenizer.from_pretrained(model_dict[model_id]["model_path"],trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(model_dict[model_id]["model_path"],trust_remote_code=True)
    return model,tokenizer
        

def get_model_price(model_id,input,out,is_dollar=True):
    """估计gpt的api开销"""
    if model_dict[model_id]["is_open"]:
        return 0,0,0
    # Initialize the tokenizer for the GPT model
    tokenizer = tiktoken.encoding_for_model(model_dict[model_id])  

    # request and response
    request = str(input)
    response = str(out)

    # Tokenize 
    request_tokens = tokenizer.encode(request)
    response_tokens = tokenizer.encode(response)

    # Counting the total tokens for request and response separately
    input_tokens = len(request_tokens)
    output_tokens = len(response_tokens)

    # Actual costs per 1 million tokens
    cost_per_1M_input_tokens = 0.15  # $0.150 per 1M input tokens
    cost_per_1M_output_tokens = 0.60  # $0.600 per 1M output tokens
    money=0.0
    if is_dollar:
        money+=input_tokens/1000000*cost_per_1M_input_tokens+output_tokens/1000000*cost_per_1M_output_tokens
    return input_tokens,output_tokens,money

def save_file(data,filename):
    """保存文件"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
