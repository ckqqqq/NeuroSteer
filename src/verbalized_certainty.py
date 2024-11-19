from config_file import certainty_dict,open_source_dict

import json  # 导入json库，用于读取本地JSON文件
import time  # 导入time模块，用于控制时间间隔
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条
import math  # 导入数学库，用于数学运算
import pandas as pd  # 导入pandas库，用于数据处理和分析
from pathlib import Path  # 导入Path类，用于操作文件路径
from openai import OpenAI  # 导入OpenAI库，用于调用本地部署的模型API
from openai import OpenAI as OpenAI_Official  # 导入官方的OpenAI库，用于调用GPT-4模型
import os
from dotenv import load_dotenv
from model_factory import get_model_and_tokenizer
import torch

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')

# 配置本地模型的API连接
client = OpenAI(
    base_url='http://10.110.147.66:11434/v1',  # 设置本地API的URL
    api_key='ollama'  # 这里必须设置api_key，但不会实际使用
)

# 配置官方OpenAI API（用于GPT-4调用）
official_openai_client = OpenAI_Official(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)  


# 调用GPT-4模型获取信心等级评估
def get_gpt4_confidence_response(prompt):
    # 使用官方OpenAI API调用GPT-4
    response = official_openai_client.chat.completions.create(
        model="gpt-4",  # 指定GPT-4模型
        messages=[{"role": "user", "content": prompt}]  # 用户输入的内容
    )
    return response.choices[0].message.content.strip()  # 提取并返回GPT-4的输出文本


# model = AutoModelForCausalLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
def get_open_source_mode_response(prompt,model_id):
    model,tokenizer=get_model_and_tokenizer(model_id=model_id)
    # 对问题进行 tokenize
    prompt_encoding = tokenizer(prompt, return_tensors="pt")
    prompt_ids = prompt_encoding.input_ids

    # 生成文本
    generated_text = tokenizer.decode(prompt_ids[0])
    for _ in range(2):  # 生成 50 个 token
        detailed_output = model(prompt_ids)
        logits = detailed_output.logits
        next_token_id = torch.argmax(logits[0, -1, :]).item() # 选择概率最高的 token
        next_token_text = tokenizer.decode(next_token_id)
        generated_text += next_token_text
        
        # 更新 prompt_ids 以继续生成
        prompt_ids = torch.cat([prompt_ids, torch.tensor([[next_token_id]])], dim=-1)
        print(next_token_text)
    return generated_text

def generate_verbalized_certainty(previous_prompt:str, previous_response:str, model_id:str, temp: float):
    """
    生成口头化的确定性描述和确定性值。
    参数:
        prompt (str): 提示信息。
        choices (dict): 包含选项的字典。
        response_text (str): 之前生成的答案.
        model_id(str)
        temp（double）温度
    """
    
    # 构建一个提示，要求模型分析其答案的确定性
    confidence_prompt = f"""  
    一个语言模型被要求完成下面的任务: {previous_prompt} 
    
    他的答案是： {previous_response}.

    Analyse its answer given other options. How certain are you about model's answer?

    1. very uncertain
    
    2. not certain
   
    3. somewhat certain
    
    4. moderately certain
    
    5. fairly certain
    
    6. very certain   
     
    please anwser with number from 1 to 6
    """
    # print(confidence_response_text)
    # 初始化置信度值为 None
    confidence_value = None
    # 遍历字典，查找与模型回答相匹配的口头化确定性描述，并获取其数值
            
    # local_confidence_text = get_local_model_response(engine_name, confidence_prompt).choices[0].message.content.strip()  # 获取本地模型信心等级文本

    # 使用GPT-4模型获取口头信心等级
    gpt4_confidence_text = get_gpt4_confidence_response(confidence_prompt)  # 获取GPT-4模型的信心等级文本
    print("GPT4给的置信度",gpt4_confidence_text)
    
    opensourced_model_confidence_text=get_open_source_mode_response(confidence_prompt,model_id=model_id)

    # 定义分数字典，将口头信心等级文本映射为数值

    print(open_source_dict[model_id]["model_name"],"给的置信度",opensourced_model_confidence_text)
    

    # # 将本地模型的信心等级文本映射为数值
    # local_confidence_value = next((v for k, v in score_dict.items() if k.lower() in local_confidence_text.lower()), None)
    # # 将GPT-4模型的信心等级文本映射为数值
    # gpt4_confidence_value = next((v for k, v in score_dict.items() if k.lower() in gpt4_confidence_text.lower()), None)

    # return confidence_response_text, confidence_value