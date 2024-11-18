'''
这是参考论文A实验重构后的代码，用于自动评估模型对特定问题的信心等级
此重构代码因为缺乏处理后的数据集格式，目前还没有运行过，需要后期debug
'''

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

# 设定数据保存的常量
dataset_file = "esconv_result_test.json"  # 本地JSON文件路径
model_name = "qwen2"  # 使用的本地模型名称

# 从本地JSON文件加载数据集
def load_local_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  # 以只读模式打开文件
        data = json.load(file)  # 读取JSON文件内容
    return data  # 返回数据集内容

# 调用本地API获取响应
def get_local_model_response(engine_name, prompt):
    # 使用配置好的API客户端创建请求
    response = client.chat.completions.create(
        model=engine_name,  # 指定模型名称
        messages=[{"role": "user", "content": prompt}],  # 用户输入的内容
        logprobs=True,  # 请求返回每个标记的log概率
        top_logprobs=5  # 请求返回前5个最高的log概率
    )
    return response  # 返回API响应

# 调用GPT-4模型获取信心等级评估
def get_gpt4_confidence_response(prompt):
    # 使用官方OpenAI API调用GPT-4
    response = official_openai_client.chat.completions.create(
        model="gpt-4",  # 指定GPT-4模型
        messages=[{"role": "user", "content": prompt}]  # 用户输入的内容
    )
    return response.choices[0].message.content.strip()  # 提取并返回GPT-4的输出文本

# 计算模型的回答及其概率
def get_response_with_confidence(prompt, engine_name):
    # 调用本地模型API获取响应
    response = get_local_model_response(engine_name, prompt)
    response_text = response.choices[0].message.content.strip()  # 提取模型的回答文本
    return response_text, None  # 返回模型回答文本和概率（目前概率为None）
    # top_probs = response.choices[0].logprobs.content[0].top_logprobs  # 获取回答的log概率
    # valid_choices_keys = ["a", "b", "c", "d", "e", "f", "g", "h"]  # 定义有效选项的键
    # tokens_list = {p.token: p.logprob for p in top_probs}  # 将词和其log概率存入字典
    # # 过滤并计算每个有效选项的概率
    # filtered_token_probs = {k: math.exp(v) for k, v in tokens_list.items() if k.lower().strip() in valid_choices_keys}
    # sum_token_probs = sum(filtered_token_probs.values())  # 计算所有选项的概率总和
    # # 获取最佳选项的概率
    # response_prob = max(filtered_token_probs.values()) / sum_token_probs if filtered_token_probs else None
    # return response_text, response_prob  # 返回模型回答文本和概率

# 将数据保存到Excel文件
def save_data(model, data_dict):
    df = pd.DataFrame(data_dict)  # 创建包含数据的DataFrame
    directory = f"data/"  # 设置数据保存的文件夹路径
    filename = f"{model}.xlsx"  # 设置文件名称
    Path(directory).mkdir(parents=True, exist_ok=True)  # 如果目录不存在则创建
    file_path = os.path.join(directory, filename)  # 合并目录和文件名称为完整路径
    try:
        df.to_excel(file_path, index=False)  # 保存DataFrame到Excel文件
    except Exception as e:  # 捕获任何文件保存时的异常
        print(f"File save error: {e}")  # 打印错误信息

# 生成模型的回答和口头信心等级
def generate_response_and_ask_confidence(question, choices, engine_name):
    # 构造选项文本
    choices_text = '\n'.join(f"{label}. {text}" for label, text in zip(choices['label'], choices['text']))
    prompt = f"{question}\n{choices_text}\nAnswer:"  # 构建提示语

    # 使用本地模型获取回答和信心分数
    # response_text, response_prob = get_response_with_confidence(prompt, engine_name)
    response_text, _ = get_response_with_confidence(prompt, engine_name)

    # 本地模型口头信心等级
    confidence_prompt = (
        f"A Language model was asked: {question}\nOptions were: {choices_text}.\n"
        f"Model's answer was: {response_text}.\nAnalyze answer confidence level:\n"
        f"a. very certain\nb. fairly certain\nc. moderately certain\nd. somewhat certain\n"
        f"e. not certain\nf. very uncertain"
    )
    local_confidence_text = get_local_model_response(engine_name, confidence_prompt).choices[0].message.content.strip()  # 获取本地模型信心等级文本

    # 使用GPT-4模型获取口头信心等级
    gpt4_confidence_text = get_gpt4_confidence_response(confidence_prompt)  # 获取GPT-4模型的信心等级文本

    # 定义分数字典，将口头信心等级文本映射为数值
    score_dict = {
        "very certain": 1.0, "fairly certain": 0.8, "moderately certain": 0.6,
        "somewhat certain": 0.4, "not certain": 0.2, "very uncertain": 0
    }

    # 将本地模型的信心等级文本映射为数值
    local_confidence_value = next((v for k, v in score_dict.items() if k.lower() in local_confidence_text.lower()), None)
    # 将GPT-4模型的信心等级文本映射为数值
    gpt4_confidence_value = next((v for k, v in score_dict.items() if k.lower() in gpt4_confidence_text.lower()), None)

    # # 返回模型回答、内部信心分数、本地模型的信心等级及其数值、GPT-4的信心等级及其数值
    # return response_text, response_prob, local_confidence_text, local_confidence_value, gpt4_confidence_text, gpt4_confidence_value

    return response_text, None, local_confidence_text, local_confidence_value, gpt4_confidence_text, gpt4_confidence_value
    
# 加载本地数据集，处理并保存分析结果
def analyze_and_save_confidence():
    dataset = load_local_dataset(dataset_file)  # 从本地JSON文件加载数据集
    model = model_name  # 设置模型名称
    temperature = 0.2  # 设置生成的温度参数
    
    # 初始化结果存储字典
    results = {
        "questions": [], "responses": [], "local_confidence_text": [], "local_confidence_values": [],
        "gpt4_confidence_text": [], "gpt4_confidence_values": [],
        "answers": []
    }

    # 遍历数据集中的每个示例
    for example in tqdm(dataset):
        time.sleep(4)  # 暂停4秒，避免API调用过频
        question, choices, answer = example['question'], example['choices'], example['answerKey']
        # 调用生成函数获取模型的回答及信心分数
        response_text, response_prob, local_conf_text, local_conf_value, gpt4_conf_text, gpt4_conf_value = generate_response_and_ask_confidence(
            question, choices, model
        )
        # 将结果添加到字典
        results["questions"].append(question)
        results["responses"].append(response_text)
        results["local_confidence_text"].append(local_conf_text)
        results["local_confidence_values"].append(local_conf_value)
        results["gpt4_confidence_text"].append(gpt4_conf_text)
        results["gpt4_confidence_values"].append(gpt4_conf_value)
        results["answers"].append(answer)

    # 将结果保存到Excel文件
    save_data(model, results)

# 主程序入口
if __name__ == "__main__":
    analyze_and_save_confidence()  # 执行数据分析和结果保存
