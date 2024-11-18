'''
这是论文A代码的重构版本，我直接把他的代码整合到了一个文件中，同时使用GPT4o添加了注释
此代码不能直接使用，不过可以阅读代码来了解他的思路
'''

import math  # 导入数学模块math，用于数学运算
import time  # 导入时间模块time，用于控制时间间隔
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条
import torch  # 导入PyTorch库，用于深度学习模型的张量操作
import torch.nn.functional as F  # 导入PyTorch中的函数式API，用于计算概率
import pandas as pd  # 导入pandas库，用于数据处理和分析
from pathlib import Path  # 导入Path类，用于操作文件路径
import os  # 导入os模块，用于文件系统操作
from datasets import load_dataset  # 从datasets库导入load_dataset，用于加载数据集
from transformers import AutoTokenizer, MistralForCausalLM  # 导入transformers库中的Tokenizer和模型类
from openai import OpenAI  # 导入OpenAI库，用于调用API
import argparse  # 导入argparse库，用于命令行参数解析



# 配置常量
open_source_models = ['phi', 'zephyr']  # 开源模型名称列表
open_source_dict = {"zephyr": "HuggingFaceH4/zephyr-7b-beta"}  # 开源模型名称到模型路径的映射
datasets = ['commonsense_qa', 'openbookqa', 'qasc', 'riddle_sense', 'ai2_arc']  # 数据集名称列表
api_key = ""  # API密钥（需要用户设置）

# 定义ModelFactory类，用于管理模型加载
class ModelFactory:
    def __init__(self):
        self.supported_models = {"HuggingFaceH4/zephyr-7b-beta"}  # 支持的模型名称集合

    def load(self, model_name):
        # 如果模型名称不在支持的集合中，则抛出异常
        if model_name not in self.supported_models:
            raise NotImplemented(f"Model not supported: {model_name}")
        # 加载模型的分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 加载因果语言模型
        model = MistralForCausalLM.from_pretrained(model_name)
        return tokenizer, model  # 返回分词器和模型对象

# 辅助函数：调用OpenAI API获取响应
def get_open_ai_response(engine_name, prompt, temp):
    client = OpenAI(api_key=api_key)  # 使用API密钥创建OpenAI客户端
    return client.chat.completions.create(
        model=engine_name,  # 指定模型
        messages=[{"role": "user", "content": prompt}],  # 传入用户输入的提示信息
        logprobs=True,  # 请求返回每个标记的log概率
        top_logprobs=5,  # 请求返回前5个最高的log概率
        temperature=temp  # 设置生成的随机性
    )

# 获取指定引擎名称的模型和分词器
def get_model_and_tokenizer(engine_name):
    return ModelFactory().load(open_source_dict[engine_name])  # 调用ModelFactory加载模型和分词器

# 将数据保存到Excel文件，包含错误处理
def save_data(dataset, model, data_dict):
    df = pd.DataFrame(data_dict)  # 创建包含数据的DataFrame
    directory = f"data/{dataset}/"  # 设置数据保存的文件夹路径
    filename = f"{model}.xlsx"  # 设置文件名称
    Path(directory).mkdir(parents=True, exist_ok=True)  # 如果目录不存在则创建
    file_path = os.path.join(directory, filename)  # 合并目录和文件名称为完整路径
    try:
        df.to_excel(file_path, index=False)  # 保存DataFrame到Excel文件
    except Exception as e:  # 捕获任何文件保存时的异常
        print(f"File save error: {e}")  # 打印错误信息

# 根据响应提示生成文本化的信心等级
def generate_verbalized_certainty(question, choices_text, response_text, engine_name, temp):
    # 设定信心等级的分数字典
    score_dict = {
        "very certain": 1.0, "fairly certain": 0.8, "moderately certain": 0.6,
        "somewhat certain": 0.4, "not certain": 0.2, "very uncertain": 0
    }
    # 构建模型评估信心等级的提示文本
    confidence_prompt = (
        f"A Language model was asked: {question}\nOptions were: {choices_text}.\n"
        f"Model's answer was: {response_text}.\nAnalyze answer confidence level:\n"
        f"a. very certain\nb. fairly certain\nc. moderately certain\nd. somewhat certain\n"
        f"e. not certain\nf. very uncertain"
    )
    # 调用API获取模型的信心等级回答
    confidence_response_text = get_open_ai_response(engine_name, confidence_prompt, temp).choices[0].message.content.strip()
    # 遍历信心分数字典，将模型返回的文本映射为数值
    confidence_value = next((v for k, v in score_dict.items() if k.lower() in confidence_response_text.lower()), None)
    return confidence_response_text, confidence_value  # 返回信心等级文本和数值

# 调用闭源模型并获取响应概率
def closed_source_models(prompt, engine_name, temp):
    response = get_open_ai_response(engine_name, prompt, temp)  # 调用API获取模型回答
    response_text = response.choices[0].message.content.strip()  # 提取回答文本
    top_probs = response.choices[0].logprobs.content[0].top_logprobs  # 获取回答的log概率
    valid_choices_keys = ["a", "b", "c", "d", "e", "f", "g", "h"]  # 定义有效选项的键
    tokens_list = {p.token: p.logprob for p in top_probs}  # 将词和其log概率存入字典
    # 过滤并计算每个有效选项的概率
    filtered_token_probs = {k: math.exp(v) for k, v in tokens_list.items() if k.lower().strip() in valid_choices_keys}
    sum_token_probs = sum(filtered_token_probs.values())  # 计算所有选项的概率总和
    # 获取最佳选项的概率
    response_prob = max(filtered_token_probs.values()) / sum_token_probs if filtered_token_probs else None
    return response_text, response_prob  # 返回模型回答文本和概率

# 调用开源模型并计算每个选项的概率
def open_source_models(prompt, engine_name, choices_list):
    tokenizer, model = get_model_and_tokenizer(engine_name)  # 获取模型及其分词器
    # 将选项转化为token ID并存储
    choices_ids = [(choice, tokenizer.encode(choice)) for choice in choices_list]
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids  # 将提示转化为ID张量
    probabilities = F.softmax(model(prompt_ids).logits[0], dim=-1)[-1]  # 计算模型输出的softmax概率
    # 查找概率最高的选项及其概率
    best_choice, best_prob = max(
        ((label, torch.sum(torch.gather(probabilities, 0, torch.tensor(ids)).float()).item()) for label, ids in choices_ids),
        key=lambda x: x[1]
    )
    return best_choice, best_prob  # 返回最佳选项和最高概率

# 生成模型回答并计算内部信心分数
def generate_internal_confidence(prompt, choices, engine_name, temp):
    # 判断模型类型并调用相应函数获取内部信心分数
    return closed_source_models(prompt, engine_name, temp) if engine_name not in open_source_models else open_source_models(prompt, engine_name, choices['label'])

# 生成模型回答和口头信心等级
def generate_response_and_ask_confidence(question, choices, engine_name, temp):
    # 构造选项文本
    choices_text = '\n'.join(f"{label}. {text}" for label, text in zip(choices['label'], choices['text']))
    prompt = f"{question}\n{choices_text}\nAnswer:"  # 构建提示语
    # 获取模型回答和内部信心分数
    response_text, response_prob = generate_internal_confidence(prompt, choices, engine_name, temp)
    # 生成模型口头信心等级
    confidence_response_text, confidence_value = generate_verbalized_certainty(question, choices_text, response_text, engine_name, temp)
    return response_text, response_prob, confidence_response_text, confidence_value  # 返回模型回答、内部信心分数、口头信心等级及其数值

# 加载数据集，处理并保存结果
def get_and_save_confidence(dataset, model):
    # 加载数据集，特殊情况按不同子集处理
    dataset = load_dataset(dataset, 'ARC-Challenge' if dataset == 'ai2_arc' else None).shuffle(seed=80)
    subset_size, temperature = (926 if dataset == 'qasc' else 1000), 0.2  # 设置子集大小和温度
    split = 'train' if dataset in ['openbookqa', 'ai2_arc'] else 'validation'  # 数据集划分

    # 初始化结果存储字典
    results = {"questions": [], "responses": [], "response_probs": [], "confidence_text": [], "confidence_values": [], "answers": []}

    # 遍历子集中的每个示例并获取模型回答
    for example in tqdm(dataset[split][:subset_size]):
        time.sleep(4)  # 暂停4秒避免API调用过频
        question, choices, answer = example['question'], example['choices'], example['answerKey']
        # 调用生成函数获取模型的回答及信心分数
        response_text, response_prob, confidence_text, confidence_value = generate_response_and_ask_confidence(question, choices, model, temperature)
        # 将结果添加到字典
        results["questions"].append(question)
        results["responses"].append(response_text)
        results["response_probs"].append(response_prob)
        results["confidence_text"].append(confidence_text)
        results["confidence_values"].append(confidence_value)
        results["answers"].append(answer)

    # 将结果保存到Excel文件
    save_data(dataset, model, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence_probability_alignment")
    parser.add_argument('--model', type=str, required=True, choices=['gpt4', 'text-davinci-03',
                                                                     'text-davinci-02', 'text-davinci-01', 'phi',
                                                                     'zephyr'],
                        help='choose a model')
    parser.add_argument('--dataset', type=str, required=True, choices=datasets, help='dataset names')

    model = parser.parse_args().model
    dataset = parser.parse_args().dataset
    get_and_save_confidence(dataset, model)