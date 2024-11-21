
from numpy import double
from utils import get_model_and_tokenizer
# 定义一个函数，用于处理开源模型的响应
import torch
import torch.nn.functional as F
from config_file import model_dict,strategy_choice
from openai import OpenAI  # 导入OpenAI库，用于调用本地部署的模型API


import math
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')

def closed_source_models(prompt, model_id="gpt-4o-mini",temp=0.2):
    """
    闭源模型的置信度概率
    """
    client= OpenAI(api_key=OPENAI_API_KEY,base_url=OPENAI_API_BASE)
    response =client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": prompt}
        ],
        logprobs=True,# 计算logprobs，计算每个token的概率分布 
        top_logprobs=5,# 采样topk个token 
        temperature=temp # 采样温度
    )
    # 通过这种方式可以对logprob进行采样，从而得到topk的概率分布，然后进行计算
    # 使用 OpenAI API 获取模型对提示的回答
    # 获取回答的文本内容，并去除首尾空格
    response_text = response.choices[0].message.content.strip()
    # 获取回答的 logprobs 内容，并提取第一个元素的 top_logprobs
    print("openai 回复结果",response_text)
    top_probs = response.choices[0].logprobs.content[0].top_logprobs
    # 定义一个包含有效选项键的列表
    valid_choices_keys = strategy_choice["label"]
    print("有效的参考",valid_choices_keys)

    # 将 log 概率转换为概率，并过滤出有效选项的概率
    tokens_list = {p.token: p.logprob for p in top_probs}
    filtered_token_probs = {k: math.exp(v) for k, v in tokens_list.items() if (k.strip()).lower() in valid_choices_keys}
    # 初始化一个字典，用于存储最终的 token 概率
    token_final = {}
    # 遍历过滤后的 token 概率，选择每个键的最大值
    for k, v in filtered_token_probs.items():
        k_stripped = (k.strip()).lower()
        if k_stripped in token_final:
            if token_final[k_stripped] < v:
                token_final[k_stripped] = v
        else:
            token_final[k_stripped] = v

    # 计算所有 token 概率的总和
    sum_token_probs = sum(token_final.values())
    # 如果存在有效选项，则计算顶部选择概率与其他概率之和的比率
    if len(filtered_token_probs) > 0:
        # 找到概率最大的 token
        top_choice_key = max(token_final, key=token_final.get)
        # 计算响应概率
        response_prob = math.exp(token_final[top_choice_key]) / math.exp(sum_token_probs)
    else:
        response_prob = None

    return response_text, response_prob,None


def calculate_probability(logits,target_choice_labels,tokenizer):
    """计算概率的函数：输入模型的logits矩阵，基于目标字符对模型进行解码"""
    # 使用 softmax 函数计算 logits 的概率分布
    probabilities = F.softmax(logits[0], dim=-1)
    # 只关注最后一个 token 的概率分布
    last_token_probabilities = probabilities[-1]
    # 初始化当前最高概率标签和概率值
    choices_prob={}
    
    for choice_label in target_choice_labels:
        choice_id= tokenizer.encode(choice_label, add_special_tokens=False)
        ids = choice_id
        # 将 ids 转换为 tensor，并确保与 probabilities 在同一设备
        ids_tensor = torch.tensor(ids)
        label_probs = torch.gather(last_token_probabilities, dim=0, index=ids_tensor)
        # 对选项的概率求和
        label_probs = torch.sum(label_probs).item()
        choices_prob[choice_label] = label_probs
    return choices_prob

def open_source_models(prompt, model_id, choices):
    """计算开源模型的内部置信度概率"""
    # 获取模型和 tokenizer
    model, tokenizer = get_model_and_tokenizer(model_id)
    # 对问题进行 tokenize
    prompt_encoding = tokenizer(prompt, return_tensors="pt")
    print("输入prompt的长度",prompt_encoding.input_ids.shape)
    prompt_ids = prompt_encoding.input_ids
    # 使用模型获取详细的输出
    detailed_output = model(prompt_ids)
    # 目标答案空间
    target_choice_labels=[i for i in choices["label"]]
    for next_n_token in range(20):
        # 生成一个 token
        detailed_output = model(prompt_ids)
        logits=detailed_output.logits
        """logits 这个矩阵的形状是（batch_size，sequence_length，vocab_size）他的值为每个batch_size中每个token在词表中的得分"""
        """简单来说是([1, 579, 32064]),就是一个batch中 ，579个token的prompt中的下一个token, 在32064个单词表上的概率"""
        next_token_id = torch.argmax(logits[0, -1, :]).item() # 选择最后一个token
        if next_n_token==0:
            print("Logits:",logits.shape )# 打印logits 矩阵的大小
        next_token = tokenizer.decode(next_token_id)
        prompt_ids = torch.cat([prompt_ids, torch.tensor([[next_token_id]])], dim=-1)
        
        print(next_token)
        if next_token in target_choice_labels:
            """如果解码到选项字母，就输出所有字母的概率"""
            label_prob_dict=calculate_probability(logits,target_choice_labels,tokenizer)
            return next_token,label_prob_dict[next_token],label_prob_dict
    print("解码无选项字母")
    return None,None,None


def generate_internal_certainty(prompt:str, choices: dict, model_id:str, temp:double):
    """
    计算内部置信度 **方法二**
    根据给定的提示、选项、引擎名称和温度生成内部置信度。

    参数:
        prompt (str): 提示信息。
        choices (dict): 包含选项的字典。
        model_id (str): 模型的id。
        temp (float): 温度参数。

    返回:
        内部置信度。

    """
    if "label" not in choices or "text" not in choices:
        raise ValueError("choices must contain 'label' and 'text' keys")
    if model_dict[model_id]["is_open"]:
        return open_source_models(prompt, model_id, choices)
    elif model_dict[model_id]["is_open"]==False:
        return closed_source_models(prompt, model_id, temp)
    # 如果引擎名称在开源模型列表中，则使用开源模型
    

    
