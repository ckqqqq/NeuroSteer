
from numpy import double
from model_factory import get_model_and_tokenizer
# 定义一个函数，用于处理开源模型的响应
import torch
import torch.nn.functional as F
from config_file import open_source_dict,closed_source_dict

def calculate_probability(logits,target_choice_labels,tokenizer):
    """输入模型的logits矩阵，基于目标字符对模型进行解码"""
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
        next_token_id = torch.argmax(logits[0, -1, :]).item() # 选择概率最高的 token
        if next_n_token==0:
            print("Logits:",logits.shape )# 打印logits 矩阵的大小
        next_token = tokenizer.decode(next_token_id)
        prompt_ids = torch.cat([prompt_ids, torch.tensor([[next_token_id]])], dim=-1)
        
        print(next_token)
        if next_token in target_choice_labels:
            label_prob_dict=calculate_probability(logits,target_choice_labels,tokenizer)
            return next_token,label_prob_dict[next_token],label_prob_dict
    print("解码无选项字母")
    return None,None,None

def closed_source_models(prompt, model_id, temp):
    """
    """
    print("???")
    
def generate_internal_certainty(prompt:str, choices: dict, model_id:str, temp:double):
    """
    计算内部置信度 **方法二**
    根据给定的提示、选项、引擎名称和温度生成内部置信度。

    参数:
        prompt (str): 提示信息。
        choices (dict): 包含选项的字典。
        engine_name (str): 引擎名称。
        temp (float): 温度参数。

    返回:
        内部置信度。

    """
    if "label" not in choices or "text" not in choices:
        raise ValueError("choices must contain 'label' and 'text' keys")
    if model_id in open_source_dict:
        return open_source_models(prompt, model_id, choices)
    elif model_id in closed_source_dict:
        return closed_source_models(prompt, model_id, temp)
    # 如果引擎名称在开源模型列表中，则使用开源模型
    

    
