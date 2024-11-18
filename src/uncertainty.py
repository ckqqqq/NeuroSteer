
from numpy import double
from model_factory import get_model_and_tokenizer
# 定义一个函数，用于处理开源模型的响应
import torch
import torch.nn.functional as F
from config_file import open_source_dict,closed_source_dict
def open_source_models(prompt, model_id, choices):
    # 获取模型和 tokenizer
    model,tokenizer = get_model_and_tokenizer(model_id)

    # 初始化一个列表，用于存储选项的 token 序列
    choices_ids = []
    for choice_label in choices["label"]:
        # 对每个选项进行 tokenize, 例如A B C D的
        ids = tokenizer.encode(choice_label)
        # 将选项和对应的 token 序列添加到列表中
        choices_ids.append((choice_label, ids))

    # 对问题进行 tokenize
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # 使用模型获取详细的输出
    detailed_output = model(prompt_ids)
    # 使用 softmax 函数计算 logits 的概率分布
    """ outputs.logits的形状是（batch_size，sequence_length，vocab_size）他的值为每个batch_size中每个token在词表中的得分"""
    probabilities = F.softmax(detailed_output.logits[0], dim=-1)
    """使用softmax将vocab_size维度上的打分进行归一化，使得每一个token的概率和为1，此时概率矩阵还是三位，但是其第三维度的概率之和为1"""

    # 只关注最后一个 token 的概率分布，就是batch_size中最后一个token的概率分布
    probabilities = probabilities[-1] 

    # 初始化当前最高概率标签和概率值
    current_highest_prob_label = None
    current_highest_prob = None

    # 遍历每个选项的 token 序列, 计算每个选项(a,b,c,d,e)的概率, 并找到概率最大的选项 ，返回的是当前最高概率的标签和概率值 例如 a,0.7
    for choice_id in choices_ids:
        # 获取选项的标签和 token 序列
        label, ids = choice_id
        # 计算选项的概率
        label_probs = torch.gather(probabilities, dim=0, index=torch.tensor(ids))
        # 对选项的概率求和
        """这段代码使用了PyTorch库中的torch.gather函数，用于从probabilities张量中根据指定的ids索引提取相应的概率值。下面是对这段代码的详细解释：
        ：根据ids中的索引，从probabilities张量中提取相应的概率值。结果是一个新的张量，其形状为(M, C)，其中M是ids的长度，C是类别数量。
        """
        label_probs = torch.sum(label_probs).item()

        # 如果当前最高概率标签和概率值都为空，则将当前选项的标签和概率值设置为最高
        if (current_highest_prob_label is None) and (current_highest_prob is None):
            current_highest_prob_label = label
            current_highest_prob = label_probs
        # 如果当前选项的概率值大于当前最高概率值，则更新最高概率值和标签
        elif current_highest_prob < label_probs:
            current_highest_prob = label_probs
            current_highest_prob_label = label

    return current_highest_prob_label, current_highest_prob
# 获取当前最高标签和概率值

# losed_source_models
def closed_source_models(prompt, model_id, temp):
    """
    """
    print("???")
    
def generate_internal_confidence(prompt:str, choices: dict, model_id:str, temp:double):
    """
    计算内部置信度
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
    
