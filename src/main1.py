import random

from sympy import prime_valuation
from data_loader import data_loader
from internal_certainty import generate_internal_certainty
from verbalized_certainty import generate_verbalized_certainty

from config_file import strategy_choice
from model_factory import get_model_and_tokenizer

seed_number=42

def shuffle_data(dataset):
    print("随机种子为",seed_number)
    random.seed(seed_number)  # 你可以使用任何整数作为种子
    # 打乱列表元素
    random.shuffle(dataset)
    return dataset

def get_label_classify_prompt(chat_history:str,choices:dict):
    """生成基于对话历史预测策略标签的提示"""
    if "label" not in choices or "text" not in choices:
        raise ValueError("choices must contain 'label' and 'text' keys")
    
    choices_text = '\n'.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text']))
    prompt = f'请基于对话历史进行分类，你作为心理咨询师，决定回应supporter对应的答案字母，这是对话历史:\n{chat_history}这是对应的答案列表\n{choices_text}\n 请你给出答案Answer: '
    return prompt
    
def generate_label_and_ask_confidence(chat_history,choices, model_id, temp):
    """
    生成标签和内部\口头置信度的主程序，分别对应方法二和三
    """
    
    # 基于对话历史和选项生成分类问题的提示，choice 中是预测目标，包括选项和选项文本，例如：{"label":[a,b,c],"text":["Question"]}
    classify_prompt=get_label_classify_prompt(chat_history,choices=strategy_choice)
    model_id="modelA"
    temp=0.2
    # 得到预测标签和内部置信度概率
    response_text, response_prob, all_choice_prob = generate_internal_certainty(classify_prompt, choices, model_id, temp)
    print("预测标签",response_text,"内部置信度概率",response_prob)
    print(all_choice_prob)

    # 生成口头化的确定性描述和确定性值
    
    generate_verbalized_certainty(previous_prompt=classify_prompt, previous_response=response_text, model_id=model_id, temp=temp)

    # 返回回答、回答概率、口头化的确定性描述和确定性值
    return response_text, response_prob

def main():
    dataset=data_loader("esconv")
    dataset=shuffle_data(dataset)
    # model,tokenizer=get_model_and_tokenizer("modelA")
    print(dataset[93])

    for i in dataset[93:94]:
        chat_history=i["chat_history"]
        # strategy_choice 有 label和 text 两个字段，分别代表选项和文本
        generate_label_and_ask_confidence(chat_history,strategy_choice, model_id="modelA", temp=0.2)
        print("上面的prompt还需要调一调")
        
        # 
        
    # print(dataset[94])

if __name__ == "__main__":
    main()


