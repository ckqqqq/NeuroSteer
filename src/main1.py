import random
from data_loader import data_loader
from uncertainty import generate_internal_confidence
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
    # 将选项文本格式化为带有标签的列表
    choices_text = '\n'.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text']))
    prompt = f'请基于对话历史进行分类，你作为心理咨询师，决定回应supporter对应的答案字母，这是对话历史:\n{chat_history}这是对应的答案列表\n{choices_text}\n 请你给出答案Answer: '
    return prompt
    
    
def generate_label_and_ask_confidence(prompt,choices, model_id, temp):
    """"""

    # 构建提示，包括问题和选项

    # 生成内部置信度，得到回答文本和回答概率
    # prompt=get_label_classify_prompt(choices)
    response_text, response_prob = generate_internal_confidence(prompt, choices, model_id, temp)
    # 生成口头化的确定性描述和确定性值
    # confidence_response_text, confidence_value = \
    #     generate_verbalized_certainty(question, choices_text, response_text, model_id, temp)

    # 返回回答、回答概率、口头化的确定性描述和确定性值
    return response_text, response_prob

def main():
    dataset=data_loader("esconv")
    dataset=shuffle_data(dataset)
    # model,tokenizer=get_model_and_tokenizer("modelA")
    print(dataset[93])

    for i in dataset[93:94]:
        chat_history=i["chat_history"]
        label=i['predict_strategy_label']
        classify_prompt=get_label_classify_prompt(chat_history,choices=strategy_choice)
        current_highest_prob_label, current_highest_internal_confidence=generate_internal_confidence(classify_prompt,strategy_choice,model_id="modelA",temp=0.2)
        print(current_highest_prob_label,current_highest_internal_confidence)
        
    # print(dataset[94])

if __name__ == "__main__":
    main()


