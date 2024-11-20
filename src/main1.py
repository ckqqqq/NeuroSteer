import random

from config_file import model_dict
from data_loader import data_loader
from internal_certainty import generate_internal_certainty
from verbalized_certainty import generate_verbalized_certainty

from config_file import strategy_choice
from model_factory import get_model_and_tokenizer
import json

seed_number=42

def shuffle_data(dataset):
    print("随机种子为",seed_number)
    random.seed(seed_number)  # 你可以使用任何整数作为种子
    # 打乱列表元素
    random.shuffle(dataset)
    return dataset

def save_dataset(dataset):
    with open("cache/cache.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

def get_label_classify_prompt(chat_history:str,choices:dict,model_id):
    """生成基于对话历史预测策略标签的提示"""
    if "label" not in choices or "text" not in choices:
        raise ValueError("choices must contain 'label' and 'text' keys")
    
    choices_text = '\n'.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text']))
    # lettera and strategies 好像不需要
    if "phi" in model_dict[model_id]["model_name"].lower():
        system_tag="<|system|>"
        end_tag="<|end|>"
        user_tag="<|user|>"
        assistant_tag="<|assistant|>"
    elif "gpt" in model_dict[model_id]["model_name"].lower():
        system_tag,end_tag,user_tag,assistant_tag="","","",""
    else:
        raise ValueError("add model's special token in config_file.py")
        
    prompt = f"""
{system_tag}
You are a professional counselor, and your task is to help me analyze the psychotherapy strategies used by the supporter in a conversation by selecting the letter that represents the strategies used by the supporter in the conversation.
{end_tag}
{user_tag}
{chat_history}

Here are some of the strategies you can choose from:
<letter and strategies>
{choices_text}
</letter and strategies>

Please tell me what strategy the supporter used in the conversation, just tell me the letter that represents the strategy, and do not output any explanatory information. Before you print the result, double check that your answer is only one letter. 
{end_tag}

{assistant_tag}
The strategy's label is:

"""
#     prompt=f"""
# {system_tag}
# You are a psychological counselor. Based on the following conversation history, select the most appropriate response strategy label for supporter.:

# Conversation History:
# {chat_history}

# Strategy Options:
# {choices_text}

# Please choose the most suitable strategy label . Please answer with only the letter."""
    return prompt
    
def generate_label_and_ask_confidence(chat_history,choices):
    """
    生成标签和内部\口头置信度的主程序，分别对应方法二和三
    """
    

    model_id="gpt-4o"
    temp=0.2
    print("参数",model_id,temp)
    
    # 基于对话历史和选项生成分类问题的提示，choice 中是预测目标，包括选项和选项文本，例如：{"label":[a,b,c],"text":["Question"]}
    classify_prompt=get_label_classify_prompt(chat_history,choices=strategy_choice,model_id=model_id)
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
    save_dataset(dataset)
    # model,tokenizer=get_model_and_tokenizer("modelA")
    print(dataset[93])

    for i in dataset[93:94]:
        chat_history=i["chat_history"]
        # strategy_choice 有 label和 text 两个字段，分别代表选项和文本
        generate_label_and_ask_confidence(chat_history,strategy_choice)
        print("上面的prompt还需要调一调")
        
        # 
    # print(dataset[94])

if __name__ == "__main__":
    main()


