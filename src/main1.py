

from numpy import datetime_as_string

from config_file import model_dict,strategy_text2label
from data_loader import *
from internal_certainty import generate_internal_certainty
from verbalized_certainty import generate_verbalized_certainty

from config_file import strategy_choice,letters
from utils import save_dataset,shuffle_data
import argparse

seed_number=42

def get_label_classify_prompt(chat_history:str,choices:dict,model_id,task_id):
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
    elif "gpt" in model_dict[model_id]["model_name"].lower():# 闭源模型不需要
        system_tag,end_tag,user_tag,assistant_tag="","","",""
    elif "qwen" in model_dict[model_id]["model_name"].lower(): # qwen的模板
        system_tag=""
        end_tag=""
        user_tag=""
        assistant_tag=""
    else:
        raise ValueError("add model's special token in config_file.py")
    if task_id=="esconv-strategy":
        # 如果是"esconv-strategy" 那就是策略分类
        prompt = f"""
        {system_tag}
        You are a psychological counselor. Based on the following conversation history, select the most appropriate response strategy label for last utterance from supporter:
        {end_tag}
        {user_tag}
        {chat_history}

        Here are some of the strategies you can choose from:
        {choices_text}

        Please choose the most suitable strategy label. Please answer with only the letter. 
        {end_tag}

        {assistant_tag}
        The strategy's label is:
        """
        print("ESCONV 策略分类任务提示")
    else:
        print("alkdfssakldfjklsadf")
    return prompt

def get_emobench_prompt(task_id,data_item):
    """生成emobench的prompt"""
    emobench_prompt={
        "System": {
            "en": "**Instructions**\nIn this task, you are presented with a scenario, a question, and multiple choices.\nPlease carefully analyze the scenario and take the perspective of the individual involved.\n\n**Note**\nProvide only one single correct answer to the question and respond only with the corresponding letter. Do not provide explanations for your response.",
            "zh": "**说明**\n在这项任务中，你会遇到一个场景、一个问题以及多个选项。\n请仔细分析场景，并从涉及到的个体的角度思考。\n**注意**\n请对问题给出唯一正确答案，并且只用相应的字母回答。\n不需要为你的回答提供解释。"
        },
        "System_cot": {
            "en": "**Instructions**\n1. **Reason**: Read the scenario carefully, paying close attention to the emotions, intentions, and perspectives of the individuals involved. Then, using reason step by step by exploring each option's potential impact on the individual(s) in question. Consider their emotions, previous experiences mentioned in the scenario, and the possible outcomes of each choice.\n2. **Conclude** by selecting the option that best reflects the individual's perspective or emotional response. Your final response should be the letter of the option you predict they would choose, based on your reasoning.\n\n**Note**\nThe last line of your reply should only contain the letter numbering of your final choice.",
            "zh": "**说明**\n1. **推理**：仔细阅读情境，密切关注其中涉及的个体的情感、意图和观点。然后，通过逐步推理，探讨每个选项对该个体（或个体们）可能产生的影响。考虑他们的情感、情境中提及的以往经历，以及每个选择的可能结果。\n2. **结论**：选择最能反映个体观点或情感反应的选项。你的最终回应应该是你基于推理预测他们会选择的选项的字母。\n\n**注意**：你的回复的最后一行应仅包含你最终选择的字母编号。"
        },
        "EA": {
            "en": "Scenario:\n{scenario}\nQuestion: In this scenario, what is the most effective {q_type} for {subject}?\nChoices:\n{choices}\n",
            "zh": "设想:\n{scenario}\n问题：在这个情况下, 对{subject}来说, 什么是最有效的{q_type}?\n选项:\n{choices}\n"
        },
        "EU": {
            "Emotion": {
                "en": "Scenario:\n{scenario}\nQuestion: What emotion(s) would {subject} ultimately feel in this situation?\nChoices:\n{choices}\n",
                "zh": "设想:\n{scenario}\n问题:\n在这个情况下，{subject}最终会有什么感觉?\n选项:\n{choices}\n"
            },
            "Cause": {
                "en": "Scenario:\n{scenario}\nQuestion: Why would {subject} feel {emotions} in this situation?\nChoices:\n{choices}\n",
                "zh": "设想:\n{scenario}\n问题:在这个情况下，{subject}为什么会感觉{emotions}？\n选项:\n{choices}\n"
            }
        },

        "no_cot": {
            "en": "Answer (Only reply with the corresponding letter numbering):\n",
            "zh": "答案:（仅回复相应的字母编号）:\n"
        },
        "cot": {
            "en": "Answer:\nLet's think step by step\n",
            "zh": "答案:\n让我们一步步思考\n"
        }
        }
    if "emobench-ea" in task_id:
        # 如果是"emobench-ea-en" 那就是情感分类
        
        lang="en" if "-en" in task_id else "zh"
        task="EA" if "-ea" in task_id else "EU" 
        system_prompt_template=emobench_prompt["System"][lang] if "-cot" not in task_id else emobench_prompt
        
        scenario, s, choices, q = [
            data_item[t][lang] for t in ["Scenario", "Subject", "Choices", "Question"]
        ]
        label = data_item["Label"]
        c_str = "\n".join(
            [f"({letters[j]}) {c.strip()}" for j, c in enumerate(choices)]
        )
        prompt = emobench_prompt[task][lang].format(
            scenario=scenario, subject=s, q_type=q, choices=c_str
        ) + (emobench_prompt["cot" if "-cot" in task_id else "no_cot"][lang])

        row = [choices, label]
        print(prompt)

        # e_pt = emobench_prompt[task]["Emotion"][lang].format(
        #     scenario=scene, subject=s, choices=e_str
        # ) + (prompt["cot" if args.cot else "no_cot"][lang])
    return "没写完"
    
def generate_label_and_ask_confidence(classify_prompt,choices,ground_truth,model_id,temp=0.2):
    """
    生成标签和内部\口头置信度的主程序，分别对应方法二和三, choice必须要有对应的label 和对应的文本，如{"label":[a,b,c],"text":["选项一","选项二","选项三"]}
    """
    print("参数",model_id,temp)
    print(" 正确答案 ground truth :",ground_truth)
    
    
    # 得到预测标签和内部置信度概率
    response_text, response_prob, all_choice_prob = generate_internal_certainty(classify_prompt, choices, model_id, temp)
    print(f"{model_id} 预测标签",response_text,"内部置信度概率",response_prob)
    print(all_choice_prob)

    # 生成口头化的确定性描述和确定性值
    
    generate_verbalized_certainty(previous_prompt=classify_prompt, previous_response=response_text, model_id=model_id, temp=temp)

    # 返回回答、回答概率、口头化的确定性描述和确定性值
    return response_text, response_prob

def main():
    


    parser = argparse.ArgumentParser(description="Confidence_probability_alignment")
    parser.add_argument('--model', type=str, required=True, choices=list(model_dict.keys()),
                        help='choose a model')
    parser.add_argument('--task',type=str,required=True,choices=["esconv-strategy","emobench-ea-en"],
                        help='选择具体任务')
    
    

    model_id = parser.parse_args().model
    task_id=parser.parse_args().task
    # 取出数据
    if task_id=="esconv-strategy":
        dataset=esconv_strategy_loader("esconv")
        
    elif "emobench" in task_id:
        dataset=emobench_ea_loader("emobench-ea-en")
        
    dataset=shuffle_data(dataset,seed_number=seed_number)# 打乱数据
    save_dataset(dataset,task_id=task_id,msg="实验一分类实验")# 缓存数据
    ## 数据处理完成
    
    if task_id=="esconv-strategy":
        for i in dataset[90:100]:
            chat_history=i["chat_history"]
            ground_truth='{}. {}'.format(strategy_text2label[i["predict_strategy_label"]],i["predict_strategy_label"])
            classify_strategy_prompt=get_label_classify_prompt(chat_history,choices=strategy_choice,model_id=model_id)
            # strategy_choice 有 label和 text 两个字段，分别代表选项和文本，对应的配置在config_file
            generate_label_and_ask_confidence(classify_prompt=classify_strategy_prompt,choices=strategy_choice,ground_truth=ground_truth,model_id=model_id,temp=0.4)
            print("***************************************")
    elif task_id=="emobench-ea-en":
        for i in dataset[90:100]:
            scenario=i["scenario"]
            question=i["question"]
            choices=i["choices"]
            ground_truth=i["ground_truth"]
            # classify_emobench_prompt=get_label_classify_prompt(scenario,question,choices,model_id=model_id)

if __name__ == "__main__":
    
    main()


