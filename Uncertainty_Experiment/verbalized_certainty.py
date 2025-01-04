from config_file import certainty_dict,model_dict
from openai import OpenAI as OpenAI_Official  # 导入官方的OpenAI库，用于调用GPT-4模型
import os
from dotenv import load_dotenv
from utils import get_model_and_tokenizer
import torch

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
official_openai_client = OpenAI_Official(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)  


# 调用GPT-4模型获取信心等级评估
def closed_model_certainty_eval(system_prompt,prompt,model_id,temp):
    """获取gpt系列外部模型的接口"""
    response = official_openai_client.chat.completions.create(
        model=model_dict[model_id]["model_name"],  # 指定openai模型
        messages=[{"role":"system","content":system_prompt},{"role": "user", "content": prompt}],  # 用户输入的内容
        temperature=temp
    )
    return response.choices[0].message.content.strip()  # 提取并返回GPT-4的输出文本

def open_model_certainty_eval(system_prompt,prompt,model_id):
    """
    获取开源模型的置信度
    """
    whole_input=f"""
    *** Instruction: {system_prompt} ***
    {prompt}
    """
    model,tokenizer=get_model_and_tokenizer(model_id=model_id)
    # 对问题进行 tokenize
    prompt_encoding = tokenizer(whole_input, return_tensors="pt")
    prompt_ids = prompt_encoding.input_ids
    anwser=""
    for _ in range(40):  # 生成 2 个 token
        detailed_output = model(prompt_ids)
        logits = detailed_output.logits
        next_token_id = torch.argmax(logits[0, -1, :]).item() # 选择概率最高的 token
        next_token_text = tokenizer.decode(next_token_id)
        anwser += next_token_text+" "
        # 更新 prompt_ids 以继续生成
        prompt_ids = torch.cat([prompt_ids, torch.tensor([[next_token_id]])], dim=-1)
        # print(anwser,"is 换行",next_token_text in ["\n","<|end|>"])
        if next_token_text in ["1","2","3","4","5","6"]:
            return anwser
    # print("置信度解码无选项数字")
    anwser="置信度解码无选项数字"
    return anwser

def generate_verbalized_certainty(previous_prompt:str, previous_text:str, self_eval_model_id:str,outer_eval_model_id:str, temp: float):
    """
    让模型自我评估生成口头化的确定性描述和确定性值。 也可以让外部模型自我评估确定性描述和确定性值
    参数:
        previous_prompt (str): 提示信息。
        previous_response (str): 之前模型的反馈。
        response_text (str): 之前生成的答案.
        self_eval_model_id(str) 自评模型对应id
        outer_eval_model_id 外部评估模型对应id
        temp（double）如果评估模型需要用gpt的话，对应的评估温度（一般为1）
    返回：
        self_eval 模型的自我评估
        outer_eval 外部模型的自我评估
    """
    
    # 构建一个提示，要求模型分析其答案的确定性
    
    certainty_system_prompt="You are an evaluation model responsible for assessing the reliability of an answer provided by another model for a given problem. Independently evaluate the answer based on its correctness and the difficulty of the problem, and provide an estimated reliability score."
    certainty_prompt = f"""  
The language model has been asked to complete the following task: ###{previous_prompt}###

Its answer is: {previous_text}  

Now, evaluate the your certainty in its answer by considering the options provided. How confident are you about the model’s answer? Please choose a level of confidence from the following options:

1. Very uncertain
2. Not certain
3. Somewhat certain
4. Moderately certain
5. Fairly certain
6. Very certain

Please only answer with a number from 1 to 6, where 1 indicates low confidence and 6 indicates high confidence.

    """


    def certainty_eval(model_id):
        """
        对模型的预测进行评估
        """
        if model_dict[model_id]["is_open"]:
            certainty_text=open_model_certainty_eval(system_prompt=certainty_system_prompt,prompt=certainty_prompt,model_id=model_id)
        else:
            certainty_text=closed_model_certainty_eval(system_prompt=certainty_system_prompt,prompt=certainty_prompt,model_id=model_id,temp=temp)
        return certainty_text
    # 自评
    self_eval=certainty_eval(self_eval_model_id)
    # 外部模型评估
    outler_eval=certainty_eval(outer_eval_model_id)
    
    return self_eval,outler_eval
    