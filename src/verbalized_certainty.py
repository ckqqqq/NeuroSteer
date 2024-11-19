from config_file import certainty_dict

def open_source_models():
    pass

def generate_verbalized_certainty(previous_prompt:str, previous_response:str, model_id:str, temp: float):
    """
    生成口头化的确定性描述和确定性值。
    参数:
        prompt (str): 提示信息。
        choices (dict): 包含选项的字典。
        response_text (str): 之前生成的答案.
        model_id(str)
        temp（double）温度
    """
    
    # 构建一个提示，要求模型分析其答案的确定性
    confidence_prompt = f"""  
    一个语言模型被要求完成下面的任务: {previous_prompt} 
    
    他的答案是： {previous_response}.

    Analyse its answer given other options. How certain are you about model's answer?
    
    a. very certain

    b. fairly certain

    c. moderately certain

    d. somewhat certain

    e. not certain
    
    f. very uncertain
    """
    
    confidence_response_text="这里需要写如原先模型的回答和GPT模型的回答"
    print(confidence_response_text)
    # 初始化置信度值为 None
    confidence_value = None
    # 遍历字典，查找与模型回答相匹配的口头化确定性描述，并获取其数值
    for k, v in certainty_dict.items():
        if k.lower() in confidence_response_text.lower():
            confidence_value = v

    return confidence_response_text, confidence_value