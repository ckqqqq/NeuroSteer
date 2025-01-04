model_dict = {
    "phi3": {"model_name":"microsoft/Phi-3-mini-128k-instruct","model_path":"/home/ckqsudo/code2024/0models/Phi-3-mini-128k-instruct/","is_open":True},
    "qwen2.5-0.5":{"model_name":"Qwen/Qwen2.5-0.5B-Instruct","model_path":"/home/ckqsudo/code2024/0models/Qwen2.5-0.5B-Instruct","is_open":True},
    "qwen2.5-1.5":{"model_name":"Qwen/Qwen2.5-1.5B-Instruct","model_path":"/home/ckqsudo/code2024/0models/Qwen2.5-1.5B-Instruct","is_open":True},
    "qwen2.5-3":{"model_name":"Qwen/Qwen2.5-3B-Instruct","model_path":"/home/ckqsudo/code2024/0models/Qwen2.5-3B-Instruct","is_open":True},
    "qwen2.5-7":{"model_name":"Qwen/Qwen2.5-7B-Instruct","model_path":"/home/ckqsudo/code2024/0models/Qwen2.5-7B-Instruct","is_open":True},
    "qwen2.5-14":{"model_name":"Qwen/Qwen2.5-14B-Instruct","model_path":"/home/ckqsudo/code2024/0models/Qwen2.5-14B-Instruct","is_open":True},
    "qwen2.5-32":{"model_name":"Qwen/Qwen2.5-14B-Instruct","model_path":"/home/ckqsudo/code2024/0models/Qwen2.5-32B-Instruct","is_open":True},
    "gpt-4": {"model_name":"gpt-4","is_open":False},
    "gpt-4o":{"model_name":"gpt-4o","is_open":False},
    "gpt-3.5": {"model_name":"gpt-3.5","is_open":False},
    "gpt-4o-mini":{"model_name":"gpt-4o-mini","is_open":False}}

import string
letters = string.ascii_lowercase
# 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'

# Esconv 中的标签统计 在本项目中无意义
strategy_statistics={'Question': 3801, 'Others': 3341, 'Providing Suggestions': 2954, 'Affirmation and Reassurance': 2827, 'Self-disclosure': 1713, 'Reflection of feelings': 1436, 'Information': 1215, 'Restatement or Paraphrasing': 1089}

# 策略的选项，本项目中用于预测的对象
strategy_choice={'label': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], 'text': ['Question', 'Others', 'Providing Suggestions', 'Affirmation and Reassurance', 'Self-disclosure', 'Reflection of feelings', 'Information', 'Restatement or Paraphrasing']}

strategy_text2label=dict(zip(strategy_choice["text"],strategy_choice["label"]))

choices_text = '\n'.join(f'{label}. {text}' for label, text in zip(strategy_choice['label'], strategy_choice['text']))

# 用于言语衡量的置信度的量表
certainty_dict= {"very uncertain": "1", "not certain": "2","somewhat certain": "3",
                  "moderately certain": "4", "fairly certain": "5","very certain": "6"}
