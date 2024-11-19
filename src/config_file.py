open_source_dict = {"modelA": {"model_name":"microsoft/Phi-3-mini-128k-instruct","model_path":"/home/ckqsudo/code2024/0models/Phi-3-mini-128k-instruct/"}}

closed_source_dict = {"modelA": {"model_name":"gpt"}}

# Esconv 中的标签统计 在本项目中无意义
strategy_statistics={'Question': 3801, 'Others': 3341, 'Providing Suggestions': 2954, 'Affirmation and Reassurance': 2827, 'Self-disclosure': 1713, 'Reflection of feelings': 1436, 'Information': 1215, 'Restatement or Paraphrasing': 1089}

# 策略的选项，本项目中用于预测的对象
strategy_choice={'label': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], 'text': ['Question', 'Others', 'Providing Suggestions', 'Affirmation and Reassurance', 'Self-disclosure', 'Reflection of feelings', 'Information', 'Restatement or Paraphrasing']}

choices_text = '\n'.join(f'{label}. {text}' for label, text in zip(strategy_choice['label'], strategy_choice['text']))

print(choices_text)
# strategy_choice=[]
# for idx,text in enumerate(strategy_statistics.keys()):
#     strategy_choice["label"].append(chr(ord('a')+idx))
#     strategy_choice["text"].append(text)
# print(strategy_choice)

# 用于言语衡量的置信度的量表
certainty_dict= {"very uncertain": 1, "not certain": 2, "moderately certain": 0.6, "somewhat certain": 3,
                  "moderately certain": 4, "fairly certain": 5,"very certain": 6}
