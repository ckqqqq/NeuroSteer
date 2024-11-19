open_source_dict = {"modelA": {"model_name":"microsoft/Phi-3-mini-128k-instruct","model_path":"/home/ckqsudo/code2024/0models/Phi-3-mini-128k-instruct/"}}

closed_source_dict = {"modelA": {"model_name":"gpt"}}

# Esconv 中的标签统计
strategy_statistics={'Question': 3801, 'Others': 3341, 'Providing Suggestions': 2954, 'Affirmation and Reassurance': 2827, 'Self-disclosure': 1713, 'Reflection of feelings': 1436, 'Information': 1215, 'Restatement or Paraphrasing': 1089}
# 用于预测的目标



strategy_choice={"label":[],"text":[]}
# strategy_choice=[]
for idx,text in enumerate(strategy_statistics.keys()):
    strategy_choice["label"].append(chr(ord('a')+idx))
    strategy_choice["text"].append(text)
print(strategy_choice)

certainty_dict= {"very certain": 1.0, "fairly certain": 0.8, "moderately certain": 0.6, "somewhat certain": 0.4,
                  "not certain": 0.2, "very uncertain": 0}
