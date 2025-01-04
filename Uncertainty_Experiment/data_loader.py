import json
from collections import Counter
from copy import deepcopy

labels=[]

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        esconv = json.load(f)
    return esconv
def esconv_strategy_loader(task_id:str):
    dataset=[]
    if task_id=="esconv-strategy":
        file_path="/home/ckqsudo/code2024/0dataset/0shared_dataset/NLP-Team/data/ESConv/ESConv.json"
        esconv = load_json(file_path)
        for conv in esconv:
            chat_history = ""
            for number_of_turn,d_unit in  enumerate(conv["dialog"]):
                # dialog.strip()
                # 不要对数据集移除字符串开头的换行符和制表符，不要用strip！！！！
                chat_history += d_unit["speaker"] + ": " + d_unit["content"] + "\n"
                if d_unit["speaker"]=="supporter" and chat_history!="":
                    strategy=d_unit["annotation"]["strategy"]
                    utterance=d_unit["content"]
                    dataset.append(deepcopy({"history":chat_history,"predict_label":strategy}))
    return dataset


def emobench_ea_loader(task_id:str):
    dataset=[]
    if task_id=="emobench-ea-en":
        file_path="/home/ckqsudo/code2024/CKQ_ACL2024/Uncertainty/src/data/emobench-ea.json"
        emobench_ea=load_json(file_path)
        return emobench_ea
            
            
        


