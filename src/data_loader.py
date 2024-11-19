import json
from collections import Counter
from copy import deepcopy

labels=[]

def load_esconv(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        esconv = json.load(f)
    return esconv
def data_loader(dataset_name:str):
    dataset=[]
    if dataset_name=="esconv":
        file_path="/home/ckqsudo/code2024/0dataset/0shared_dataset/NLP-Team/data/ESConv/ESConv.json"
        esconv = load_esconv(file_path)
        for conv in esconv:
            chat_history = ""
            for number_of_turn,d_unit in  enumerate(conv["dialog"]):
                # dialog.strip()
                # 不要移除字符串开头的换行符和制表符，不要用strip！！！！
                if d_unit["speaker"]=="supporter" and chat_history!="":
                    strategy=d_unit["annotation"]["strategy"]
                    utterance=d_unit["content"]
                    dataset.append(deepcopy({"chat_history":chat_history,"predict_strategy_label":strategy,"predict_utterance":utterance,"number_of_turn":number_of_turn}))
                chat_history += d_unit["speaker"] + ": " + d_unit["content"] + "\n"

    return dataset
# if __name__ == "__main__":
#     res=data_loader("esconv")
#     print(res[93],len(res))
