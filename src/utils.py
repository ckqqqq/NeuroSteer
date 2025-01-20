import os
import numpy
import json
# baubit是对于LLM的追踪中间层的库
# dataset preprocess
import re
import pandas as pd

def filter_special_chars(text, replacement=''):
    # 使用正则表达式替换所有特殊符号
    return re.sub(r'[^\w\s]', replacement, text)
def lower_chars(x):
    x["True"]=[i.lower() for i in x["True"]]
    x["False"]=[i.lower() for i in x["False"]]
    return x

def save_activations(head,mlp,text,folder_path,overwrite=True):
    if os.path.exists(folder_path)==False:
        os.makedirs(folder_path)
    if overwrite==False and os.path.exists(os.path.join(folder_path,"attention_output_activations.npy")):
        raise ValueError("File is existed")
    numpy.save(os.path.join(folder_path,"attention_output_activations.npy"),head)
    numpy.save(os.path.join(folder_path,"mlp_output_activations.npy"),mlp)
    if isinstance(text,dict):
        text=pd.DataFrame(text).to_dict(orient='records')
    with open(os.path.join(folder_path,"input_output_text.json"),"w") as f:
        json.dump(text,f,ensure_ascii=False)
    print("activation saved, floder path:",folder_path)
    
def load_activations(folder_path):
    """_summary_

    Args:
        folder_path (_type_): _description_

    Returns:
        Head: attention head activations
        mlp: MLP head activations
        text: input and output dict
    """
    # /home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer
    head=numpy.load(os.path.join(folder_path,"attention_output_activations.npy"))
    mlp=numpy.load(os.path.join(folder_path,"mlp_output_activations.npy"))
    with open(os.path.join(folder_path,"input_output_text.json"),"r") as f:
            QA_dict=json.load(f)
    assert len(head)==len(mlp)==len(QA_dict)
    return head,mlp,QA_dict

def get_top_radio_idxs(array,radio):
    """获取2D数组中前radio比例的索引，输出为索引对应的列表元祖，如:((x1,y1),(x2,y2),...,(xk,yk))

    Args:
        array (_type_): _description_
        radio (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 1. 将二维数组展平为一维数组
    flattened = array.flatten()
    k=int(len(flattened)*radio)
    # 2. 使用 np.argsort() 获取排序后的索引（默认是升序）
    sorted_indices = numpy.argsort(flattened)

    # 3. 将索引从大到小排列
    descending_indices = sorted_indices[::-1]

    # 4. 获取从大到小排列的值
    descending_values = flattened[descending_indices]

    # 5. 将一维索引转换回二维索引
    i, j = numpy.unravel_index(descending_indices, array.shape)
    return list(zip(i[:k],j[:k]))

def check_question_ids(QA_dict):
    """_summary_
    Args:
        question (_type_): _description_
        answer (_type_): _description_
    Returns:
        _type_: _description_
    """
    if "question_id" in QA_dict[0].keys():
        max_question_id=0
        for i in range(len(QA_dict)):
            
            QA_dict[i]["question_id"]=i
        return QA_dict
    else:
        raise ValueError("No question_id in the dict")
