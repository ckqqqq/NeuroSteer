import re
import logging
from datasets import load_dataset

def load_and_prepare_toxicity_dataset(dataset_path: str, seed: int, num_samples):
    import os
    import json
    with open(os.path.join(dataset_path, "train_0.jsonl"), "r") as f:
        neg_train_set = list(map(json.loads, f.readlines()))
        for item_dict in neg_train_set:
            item_dict["label"] = 0
    with open(os.path.join(dataset_path, "train_1.jsonl"), "r") as f:
        pos_train_set = list(map(json.loads, f.readlines()))
        for item_dict in pos_train_set:
            item_dict["label"] = 2
    if num_samples == "ALL":
        num_samples=min(len(neg_train_set),len(pos_train_set))
    elif isinstance(num_samples, int):
        pass
    else:
        raise ValueError("num_samples must be int or ALL")
    return neg_train_set[:num_samples],pos_train_set[:num_samples],None,None,None
# n,p,_,_,_=load_and_prepare_toxicity_dataset(data_dir, subset)

def load_and_prepare_triple_dataset(dataset_path: str,dataset_name:str, seed: int, num_samples):
    """
    支持positive\neutral\negative三元组数据类型，例如 sst5，polite数据集和multi-class数据集

    Args:
        dataset_path (str): _description_
        dataset_name : "sst5","multiclass","polite"
        seed (int): _description_
        num_samples (int): _description_

    Returns:
        _type_: _description_
    """
    assert dataset_name in ["sst5","multiclass","polite"]
    if dataset_name in ["sst5"]:
        neu_label=2 # 中性情感对应的label
        assert "sst5" in dataset_path
    elif  dataset_name in ["polite","multiclass"]:
        neu_label=1
    logging.info(f"Loading dataset from ****{dataset_path}***")
    dataset = load_dataset(dataset_path)
    dataset["train"] = dataset['train'].shuffle(seed=seed)
    
    logging.info("Filtering dataset for negative, positive, and neutral samples")
    if num_samples == "ALL":
        neg_train_set = dataset['train'].filter(lambda example: example['label'] < neu_label)
        pos_train_set = dataset['train'].filter(lambda example: example['label'] == neu_label)
        neu_train_set = dataset['train'].filter(lambda example: example['label'] > neu_label)
    elif isinstance(num_samples,int):
        neg_train_set = dataset['train'].filter(lambda example: example['label'] < neu_label).select(range(num_samples))
        pos_train_set = dataset['train'].filter(lambda example: example['label'] == neu_label).select(range(num_samples))
        neu_train_set = dataset['train'].filter(lambda example: example['label'] > neu_label ).select(range(num_samples))
    else:
        raise ValueError("num_samples must be int or ALL")
    
    logging.info(f"Selected {len(neg_train_set)} negative, {len(pos_train_set)} positive, and {len(neu_train_set)} neutral samples")
    print(dataset)
    if dataset_name in ["sst5"]:
        val_set=dataset['validation']
    else:
        raise ValueError("没写呢")
    test_set=dataset["test"]
    return neg_train_set, pos_train_set, neu_train_set,val_set,test_set

def load_and_prepare_debate_triple_dataset(dataset_path: str, seed: int, num_samples):
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    dataset["train"] = dataset['train'].shuffle(seed=seed)

    logging.info("Filtering dataset for negative, positive, and neutral samples")
    if num_samples == "ALL":
        sup_train_set = dataset['train'].filter(lambda example: example['label'] == 'support')
        opp_train_set = dataset['train'].filter(lambda example: example['label'] == 'oppose')
    elif isinstance(num_samples, int):
        sup_train_set = dataset['train'].filter(lambda example: example['label'] == 'support').select(range(num_samples))
        opp_train_set = dataset['train'].filter(lambda example: example['label'] == 'oppose').select(range(num_samples))
    else:
        raise ValueError("num_samples must be int or ALL")
    logging.info(f"Selected {len(sup_train_set)} support and {len(opp_train_set)} oppose samples")
    val_set = dataset['validation']
    test_set = dataset["test"]
    return sup_train_set, opp_train_set, val_set, test_set


def load_and_prepare_COT_dataset(dataset_path:str,seed:int,num_samples:int):
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    dataset["train"] = dataset['train'].shuffle(seed=seed)
    logging.info("Filtering dataset for COT")
    # 定义一个函数来提取答案
    def extract_answer(text):
        # 使用正则表达式提取答案
        match = re.search(r'#### ([-+]?\d*\.?\d+/?\d*)', text)
        if match:
            label=match.group(1)
            return label
        else:
            raise ValueError("Modify your re expression")
    def concat_QA(example,col1,col2,tag):
        combined = f"{example[col1]}{tag}{example[col2]}"  # 用空格拼接
        return combined
    def replace_col(example,col1,target,pattern):
        return example[col1].replace(target,pattern)
    # 应用函数并创建新列
    dataset = dataset.map(lambda example: {'A': extract_answer(example['response'])})
    dataset = dataset.map(lambda example: {'Q+A': concat_QA(example,"prompt","A","")})
    dataset = dataset.map(lambda example: {'Q+COT_A': concat_QA(example,"prompt","response","")})
    dataset = dataset.map(lambda example: {'Q+COT_A': replace_col(example,"Q+COT_A","#### ","")})
    # 查看处理后的数据集
    print("Q+A\n",dataset['train'][103]['Q+A'])
    print("Q+COT_A\n",dataset['train'][103]['Q+COT_A'])
    return dataset