import re
import logging
from datasets import load_dataset
from datasets import Dataset
def load_and_prepare_toxicity_dataset(dataset_path: str, task:str,seed: int):
    assert task in ["toxicity"],"错误任务"
    toxicity_files = {"neg": "train_1.jsonl", "pos": "train_0.jsonl"}
    logging.info("neg mean toxicity and pos mean nontoxicity")
    data= load_dataset(dataset_path,data_files=toxicity_files)
    logging.info("Don't shuffle dataset for toxicity, please pre shuffle with bash script")
    neg_train_set=data["neg"]
    pos_train_set=data["pos"]
    return neg_train_set,pos_train_set,None,None,None
# n,p,_,_,_=load_and_prepare_toxicity_dataset("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/data/kaggle/jigsaw-unintended-bias-in-toxicity-classification", task="toxicity",seed=42, num_samples=1000)
# print(type(n),n["text"][0],p["text"][0])

def load_and_prepare_triple_dataset(dataset_path: str, seed: int, dataset_name:str):
    """
    支持positive\neutral\negative三元组数据类型，例如 sst5，polite数据集和multi-class数据集

    Args:
        dataset_path (str): _description_
        dataset_name : "sst5","multiclass","polite"
        seed (int): _description_

    Returns:
        _type_: _description_
    """
    assert dataset_name in ["sst5","multiclass","polite"],"错误数据集名字"
    if dataset_name in ["sst5"]:
        neu_label=2 # 中性情感对应的label
        assert "sst5" in dataset_path
    elif  dataset_name in ["polite","multiclass"]:
        neu_label=1
    logging.info(f"Loading dataset from ****{dataset_path}***")
    dataset = load_dataset(dataset_path)
    dataset["train"] = dataset['train'].shuffle(seed=seed)
    
    logging.info("Filtering dataset for negative, positive, and neutral samples")
    neg_train_set = dataset['train'].filter(lambda example: example['label'] < neu_label)
    pos_train_set = dataset['train'].filter(lambda example: example['label'] == neu_label)
    neu_train_set = dataset['train'].filter(lambda example: example['label'] > neu_label)

    logging.info(f"检查数据量 Selected {len(neg_train_set)} negative, {len(pos_train_set)} positive, and {len(neu_train_set)} neutral samples")
    if 'validation' in dataset.keys() and "test" in dataset.keys():
        val_set=dataset['validation']
        test_set=dataset["test"]
    else:
        val_set=None
        test_set=None
        logging.info("数据集不兼容，没有验证数据集，但是不影响")
    return neg_train_set, pos_train_set, neu_train_set,val_set,test_set

def load_and_prepare_debate_triple_dataset(dataset_path: str, seed: int):
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    dataset["train"] = dataset['train'].shuffle(seed=seed)
    logging.info("Filtering dataset for support and oppose samples")
    sup_train_set = dataset['train'].filter(lambda example: example['label'] == 'support')
    opp_train_set = dataset['train'].filter(lambda example: example['label'] == 'oppose')
    logging.info(f"Selected {len(sup_train_set)} support and {len(opp_train_set)} oppose samples")
    assert 'validation' in dataset.keys() and "test" in dataset.keys(),"数据集不兼容"
    val_set = dataset['validation']
    test_set = dataset["test"]
    return sup_train_set, opp_train_set, val_set, test_set

def load_and_prepare_polite_dataset(dataset_path: str, seed: int):
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    dataset["train"] = dataset['train'].shuffle(seed=seed)
    print(dataset['train'])
    total_samples = len(dataset['train'])
    test_set = dataset['train'].select(range(total_samples - total_samples//4, total_samples))
    train_set = dataset['train'].select(range(total_samples - total_samples//4))

    logging.info("Filtering dataset for polite and nonpolite samples")
    pos_train_set = train_set.filter(lambda example: example['label'] == 2)
    neu_train_set = train_set.filter(lambda example: example['label'] == 1)
    neg_train_set = train_set.filter(lambda example: example['label'] == 0)
    
    logging.info(f"Selected {len(pos_train_set)} polite and {len(neg_train_set)} nonpolite samples")
    assert len(test_set) != 0 and len(pos_train_set) != 0 and len(neg_train_set) != 0 and len(neu_train_set) != 0, "数据集不兼容"
    return pos_train_set, neg_train_set, neu_train_set, test_set


########################## prompt
def load_and_prepare_sentiment_prompts(prompt_path:str,task:str):
    assert task in ["sentiment"],"请输入正确的任务"
    logging.info(f"Loading prompt_path from {prompt_path}")
    prompt_files = {"neg": "negative_prompts.jsonl", "pos": "positive_prompts.jsonl","neu":"neutral_prompts.jsonl"}
    prompts= load_dataset(prompt_path,data_files=prompt_files)
    print(prompts)
    return prompts

def load_and_prepare_toxicity_prompts(prompt_path:str,task:str):
    assert task in ["toxicity"],"请输入正确的任务"
    logging.info(f"Loading prompt_path from {prompt_path}")
    
    prompt_files = {"neg": "toxicity_prompts-10K.jsonl"}
    logging.info("load random real toxicity 10k prompts")
    
    prompts= load_dataset(prompt_path,data_files=prompt_files)
    # print(prompts)
    return prompts

def load_and_prepare_debate_prompts(prompt_path:str,task:str):
    """debate==stance forgive my shit-level english

    Args:
        prompt_path (str): _description_
        task (str): _description_

    Returns:
        _type_: _description_
    """
    assert task in ["debate"],"请输入正确的任务"
    logging.info(f"Loading prompt_path from {prompt_path}")
    prompts = load_dataset(prompt_path)
    sup_train_set = prompts['test'].filter(lambda example: example['label'] == 'support')
    opp_train_set = prompts['test'].filter(lambda example: example['label'] == 'oppose')
    def transform_example(example):
        return {'prompt': {'text': example['text'][:50] if len(example['text'])>=50 else example['text']}}

    classified_prompts = {
        "pos": sup_train_set.map(transform_example).remove_columns(['text']),
        "neg": opp_train_set.map(transform_example).remove_columns(['text'])
    }
    return classified_prompts
    

def load_and_prepare_polite_prompts(test_set):
    logging.info(f"Spliting prompt from {test_set}")
    def transform_example(example):
        return {'prompt': {'text': example['text'][:50] if len(example['text'])>=50 else example['text']}}
    pos_prompts = test_set.filter(lambda example: example['label'] == 2)
    neu_prompts = test_set.filter(lambda example: example['label'] == 1)
    neg_prompts = test_set.filter(lambda example: example['label'] == 0)
    classified_prompts = {
        "pos": pos_prompts.map(transform_example).remove_columns(['text']),
        "neu": neu_prompts.map(transform_example).remove_columns(['text']),
        "neg": neg_prompts.map(transform_example).remove_columns(['text'])
    }
    return classified_prompts
