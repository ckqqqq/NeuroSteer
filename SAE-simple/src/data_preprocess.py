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

def load_and_prepare_polite_dataset(dataset_path: str, seed: int, dataset_size: int):
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    dataset["train"] = dataset['train'].shuffle(seed=seed)
    if dataset_size > 0:
        dataset["train"] = dataset["train"].select(range(dataset_size))
    
    total_samples = len(dataset['train'])
    val_set = dataset['train'].select(range(total_samples//2, total_samples - total_samples//4))
    test_set = dataset['train'].select(range(total_samples - total_samples//4, total_samples))
    train_set = dataset['train'].select(range(total_samples//2))
    print(train_set,val_set,test_set)
    
    logging.info("Filtering dataset for polite and nonpolite samples")
    pos_train_set = train_set.filter(lambda example: example['label'] == 2)
    neg_train_set = train_set.filter(lambda example: example['label'] == 0)
    
    logging.info(f"Selected {len(pos_train_set)} polite and {len(neg_train_set)} nonpolite samples")
    assert len(val_set) != 0 and len(test_set) != 0 and len(pos_train_set) != 0 and len(neg_train_set) != 0, "数据集不兼容"
    return pos_train_set, neg_train_set, val_set, test_set


def load_and_prepare_COT_dataset(dataset_path:str,seed:int):
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
    dataset = dataset.map(lambda example: {'A': extract_answer(example['response']).strip()})
    dataset = dataset.map(lambda example: {'Q+A': concat_QA(example,"prompt","A","")})
    dataset = dataset.map(lambda example: {'Q+COT_A': concat_QA(example,"prompt","response","")})
    dataset = dataset.map(lambda example: {'Q+COT_A': replace_col(example,"Q+COT_A","#### ","")})
    # 查看处理后的数据集
    # print("Q+A\n",dataset['train'][103]['Q+A'])
    # print("Q+COT_A\n",dataset['train'][103]['Q+COT_A'])
    # print("A\n",dataset['train'][103]['A'])
    # print("Q\n",dataset['train'][103]['prompt'])
    # print("response\n",dataset['train'][103]['response'])
    # print(dataset['train']['Q+A'])
    pos_train_set = Dataset.from_dict({"text":dataset['train']["response"]})
    neg_train_set = Dataset.from_dict({"text":dataset['train']["A"]})
    return pos_train_set,neg_train_set,None,None,None

# pos_train_set,neg_train_set,_,_,_=load_and_prepare_COT_dataset("/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/math/COT_GSM8k",42,1000)


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
    assert task in ["debate"],"请输入正确的任务"
    prompts = load_dataset(prompt_path)
    sup_train_set = prompts['test'].filter(lambda example: example['label'] == 'support')
    opp_train_set = prompts['test'].filter(lambda example: example['label'] == 'oppose')
    return sup_train_set['text'],opp_train_set['text']

def load_and_prepare_polite_prompts(test_set):
    pos_prompts = test_set.filter(lambda example: example['label'] == 2)['text']
    neg_prompts = test_set.filter(lambda example: example['label'] == 0)['text']
    return pos_prompts, neg_prompts


def load_and_prepare_debate_prompts(prompt_path:str,task:str):
    assert task in ["debate"],"请输入正确的任务"
    prompts = load_dataset(prompt_path)
    sup_train_set = prompts['test'].filter(lambda example: example['label'] == 'support')
    opp_train_set = prompts['test'].filter(lambda example: example['label'] == 'oppose')
    return sup_train_set['text'],opp_train_set['text']