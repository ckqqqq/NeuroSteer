# Imports and Setup
import argparse
import os
import logging
from typing import Tuple
import json
from log import setup_logging
from tqdm import tqdm  # For progress bars
import codecs
from collections import Counter
from typing import List, Dict
import re


import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np

# Define hyperparameters
task="sentiment"
if task=="sentiment":
    args_dict = {
        "layer": 6,  # Example layer number to analyze
        "LLM": "gpt2-small",
        "dataset_path": "/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5",
        "prompt_path":"/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k",
        "output_dir": "./results",
        "env_path": "/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env",
        "task":"sentiment",# “sentiment”,"cot","polite"
        "seed": 42,
        "data_size": 1000,
        "device": "cuda",  # Options: "cpu", "cuda", "mps", "auto"
        "alpha": 100, # 这个alpha后面慢慢调节
        "steer": "pos-neg",  # Options: "pos", "neg", "neu","pos-neg","cot-direct"
        "method": "val_mul",  # Options: "mean", "val_mul" 用val_mul会比较好
        "topk_mean": 100, # 选取前topk 个均值激活，这个效果一般，会导致很多如what？why？这种被激活
        "topk_cnt": 100, # 选取前topk个频率激活，目前默认这个，效果很好
        "batch_size": 32 # 这个好像没用上
    }

args = argparse.Namespace(**args_dict)

# Logging Setup
import os
from log import setup_logging
import logging
# Create output directory base path
output_dir_base = os.path.join(
    args.output_dir,
    f"LLM_{args.LLM}_layer_{args.layer}_steer_{args.steer}_alpha_{args.alpha}_cnt_{args.topk_cnt}_mean_{args.topk_mean}_device_{args.device}"
)

# Setup logging
setup_logging(output_dir_base)

# Save hyperparameters
hyperparams = args_dict

# Log hyperparameters
logging.info("Hyperparameters:")
for key, value in hyperparams.items():
    logging.info(f"  {key}: {value}")

# Load Environment Variables
 
def load_environment(env_path: str):
    load_dotenv(env_path)
    hf_endpoint = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
    logging.info(f"HF_ENDPOINT: {hf_endpoint}")

load_environment(args.env_path)

import re

def load_and_prepare_triple_dataset(dataset_path: str,dataset_name:str, seed: int, num_samples: int):
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
    neg_train_set = dataset['train'].filter(lambda example: example['label'] < neu_label).select(range(num_samples))
    pos_train_set = dataset['train'].filter(lambda example: example['label'] == neu_label).select(range(num_samples))
    neu_train_set = dataset['train'].filter(lambda example: example['label'] > neu_label ).select(range(num_samples))

    logging.info(f"Selected {len(neg_train_set)} negative, {len(pos_train_set)} positive, and {len(neu_train_set)} neutral samples")
    print(dataset)
    if dataset_name in ["sst5"]:
        val_set=dataset['validation']
    else:
        raise ValueError("没写呢")
    test_set=dataset["test"]
    return neg_train_set, pos_train_set, neu_train_set,val_set,test_set
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
    
# Load and Prepare Dataset

logging.info("dataset path "+args.dataset_path)
if "neg" in args.steer or "pos" in args.steer and args.steer=="sentiment":
    neg_train_set, pos_train_set, neu_train_set,val_set,test_set = load_and_prepare_triple_dataset(
        args.dataset_path, "sst5",args.seed, args.data_size
    )
elif "cot" in args.steer or "COT" in args.steer:
    logging.info("COT "*10)
    all_dataset=load_and_prepare_COT_dataset(
        args.dataset_path, args.seed, args.data_size
    )
elif "polite" in args.steer:
    logging.info("polite"*10)
    neg_train_set, pos_train_set, neu_train_set,val_set,test_set=load_and_prepare_triple_dataset(args.dataset_path,"polite", args.seed, args.data_size)
else:
    raise ValueError("No Supported")
assert neg_train_set[10]!=pos_train_set[10]


def compute_steering_vectors(sae: SAE, overlap_indices: Tensor, nz_mean: Tensor, method: str = "val_mul") -> Tensor:
    logging.info(f"Computing steering vectors using method: {method}")
    if method == "mean":
        steering_vectors = torch.mean(sae.W_dec[overlap_indices], dim=0)
    elif method == "val_mul":
        steering_vectors = torch.zeros(sae.W_dec.shape[1], device=sae.W_dec.device)
        for important_idx in overlap_indices:
            steering_vectors += nz_mean[important_idx].item() * sae.W_dec[important_idx]
    else:
        raise ValueError(f"Unknown method: {method}")
    logging.info(f"Steering vectors computed with shape: {steering_vectors.shape}")
    return steering_vectors


def save_results(output_dir: str, nz_mean: Tensor, act_cnt: Tensor, generated_texts: list, hyperparams: dict):
    os.makedirs(output_dir, exist_ok=True)
    # Save nz_mean and act_cnt
    nz_stats_path = os.path.join(output_dir, 'nz_stats.pt')
    logging.info(f"Saving nz_mean and act_cnt to {nz_stats_path}")
    torch.save({
        'nz_mean': nz_mean,
        'act_cnt': act_cnt,
    }, nz_stats_path)

    # Save generated texts
    generated_texts_path = os.path.join(output_dir, 'generated_texts.txt')
    logging.info(f"Saving generated texts to {generated_texts_path}")
    with open(generated_texts_path, 'w') as f:
        for text in generated_texts:
            f.write(text + "\n")

    # Save hyperparameters
    hyperparams_path = os.path.join(output_dir, 'hyperparameters.json')
    logging.info(f"Saving hyperparameters to {hyperparams_path}")
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)

    logging.info("All results saved successfully.")
    
output_dir_base = os.path.join(
    args.output_dir,
    f"mask_test/LLM_{args.LLM}_layer_{args.layer}_steer_{args.steer}_alpha_{args.alpha}_cnt_{args.topk_cnt}_mean{args.topk_mean}"
)

def load_from_cache():
    cache_exists = False
    cache_file = os.path.join(output_dir_base, 'hyperparameters.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        cached_hash = cached_data.get('hyperparams_hash')

    if cache_exists:
        # Load nz_mean and act_cnt from cache
        # nz_stats_path = os.path.join(output_dir_base, 'nz_stats.pt')
        # nz_act = torch.load(nz_stats_path)
        # nz_mean = nz_act['nz_mean']
        # act_cnt = nz_act['act_cnt']
        # overlap_indices = nz_act.get('overlap_indices', None)  # If overlap_indices was saved
        logging.info("load from cache")
    else:
        # overlap_indices = None  # Will be computed later
        logging.info("non cache: "+cache_file)
load_from_cache()


# Save hyperparameters
hyperparams = vars(args)

# Log hyperparameters
logging.info("Hyperparameters:")
for key, value in hyperparams.items():
    logging.info(f"  {key}: {value}")

# Load environment
load_environment(args.env_path)

# Load model and SAE
logging.info(f"Loading model: {args.LLM}")
model = HookedTransformer.from_pretrained(args.LLM, device=args.device)

logging.info(f"Loading SAE for layer {args.layer}")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id=f"blocks.{args.layer}.hook_resid_pre",
    device=args.device
)



def compute_latents(sae: SAE, model: HookedTransformer, texts: list, hook_point: str, device: str, batch_size: int) -> list:
    """
    计算 latents，支持批次处理。

    Args:
        sae (SAE): SAE 实例。
        model (HookedTransformer): Transformer 模型实例。
        texts (list): 文本列表。
        hook_point (str): 钩子点名称。
        device (str): 计算设备。
        batch_size (int): 每个批次的大小。

    Returns:
        list: 包含每个批次 latents 的张量列表。
    """
    SAE_LATENT_SIZE=sae.W_dec.shape[0]
    logging.info("Running model with cache to obtain hidden states")
    # batch_latents = []
    lat_freq,lat_val_sum=torch.zeros(SAE_LATENT_SIZE).to(device),torch.zeros(SAE_LATENT_SIZE).to(device)
    # 使用 tqdm 显示进度条
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        logging.info(f"Batch {i // batch_size + 1}: batch_size {batch_size}")
        sv_logits, cache = model.run_with_cache(batch_texts, prepend_bos=False, device=device)
        batch_hidden_states = cache[hook_point]
        logging.info(f"Batch {i // batch_size + 1}: Hidden states shape: {batch_hidden_states.shape}")

        # logging.info(f"Encoding hidden states for batch {i // batch_size + 1}")
        # 假设 sae.encode 支持批量编码
        batch_latents = sae.encode(batch_hidden_states)  # 形状: (batch_size, latent_dim)
        batch_info=analyze_latents(batch_latents)
        lat_freq=lat_freq+batch_info["latent_frequency"].clone()
        lat_val_sum=lat_val_sum+batch_info["latent_value_sum"].clone()
    lat_val_mean=torch.where(lat_freq != 0, lat_val_sum / lat_freq, torch.tensor(0.0, device=batch_latents.device))
    logging.info(f"Total non-zero element shape: {lat_freq.shape}")
    assert lat_freq.shape[0]==lat_freq.shape[0]==sae.W_dec.shape[0], "sae latent dimension mismatch"
    return {"latent_frequency":lat_freq,"latent_value_mean":lat_val_mean}
# %%
# args.steer

def analyze_latents(batch_latents: Tensor, top_k_mean: int = 100, top_k_cnt: int = 100) -> Tuple[Tensor, Tensor, Tensor]:
    SAE_LATENT_SIZE=sae.W_dec.shape[0]
    
    # 计算非0激活在对应位置的激活频率   
    logging.info("Computing non-zero element counts") 
    lat_freq = (batch_latents != 0).sum(dim=(0, 1))
    # 计算非0激活在对应位置的激活值的和
    logging.info("Computing sum of non-zero elements")
    lat_val_sum = batch_latents.sum(dim=(0, 1))

    logging.info("Computing mean of non-zero elements")
    # 
    assert batch_latents.shape[-1]==SAE_LATENT_SIZE==lat_val_sum.shape[0], "Latent dimension mismatch"
    return {"latent_frequency":lat_freq,"latent_value_sum":lat_val_sum}


def get_activation_by_steer(texts:list):

    hook_point = sae.cfg.hook_name

    # Compute latents with batch processing
    lat_info=compute_latents(sae, model, texts, hook_point, args.device, args.batch_size)
    # act_cnt=
    # nz_mean=
    # # 计算第二个维度的最大值
    # max_dim1 = max(latent.shape[1] for latent in batch_latents)  # 第二个维度的最大值
    # logging.info(f"最大长度:{max_dim1}")
    # # 对每个 Tensor 进行填充（仅填充第二个维度）
    # padded_latents_right = [
    #     torch.nn.functional.pad(latent, (0, 0, 0, max_dim1 - latent.size(1)), "constant", 0)
    #     for latent in batch_latents
    # ]

    # batch_latents_concatenated = torch.cat(padded_latents_right, dim=0)
    # logging.info(f"Concatenated batch latents shape: {batch_latents_concatenated.shape}")

    # Analyze latents 
    # nz_mean, act_cnt, _ = analyze_latents(batch_latents_concatenated, top_k_mean=args.topk_mean, top_k_cnt=args.topk_cnt)
    return {"latent_value_mean":lat_info["latent_value_mean"],"latent_frequency":lat_info["latent_frequency"]}


args.steer=args.steer.lower()

steer_info={}
if args.steer=='polite-impolite' or args.steer=="pos-neg":
    logging.info(args.steer)
    text=pos_train_set["text"][:args.data_size]
    steer_info["pos"]=get_activation_by_steer(text)
    text=neg_train_set["text"][:args.data_size]
    steer_info["neg"]=get_activation_by_steer(text)
    text=neu_train_set["text"][:args.data_size]
    steer_info["neu"]=get_activation_by_steer(text)
    

assert bool(torch.all((steer_info["pos"]["latent_value_mean"]-steer_info["neg"]["latent_value_mean"])==0))==False,"数据库读取有问题"


source="neg"
target="pos"
# 调整样本正负性在这里调整 从负样本到正样本还是从正样本()到负样本
# 正
steer_info[f"dif_{target}-{source}_relu"]={"latent_frequency":torch.relu(steer_info[target]["latent_frequency"]-steer_info[source]["latent_frequency"]),"latent_value_mean":torch.relu(steer_info[target]["latent_value_mean"]-steer_info[source]["latent_value_mean"])}
top_k=100
_,steer_indices=torch.topk(steer_info[f"dif_{target}-{source}_relu"]["latent_frequency"],top_k)
steer_indices


def random_half_indices(steer_indices_remain):
    permutation = torch.randperm(steer_indices_remain.size(0))
    shuffled_tensor = steer_indices_remain[permutation]
    half_size = steer_indices_remain.size(0) // 2
    return shuffled_tensor[half_size:], shuffled_tensor[:half_size]


# 假设 steer_info[f"dif_{b}-{a}_relu"]["latent_frequency"] 是一个 NumPy 数组
nz_cnt = steer_info[f"dif_{target}-{source}_relu"]["latent_frequency"]

# 先获取非零元素的索引
nz_indices = np.nonzero(nz_cnt)

def compute_steering_vectors(sae: SAE, indices: Tensor, nz_mean_val: Tensor, method: str = "val_mul") -> Tensor:
    logging.info(f"Computing steering vectors using method: {method}")
    if method == "mean":
        steering_vectors = torch.mean(sae.W_dec[indices], dim=0)
    elif method == "val_mul":
        steering_vectors = torch.zeros(sae.W_dec.shape[1], device=sae.W_dec.device)
        for idx in indices:
            steering_vectors += nz_mean_val[idx].item() * sae.W_dec[idx]
    else:
        raise ValueError(f"Unknown method: {method}")
    logging.info(f"Steering vectors computed with shape: {steering_vectors.shape}")
    return steering_vectors

def steering_hook(resid_pre, hook,steer_type="last", steering_on=True, alpha=0, delta_matrix=None):
    if resid_pre.shape[1] == 1:
        return
    if steering_on:
        if steer_type=="last":
            resid_pre[:, :-1, :] += alpha * delta_matrix
            # count_steering+=1
            # steer_cnt+=1
            # logging.info("干预"+str(steer_cnt)+"次")
            
        # elif steer_type=="last2":
        #     resid_pre[:, :-2, :] += args.alpha * steering_vectors
        # elif steer_type=="gaussian":
        #     # 高斯卷积的方式放缩干预矩阵，
        #     # 这里需要一个高斯核，然后对steering_vectors进行卷积
        #     gaussian_kernel = torch.tensor([1,2,1])
        #     steering_vectors = torch.conv1d(steering_vectors, gaussian_kernel, padding=1)
        #     resid_pre[:, :-1, :] += args.alpha * steering_vectors
        # else:
        #     raise ValueError("Unknown steering type")

        # 修改这里的干预方式，增加干预的选择，例如从倒数第一个token开始干预，或者从倒数第二个token开始干预，或者使用高斯卷积的方式放缩干预矩阵
        
        # 这里是干预调整的关键，很有意思的是，如果提前干预效果会更好更连贯，同样加上正弦波之类的效果也很棒
        #注意这里是对batch_size,0:-9,hidden_size这样的隐藏层做干预

def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch,prepend_bos=False)
        result = model.generate(
            stop_at_eos=True,  # avoids a bug on MPS
            input=tokenized,
            max_new_tokens=100,
            do_sample=True,
            **kwargs,
        )
    return result

def run_generate(example_prompt, sampling_kwargs, show_res=False, steering_on=False, alpha=0, delta_matrix=None):
    model.reset_hooks()
    editing_hooks = [(f"blocks.{args.layer}.hook_resid_post", lambda resid_pre, hook: steering_hook(resid_pre, hook, steer_type="last", steering_on=steering_on, alpha=alpha, delta_matrix=delta_matrix))]

    res = hooked_generate(
        example_prompt, editing_hooks, seed=None, **sampling_kwargs
    )

    # Print results, removing the ugly beginning of sequence token
    res_str = model.to_string(res[:, :])
    # print(("\n\n" + "-" * 80 + "\n\n").join(res_str))
    # generated_texts = res_str
    if show_res:
        for idx, text in enumerate(res_str):
            logging.info(f"Generated Text: {idx+1}:\n{text}\n{'-'*80}")
        
    return res_str

def load_positive_lexicon(lexicon_file: str) -> Dict[str, float]:
    """
    读取 VADER 词典文件，提取所有正向情感词汇及其情感得分。
    :param lexicon_file: VADER 词典文件的路径
    :return: 一个字典，键为正向情感词汇，值为其情感得分
    """
    positive_lexicon = {}
    if not os.path.isfile(lexicon_file):
        raise FileNotFoundError(f"The lexicon file '{lexicon_file}' does not exist.")
    with codecs.open(lexicon_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            parts = line.split('\t')
            if len(parts) < 2:
                continue  # 确保至少有两个字段
            word = parts[0].lower()  # 转为小写以便后续匹配
            try:
                sentiment_score = float(parts[1])
            except ValueError:
                continue  # 跳过无法转换为浮点数的行
            if sentiment_score > 0:
                positive_lexicon[word] = sentiment_score
    return positive_lexicon

def simple_tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())

def get_top_positive_words(word_lists: List[str], positive_lexicon: Dict[str, float], n: int) -> Dict[str, int]:
    """
    统计输入字符串列表中所有正向情感词语的词频，并返回词频最高的前 n 个。
    :param word_lists: 一个由字符串组成的列表，表示要分析的文本
    :param positive_lexicon: 正向情感词汇及其情感得分的字典
    :param n: 要返回的前 n 个词语
    :return: 一个字典，包含前 n 个正向情感词语及其出现次数
    """
    word_counter = Counter()
    for text in word_lists:
        words = simple_tokenize(text)
        positive_words = [word for word in words if word in positive_lexicon]
        word_counter.update(positive_words)
    top_n = word_counter.most_common(n)
    top_n_dict = {word: count for word, count in top_n}
    return top_n_dict


# Define sampling parameters
sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)
# Example prompt from the selected set
example_prompt = "Aha, we have good result!"
logging.info(f"Example prompt: {example_prompt}")

def load_and_prepare_sentiment_prompts():
    data_files = {"neg": "negative_prompts.jsonl", "pos": "positive_prompts.jsonl","neu":"neutral_prompts.jsonl"}
    prompts= load_dataset("/home/ckqsudo/code2024/0refer_ACL/LM-Steer/data/data/prompts/sentiment_prompts-10k",data_files=data_files)
    print(prompts)
    return prompts
prompts=load_and_prepare_sentiment_prompts()


lexicon_path = '/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/explanation/vader_lexicon.txt'
positive_lexicon = load_positive_lexicon(lexicon_path)

senti_prompt = []
prompt_idx = 0
for i in range(2):
    new_batch = []
    for k in range(50):
        # 用neutral的prompt去生成positive的text，感觉会更有解释性
        new_batch.append(prompts['neu'][prompt_idx]['prompt']['text'])    
        prompt_idx += 1
    senti_prompt.append(new_batch)


# current_indices_list = [steer_indices]
# for i in range(3):
#     next_indices_list = []
#     for remain_indices in current_indices_list:
#         l_steer_indices, r_steer_indices = random_half_indices(remain_indices)
#         next_indices_list.extend([l_steer_indices, r_steer_indices])
#     current_indices_list = next_indices_list

bianli_indices = [torch.tensor([indice]) for indice in steer_indices.tolist()]
baseline_text_wo_steer = []
for batch_prompt in senti_prompt:
    generated_texts_wo_steer = run_generate(batch_prompt, sampling_kwargs, show_res=False, steering_on=False, alpha=0, delta_matrix=None)
    baseline_text_wo_steer.extend(generated_texts_wo_steer)
wo_steer_top50_positive_sentiment_words = get_top_positive_words(baseline_text_wo_steer, positive_lexicon, 50)

final_generated_texts = [{"steer_indices": [], "generated_texts": baseline_text_wo_steer, 'top_50_positive_sentiment_words': wo_steer_top50_positive_sentiment_words}]
for current_indices in bianli_indices:
    current_delta_matrix = compute_steering_vectors(sae, indices=current_indices, nz_mean_val=steer_info[f"dif_{target}-{source}_relu"]["latent_value_mean"], method="val_mul")
    current_all_generated_texts = []
    for batch_prompt in senti_prompt:
        generated_texts_with_current_steer = run_generate(batch_prompt, sampling_kwargs, show_res=False, steering_on=True, alpha=100, delta_matrix=current_delta_matrix)
        current_all_generated_texts.extend(generated_texts_with_current_steer)
    top50_positive_sentiment_words =  get_top_positive_words(current_all_generated_texts, positive_lexicon, 50)
    for word, count in top50_positive_sentiment_words.items():
        if word in wo_steer_top50_positive_sentiment_words:
            top50_positive_sentiment_words[word] = count - wo_steer_top50_positive_sentiment_words[word]
        else:
            top50_positive_sentiment_words[word] = count
    final_generated_texts.append({"steer_indices": current_indices.tolist(), "generated_texts": current_all_generated_texts, 'top_50_positive_sentiment_words': top50_positive_sentiment_words})

with open("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/explanation/result/generated_texts_neu2pos_cuda_bianliquzao.json", "w", encoding="utf-8") as f:
    json.dump(final_generated_texts, f, ensure_ascii=False, indent=4)

