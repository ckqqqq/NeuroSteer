# %%
# Imports and Setup
import argparse
import os
import logging
from typing import Tuple
import json
from log import setup_logging
from tqdm import tqdm  # For progress bars


# %%

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from sae_lens import SAE

from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
# import plotly_express as px

# %% [markdown]
# # 进行基础情感干预实验

# %%


# %%
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
        "device": "cpu",  # Options: "cpu", "cuda", "mps", "auto"
        "alpha": 100, # 这个alpha后面慢慢调节
        "steer": "neg-pos",  # Options: "pos", "neg", "neu","pos-neg","cot-direct"
        "method": "val_mul",  # Options: "mean", "val_mul" 用val_mul会比较好
        "topk_mean": 100, # 选取前topk 个均值激活，这个效果一般，会导致很多如what？why？这种被激活
        "topk_cnt": 100, # 选取前topk个频率激活，目前默认这个，效果很好
        "batch_size": 32 # 这个好像没用上
    }

# %% [markdown]
# # 进行COT相关实验

# %%
# Define hyperparameters
if task=="cot":
    args_dict = {
        "layer": 6,  # Example layer number to analyze
        "LLM": "gpt2-small",
        "dataset_path": "/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/math/COT_GSM8k",
        "output_dir": "./results",
        "env_path": "/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env",
        "seed": 42,
        "data_size": 1000,
        "device": "cpu",  # Options: "cpu", "cuda", "mps", "auto"
        "alpha": 100,
        "steer": "cot-direct",  # Options: "pos", "neg", "neu","pos-neg","cot-direct"
        "method": "val_mul",  # Options: "mean", "val_mul"
        "topk_mean": 100,
        "topk_cnt": 100,
        "batch_size": 32
    }

# %% [markdown]
# # 进行礼貌实验
# /home/ckqsudo/code2024/0dataset/ACL_useful_dataset/style_transfer/politeness-corpus

# %%
# Define hyperparameters
if task=="polite":
    args_dict = {
        "layer": 6,  # Example layer number to analyze
        "LLM": "gpt2-small",
        "dataset_path": "/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/style_transfer/politeness-corpus",
        "output_dir": "./results",
        "env_path": "/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env",
        "seed": 42,
        "data_size": 1000,
        "device": "cpu",  # Options: "cpu", "cuda", "mps", "auto"
        "alpha": 100,
        "steer": "polite-impolite",  # Options: "pos", "neg", "neu","pos-neg","cot-direct"
        "method": "val_mul",  # Options: "mean", "val_mul"
        "topk_mean": 100,
        "topk_cnt": 100,
        "batch_size": 32
    }

# %%
# 进行礼貌实验

# %%
# Configuration and Hyperparameters
# 将字典转换为 argparse.Namespace 对象
args = argparse.Namespace(**args_dict)
# 测试访问属性
print(args.layer)  # 输出: 10
print(args.LLM)  # 输出: gpt2-small
print(args.output_dir)  # 输出: ./results
print(args.steer)

# %%
# Logging Setup
import os
from log import setup_logging
import logging
# Create output directory base path
output_dir_base = os.path.join(
    args.output_dir,
    f"LLM_{args.LLM}_layer_{args.layer}_steer_{args.steer}_alpha_{args.alpha}_{task}_cnt_{args.topk_cnt}_mean{args.topk_mean}"
)

# Setup logging
setup_logging(output_dir_base)

# Save hyperparameters
hyperparams = args_dict

# Log hyperparameters
logging.info("Hyperparameters:")
for key, value in hyperparams.items():
    logging.info(f"  {key}: {value}")


# %%
# Load Environment Variables
def load_environment(env_path: str):
    load_dotenv(env_path)
    hf_endpoint = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
    logging.info(f"HF_ENDPOINT: {hf_endpoint}")
load_environment(args.env_path)


# %%
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
    

# %%
args.steer

# %%
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


# %%
assert neg_train_set[10]!=pos_train_set[10]

# %%
pos_train_set[10],neg_train_set[10]

# %%


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
    logging.info("Running model with cache to obtain hidden states")
    batch_latents = []

    # 使用 tqdm 显示进度条
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        sv_logits, cache = model.run_with_cache(batch_texts, prepend_bos=False, device=device)
        batch_hidden_states = cache[hook_point]
        logging.info(f"Batch {i // batch_size + 1}: Hidden states shape: {batch_hidden_states.shape}")

        logging.info(f"Encoding hidden states for batch {i // batch_size + 1}")
        # 假设 sae.encode 支持批量编码
        latents = sae.encode(batch_hidden_states)  # 形状: (batch_size, latent_dim)
        batch_latents.append(latents)
        

    logging.info(f"Total batches processed: {len(batch_latents)}")
    return batch_latents

# %%


# %%



# %%

# def save_results(output_dir: str, nz_mean: Tensor, act_cnt: Tensor, generated_texts: list, hyperparams: dict):
#     os.makedirs(output_dir, exist_ok=True)
#     # Save nz_mean and act_cnt
#     nz_stats_path = os.path.join(output_dir, 'nz_stats.pt')
#     logging.info(f"Saving nz_mean and act_cnt to {nz_stats_path}")
#     torch.save({
#         'nz_mean': nz_mean,
#         'act_cnt': act_cnt,
#     }, nz_stats_path)

#     # Save generated texts
#     generated_texts_path = os.path.join(output_dir, 'generated_texts.txt')
#     logging.info(f"Saving generated texts to {generated_texts_path}")
#     with open(generated_texts_path, 'w') as f:
#         for text in generated_texts:
#             f.write(text + "\n")

#     # Save hyperparameters
#     hyperparams_path = os.path.join(output_dir, 'hyperparameters.json')
#     logging.info(f"Saving hyperparameters to {hyperparams_path}")
#     with open(hyperparams_path, 'w') as f:
#         json.dump(hyperparams, f, indent=4)

#     logging.info("All results saved successfully.")

# %%
output_dir_base = os.path.join(
    args.output_dir,
    f"LLM_{args.LLM}_layer_{args.layer}_steer_{args.steer}_alpha_{args.alpha}_cnt_{args.topk_cnt}_mean{args.topk_mean}"
)
output_dir_base

# %%
# def load_from_cache():
#     cache_exists = False
#     cache_file = os.path.join(output_dir_base, 'hyperparameters.json')
#     if os.path.exists(cache_file):
#         with open(cache_file, 'r') as f:
#             cached_data = json.load(f)
#         cached_hash = cached_data.get('hyperparams_hash')

#     if cache_exists:
#         # Load nz_mean and act_cnt from cache
#         # nz_stats_path = os.path.join(output_dir_base, 'nz_stats.pt')
#         # nz_act = torch.load(nz_stats_path)
#         # nz_mean = nz_act['nz_mean']
#         # act_cnt = nz_act['act_cnt']
#         # overlap_indices = nz_act.get('overlap_indices', None)  # If overlap_indices was saved
#         logging.info("load from cache")
#     else:
#         # overlap_indices = None  # Will be computed later
#         logging.info("non cache: "+cache_file)
# load_from_cache()

# %%
# setup_logging(output_dir_base)

# %%
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

# %%
# # Load dataset
# all_dataset = load_and_prepare_triple_dataset(
#     args.dataset_path, args.seed, args.data_size
# )

# %%
args.steer

# %%


# %%

def analyze_latents(batch_latents: Tensor, top_k_mean: int = 100, top_k_cnt: int = 100) -> Tuple[Tensor, Tensor, Tensor]:
    logging.info("Computing non-zero element counts")
    act_cnt = (batch_latents != 0).sum(dim=(0, 1))

    logging.info("Computing sum of non-zero elements")
    nz_sum = torch.where(batch_latents != 0, batch_latents, torch.tensor(0.0, device=batch_latents.device)).sum(dim=(0, 1))

    logging.info("Computing mean of non-zero elements")
    nz_mean = torch.where(act_cnt != 0, nz_sum / act_cnt, torch.tensor(0.0, device=batch_latents.device))

    logging.info("Selecting top-k indices based on nz_mean")
    nz_act_val, nz_val_indices = torch.topk(nz_mean, top_k_mean)
    logging.info(f"Top {top_k_mean} nz_mean values selected.")

    logging.info("Selecting top-k indices based on act_cnt")
    nz_cnt, cnt_indices = torch.topk(act_cnt, top_k_cnt)
    logging.info(f"Top {top_k_cnt} act_cnt values selected.")

    # logging.info("Finding overlapping indices between nz_mean and act_cnt top-k")
    # overlap_mask = torch.isin(nz_val_indices, cnt_indices)
    # overlap_indices = nz_val_indices[overlap_mask]
    # logging.info(f"Number of overlapping indices: {len(overlap_indices)}")
    # overlap_indices=overlap_indices
    return nz_mean, act_cnt,None

# %%
# Select a dataset steer based on steering preference

def get_activation_by_steer(texts:list):

    hook_point = sae.cfg.hook_name

    # Compute latents with batch processing
    batch_latents = compute_latents(sae, model, texts, hook_point, args.device, args.batch_size)
    # 计算第二个维度的最大值
    max_dim1 = max(latent.shape[1] for latent in batch_latents)  # 第二个维度的最大值
    logging.info(f"最大长度:{max_dim1}")
    # 对每个 Tensor 进行填充（仅填充第二个维度）
    padded_latents_right = [
        torch.nn.functional.pad(latent, (0, 0, 0, max_dim1 - latent.size(1)), "constant", 0)
        for latent in batch_latents
    ]

    batch_latents_concatenated = torch.cat(padded_latents_right, dim=0)
    logging.info(f"Concatenated batch latents shape: {batch_latents_concatenated.shape}")

    # Analyze latents 
    nz_mean, act_cnt, _ = analyze_latents(batch_latents_concatenated, top_k_mean=args.topk_mean, top_k_cnt=args.topk_cnt)
    return {"nz_mean":nz_mean,"nz_cnt":act_cnt}

# %%
args.steer=args.steer.lower()

# %%
# all_dataset["train"]["Q+A"][:args.data_size][193]

# %% [markdown]
# ## 26000(SAE稀疏神经元)对应的非零激活神经元激活统计信息，和激活值统计信息
# 

# %%
steer_info={}

# %%
task

# %%
steer_info={}
if args.steer=='polite-impolite' or task=="sentiment":
    logging.info(args.steer)
    text=pos_train_set["text"][:args.data_size]
    steer_info["pos"]=get_activation_by_steer(text)
    text=neg_train_set["text"][:args.data_size]
    steer_info["neg"]=get_activation_by_steer(text)
    text=neu_train_set["text"][:args.data_size]
    steer_info["neu"]=get_activation_by_steer(text)

# %%

if "cot" in args.steer:
    if args.steer in "cot-direct":
        texts=all_dataset["train"]["Q+A"][:args.data_size]
        print(type(texts))
        steer_info["direct"]=get_activation_by_steer(texts)
        
        texts=all_dataset["train"]["Q+COT_A"][:args.data_size]
        print(type(texts))
        steer_info["cot"]=get_activation_by_steer(texts)
        # print(texts[123])
    else:
        raise ValueError("????")

# %%
args.steer

# %%
# pos_train_set["text"][:args.data_size]

# %%
len(pos_train_set["text"][:args.data_size])

# %%
args.data_size

# %%

# for steer_key in ["pos","neu","neg"]:
#     if steer_key == "pos":
#         selected_set = pos_train_set
#     elif steer_key == "neg":
#         selected_set = neg_train_set
#     elif steer_key=="neu":
#         selected_set = neu_train_set

#     texts = selected_set["text"][:args.data_size]
#     a=get_activation_by_steer(texts)
#     steer_info[steer_key]=a


# %%
# steer_info["dif_neg_pos"]={"steer":"dif_neg_pos","nz_cnt":steer_info["pos"]["nz_cnt"]-steer_info["neg"]["nz_cnt"],"nz_mean":steer_info["pos"]["nz_mean"]-steer_info["neg"]["nz_mean"]}

# %%
# steer_info["dif_neg_pos_relu"]={"nz_cnt":torch.relu(steer_info["pos"]["nz_cnt"]-steer_info["neg"]["nz_cnt"]),"nz_mean":torch.relu(steer_info["pos"]["nz_mean"]-steer_info["neg"]["nz_mean"])}

# %%
# steer_info["dif_neg_pos_relu"],steer_info["dif_neg_pos"]

# %%
sourceource="cot"
target="direct"

# %%
source="pos"
target="neg"
# 调整样本正负性在这里调整 从负样本到正样本还是从正样本()到负样本
# pos 代表积极情绪
# neg 代表消极情绪


# %%
assert bool(torch.all((steer_info["pos"]["nz_mean"]-steer_info["neg"]["nz_mean"])==0))==False,"数据库读取有问题"

# %%
assert torch.all(steer_info[target]["nz_mean"]>=0),"所有SAE的激活需要大于d等于0（maybe）"

# %% [markdown]
# # 泽凯在这里做mask

# %%
# 
steer_info[f"dif_{target}-{source}_relu"]={"nz_cnt":torch.relu(steer_info[target]["nz_cnt"]-steer_info[source]["nz_cnt"]),"nz_mean":torch.relu(steer_info[target]["nz_mean"]-steer_info[source]["nz_mean"]),"target_nz_mean":torch.relu(steer_info[target]["nz_mean"])}
"""
nz_cnt: 神经元被激活的次数
nz_mean: 神经元被激活后的平均值
nz_mean_pos: 正样本神经元被激活后的平均值
"""
top_k=100
_,steer_indices=torch.topk(steer_info[f"dif_{target}-{source}_relu"]["nz_cnt"],top_k)

# %%

# steer_info[f"dif_{a}-{b}_relu"]={"nz_cnt":torch.relu(steer_info[a]["nz_cnt"]-steer_info[b]["nz_cnt"]),"nz_mean":torch.relu(steer_info[a]["nz_mean"]-steer_info[b]["nz_mean"])}
# top_k=100
# steering_vectors,steer_indices=torch.topk(steer_info[f"dif_{a}-{b}_relu"]["nz_cnt"],top_k)

# %%
# 假设 steer_info[f"dif_{b}-{a}_relu"]["nz_cnt"] 是一个 NumPy 数组
nz_cnt = steer_info[f"dif_{target}-{source}_relu"]["nz_cnt"]

# 先获取非零元素的索引
nz_indices = np.nonzero(nz_cnt)
torch.all(nz_cnt == 0)

# %%
steer_info[f"dif_{target}-{source}_relu"]["nz_cnt"].shape

# %%
_,steer_indices,

# %%
steer_indices

# %%
steer_info[f"dif_{target}-{source}_relu"]["nz_mean"][steer_indices]# 这里有0,没有负数比较正常


# %%
# steer_info["dif_-pos"]["nz_mean"][steer_indices]

# %%
steer_indices

# %%
# mean_type="dif_mean"
mean_type="dif_mean"
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
if mean_type=="dif_mean":
    delta_matrix=compute_steering_vectors(sae,indices=steer_indices,nz_mean_val=steer_info[f"dif_{target}-{source}_relu"]["nz_mean"],method="val_mul")
elif mean_type=="tar_mean":
    delta_matrix=compute_steering_vectors(sae,indices=steer_indices,nz_mean_val=steer_info[f"dif_{target}-{source}_relu"]["target_nz_mean"],method="val_mul")
else:
    raise ValueError("Unsupported")

# %%
# model.to_tokens("<|endoftext|>")

# %%
model.tokenizer.eos_token

# %%
model.tokenizer

# %%
delta_matrix #理论上这里有正有负比较正常

# %%


# %% [markdown]
# # 这里得到的就是delta_matricx

# %%
sae.cfg.hook_name

# %%
f"blocks.{args.layer}.hook_resid_post"

# %%
import torch.nn.functional as F
import torch

def half_gaussian_kernel(half_len):
    # 设置均值和标准差
    cov_len=2*half_len
    mean = cov_len // 2  # 正态分布的均值
    std = cov_len / 6    # 设置标准差，可以根据需要调整

    # 创建正态分布
    x = torch.arange(cov_len, dtype=torch.float32)
    kernel = torch.exp(-0.5 * ((x - mean) / std) ** 2)
    # print(kernel)

    # 仅保留正态分布的前半部分（右侧值设置为0）
    kernel[int(cov_len // 2):] = 0  # 保留前半部分，右半部分置为零

    # 归一化，确保总和为 1
    kernel = kernel / kernel.sum()
    return kernel[:half_len]

gauss=half_gaussian_kernel(4)
gauss
# k_gau=torch.cat([gauss, torch.tensor([0])])

# %%
gauss

# %%
# import einops
# test=torch.ones(2, 6,3)
# # test=einops.rearrange(test,"b s d->b d s")
# test.shape,k_gau.shape

# %%
# re_gau=einops.repeat(k_gau,"s -> b s h",b=2,h=3)

# %%
# test*re_gau,test

# %%
# # Define steering hook
# steering_on = True  # This will be toggled in run_generate
# alpha = args.alpha
# method = args.method  # Store method for clarity
from functools import partial
import einops
steer_cnt=0


def steering_hook(resid_pre, hook, steer_on, alpha, steer_type="last"):
    # 如果 seq_len 只有 1，则直接返回，不进行操作

    if resid_pre.shape[1] == 1:
        return
    # 判断是否进行干预
    if steer_on:
        if steer_type == "last":
            # 对最后一个token前的部分应用干预，使用给定的 delta_matrix
            
            resid_pre[:, :-1, :] += alpha * delta_matrix
            
            # logging.info(f"干预类型：last")
            # d_m_repeat=einops.repeat(d_m,"h -> b s h",b=b,s=s)
            # logging.info(f"干预矩阵: {alpha * d_m_repeat}")
        elif steer_type == "gaussian":
            # 使用高斯卷积对输入进行干预
            # s_idx=-1
            d_m=torch.clone(delta_matrix)
            s = resid_pre[:, :-1, :].shape[1]
            b=resid_pre[:, :-1, :].shape[0]
            h=resid_pre[:, :-1, :].shape[2]
            h_gauss = half_gaussian_kernel(s)  # 获取高斯卷积核
            # k_gauss=torch.cat([h_gauss, torch.tensor([0])])
            k_gauss=h_gauss
            k_gau_repeat=einops.repeat(k_gauss,"s -> b s h",b=b,h=h)
            d_m_repeat=einops.repeat(d_m,"h -> b s h",b=b,s=s)
            # 根据卷积结果更新 resid_pre（注意：保留其他维度不变）,逐一元素相乘
            resid_pre[:, :-1, :] += alpha * d_m_repeat* k_gau_repeat
            # logging.info(f"干预类型：高斯")
            # logging.info(f"干预矩阵: {alpha * d_m_repeat* k_gau_repeat}")
        else:
            raise ValueError("Unknown steering type")


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
        # 修改这里的干预方式，增加干预的选择，例如从倒数第一个token开始干预，或者从倒数第二个token开始干预，或者使用高斯卷积的方式放缩干预矩阵，这里是干预调整的关键，很有意思的是，如果提前干预效果会更好更连贯，还没尝试高斯卷积的方法

def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch,prepend_bos=False)
        result = model.generate(
            stop_at_eos=True,  # avoids a bug on MPS
            input=tokenized,
            max_new_tokens=50,
            do_sample=True,
            **kwargs,
        )
    return result

def run_generate(example_prompt,sampling_kwargs,steer_on,alpha,steer_type="last",repeat_num=3,show_res=False):
    model.reset_hooks()
    if steer_on:
        steering_hook_fn=partial(steering_hook,steer_on=steer_on,alpha=alpha,steer_type=steer_type)
        editing_hooks = [(f"blocks.{args.layer}.hook_resid_post", steering_hook_fn)]
    else:
        editing_hooks=[]
    res = hooked_generate(
        [example_prompt] * repeat_num, editing_hooks, seed=None, **sampling_kwargs
    )

    # Print results, removing the ugly beginning of sequence token
    res_str = model.to_string(res[:, :])
    # print(("\n\n" + "-" * 80 + "\n\n").join(res_str))
    # generated_texts = res_str
    if show_res:
        for idx, text in enumerate(res_str):
            logging.info(f"Generated Text: {idx+1}:\n{text}")
        
    return res_str

# Define sampling parameters
sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)

# Example prompt from the selected set
example_prompt = "What really matters is that they know"
# example_prompt=model.tokenizer
# print( all_dataset["val"]["prompt"][:3],all_dataset["val"]["A"][:3])
logging.info(f"Example prompt: {example_prompt}")

# Generate without steering
steer_on = False
alpha = 0
logging.info("Generating texts **without** steering... ")
generated_texts_no_steer = run_generate(example_prompt, sampling_kwargs,steer_on=steer_on,alpha=alpha,show_res=True)
logging.info("干预之后的结果")
# bef,aft=args.steer.split("-")
logging.info(f"干预方向{source}->{target},礼貌任务下，neg=impolite，情感任务下 pos=积极情感")
# Generate with steering
steer_on = True
alpha = 300
# alpha=args.aplha
logging.info("** Generating texts with steering... Target **")
generated_texts_with_steer = run_generate(
    example_prompt, 
    sampling_kwargs,
    steer_on=steer_on,
    alpha=alpha,
    steer_type="last",
    show_res=True)

# Combine generated texts
# all_generated_texts = generated_texts_no_steer + generated_texts_with_steer



# %% [markdown]
# # 理论上来讲，
# * 不礼貌的输出应该有很多疑问句？例如what？ ha？why？
# * 而礼貌的输出应该有很多正常的词语
# * 积极情感和不积极情感同理
# * 从目前的实验来看，负向情感干预+礼貌情感干预表现比较好，可以拿这个做可解释性
# * 频率很重要，我选取的latents选了前100频次的激活神经元
# * 如果对[0:-1]的区间进行干预，效果异常优秀，生成比较连贯，但是如果对[-1:length]的区间进行干预，效果就很差，生成的词语很零碎

# %% [markdown]
# # 下面进行的是扭转实验，使用prompt对模型进行诱导，再进行转向

# %%
args.prompt_path

# %%
import os
def load_and_prepare_sentiment_prompts(prompt_path:str,seed:int,num_samples:int):
    logging.info(f"Loading prompt_path from {prompt_path}")
    data_files = {"neg": "negative_prompts.jsonl", "pos": "positive_prompts.jsonl","neu":"neutral_prompts.jsonl"}
    
    prompts= load_dataset("/home/ckqsudo/code2024/0refer_ACL/LM-Steer/data/data/prompts/sentiment_prompts-10k",data_files=data_files)
    print(prompts)
    return prompts
prompts=load_and_prepare_sentiment_prompts(args.prompt_path,args.seed,1000)

# %%
# prompts["pos"]

# %%
# prompts["pos"][0]

# %%
target

# %%
sampling_kwargs

# %%
# # 遍历数据集
# for idx,example in enumerate(list(prompts["pos"])[:10]):
#     print(idx,example)  # example 是一个字典

# %%

#     # 转换为 Pandas DataFrame
# df = prompts["pos"].to_pandas()

# # 遍历 DataFrame
# for row in df.iterrows():
#     # print(row.to_dict())  # 将每一行转换为字典

# %%
prompts["pos"][1:4]


# %%
args.data_size

# %%

alpha

# %%


import copy
# Example prompt from the selected set
import jsonlines
res=[]
params={}
params["params"]={**vars(args),**sampling_kwargs,"max_new_tokens":50,"steer":f"from {source} to {target}"}
params["alpha"]=alpha
logging.info(f"Running with alpha: {alpha}")
logging.info(f"Running with prompt_type: "+str(params["params"]["steer"]))
# res.append(params)

no_steer_res=[]
steer_res=[]

pos_or_neg="pos"
assert pos_or_neg!=target,"prompt和转向的方向是一致的"
# 打开文件（模式为追加模式 'a'）
senti_gen_dir="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/evaluations/gen_files"
senti_gen_dir=os.path.join(senti_gen_dir,f"alpha_{alpha}_senti_{pos_or_neg}_datasize_{args.data_size}_layer{args.layer}_meantype_{mean_type}")
os.makedirs(senti_gen_dir,exist_ok=True)
with jsonlines.open(os.path.join(senti_gen_dir,"params.jsonl"), mode='w') as writer:
    writer.write(params)  # 逐行写入

show_compared=False
with jsonlines.open(os.path.join(senti_gen_dir,"no_steer_gen_res.jsonl"), mode='a') as nt_file:
    with jsonlines.open(os.path.join(senti_gen_dir,"steer_gen_res.jsonl"), mode='a') as t_file: 
        for idx,item in tqdm(enumerate(list(prompts[pos_or_neg])[:1000])):
            prompt=item["prompt"]["text"]
            item["label"]=pos_or_neg

            # 没转向的结果
            if show_compared:
                no_steer_gen_texts = run_generate(
                    prompt, 
                    sampling_kwargs,
                    steer_on=False,
                    alpha=None,
                    repeat_num=2,
                    steer_type=None,
                    show_res=False)
                no_steer_item=copy.deepcopy(item)
                no_steer_item["generations"]=[]
                for gen_text in no_steer_gen_texts:
                    no_steer_item["generations"].append({"text":gen_text})
                # no_steer_res.append(no_steer_item)
                nt_file.write(no_steer_item)
                # 转向的结果
            
            steer_on = True
            steered_texts=run_generate(prompt, 
                                       sampling_kwargs,
                                       steer_on=steer_on,
                                       alpha=alpha,
                                       steer_type="gaussian",
                                       repeat_num=2,
                                       show_res=False
                                       )
            
            steer_item=copy.deepcopy(item)
            steer_item["generations"]=[]
            for steer_gen in steered_texts:
                steer_item["generations"].append({"text":steer_gen})
            steer_res.append(steer_item)
            # res.append(copy.deepcopy(item))
            t_file.write(steer_item)
    

# %%
 import jsonlines

# 要添加的数据（字典列表）
data = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
    {"name": "Charlie", "age": 35}
]

# 打开文件（模式为追加模式 'a'）
with jsonlines.open('data.jsonl', mode='a') as writer:
    for item in data:
        writer.write(item)  # 逐行写入

# %%
import json
with open("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/evaluations/res.json",mode="w",encoding="utf-8") as res_f:
    res_f.write(json.dumps(res,ensure_ascii=False))
    

# %%



