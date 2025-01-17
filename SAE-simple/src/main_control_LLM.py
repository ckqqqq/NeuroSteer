import argparse
import os
import re
import logging
from typing import Tuple
from tqdm import tqdm  # For progress bars
from log import setup_logging
import logging
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
import var
# from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list
# from sae_lens.analysis.feature_statistics import (
#     get_all_stats_dfs,
#     get_W_U_W_dec_stats_df,
# )
# from sae_lens.analysis.tsea import (
#     get_enrichment_df,
#     manhattan_plot_enrichment_scores,
#     plot_top_k_feature_projections_by_token_and_category,
#     get_baby_name_sets,
#     get_letter_gene_sets,
#     generate_pos_sets,
#     get_test_gene_sets,
#     get_gene_set_from_regex,
# )
# import plotly_express as px

# Define sampling parameters
sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)
# 解码的参数
MAX_NEW_TOKENS=50
# 最多生成的tokens


# %%
# Define hyperparameters
TASK="sentiment"
if TASK=="sentiment":
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
        "alpha": 500, # 这个alpha后面慢慢调节
        # "steer": "neg-pos",  # Options: "pos", "neg", "neu","pos-neg","cot-direct"
        "method": "val_mul",  # Options: "mean", "val_mul" 用val_mul会比较好
        "topk_mean": 100, # 选取前topk 个均值激活，这个效果一般，会导致很多如what？why？这种被激活0
        "topk_cnt": 100, # 选取前topk个频率激活，目前默认这个，效果很好
        "batch_size": 32 ,# 这个好像没用上
        "source":"pos",
        "target":"neg",
        "mean_type":"dif_mean",
        "steer_type":"last",
        "save_compared":False,
        "debug":False
    }
elif TASK=="cot":
# %% [markdown]
# # 进行COT相关实验
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
        "batch_size": 4
    }
elif TASK=="polite":
    # # 进行礼貌实验
# /home/ckqsudo/code2024/0dataset/ACL_useful_dataset/style_transfer/politeness-corpus
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
else:
    raise ValueError("No Supported Task")

# %%
# 进行礼貌实验

# %%
# Configuration and Hyperparameters
# 将字典转换为 argparse.Namespace 对象
args = argparse.Namespace(**args_dict)
# 测试访问属性


output_dir="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/evaluations/gen_files"
output_dir=os.path.join(output_dir,f"alpha_{args.alpha}_from_{args.source}_to_{args.target}_datasize_{args.data_size}_layer_{args.layer}_mean_{args.mean_type}_steertype_{args.steer_type}_device_{args.device}_batchsize{args.batch_size}")
os.makedirs(output_dir,exist_ok=True)

# Setup logging
setup_logging(output_dir)

# Save hyperparameters
hyperparams = vars(args)

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
# args.steer
logging.info("dataset path "+args.dataset_path)
if TASK=="sentiment":
    neg_train_set, pos_train_set, neu_train_set,val_set,test_set = load_and_prepare_triple_dataset(
        args.dataset_path, "sst5",args.seed, args.data_size
    )
elif "cot"==TASK:
    logging.info("COT "*10)
    all_dataset=load_and_prepare_COT_dataset(
        args.dataset_path, args.seed, args.data_size
    )
elif "polite"==TASK:
    logging.info("polite"*10)
    neg_train_set, pos_train_set, neu_train_set,val_set,test_set=load_and_prepare_triple_dataset(args.dataset_path,"polite", args.seed, args.data_size)
else:
    raise ValueError("No Supported")


# %%
assert neg_train_set[10]!=pos_train_set[10]

# %%
pos_train_set[10],neg_train_set[10]

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

# %%

# %% [markdown]
# 26000(SAE稀疏神经元)对应的非零激活神经元激活统计信息，和激活值统计信息
# %%
steer_info={}

# %%
TASK

# %%
steer_info={}
from functools import partial
if TASK=='polite' or TASK=="sentiment":
    if args.device=="cpu":
        from cpu_utils import get_activation_by_steer_cpu
        get_activation_by_steer_cpu=partial(get_activation_by_steer_cpu,sae=sae,model=model,device=args.device,batch_size=args.batch_size,top_k_mean=args.topk_mean,top_k_cnt=args.topk_cnt)
    logging.info("from"+args.source+"to"+args.target)
    logging.info(f"positive")
    text=pos_train_set["text"][:args.data_size]
    steer_info["pos"]=get_activation_by_steer(text)
    logging.info(f"negative")
    text=neg_train_set["text"][:args.data_size]
    steer_info["neg"]=get_activation_by_steer(text)
    logging.info(f"neutral")
    text=neu_train_set["text"][:args.data_size]
    steer_info["neu"]=get_activation_by_steer(text)
    
elif "cot"=="TASK":
    raise ValueError("BUG")
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

source=args.source
target=args.target
# 调整样本正负性在这里调整 从负样本到正样本还是从正样本()到负样本
# pos 代表积极情绪
# neg 代表消极情绪


# %%
assert bool(torch.all((steer_info["pos"]["latent_value_mean"]-steer_info["neg"]["latent_value_mean"])==0))==False,"数据库读取有问题"

# %%
assert torch.all(steer_info[target]["latent_value_mean"]>=0),"所有SAE的激活需要大于d等于0（maybe）"

# %% [markdown]
# # 泽凯在这里做mask

# %%
# 
logging.info(f"转向方向 dif_{target}-{source}_relu")
steer_info[f"dif_{target}-{source}_relu"]={"latent_frequency":torch.relu(steer_info[target]["latent_frequency"]-steer_info[source]["latent_frequency"]),"latent_value_mean":torch.relu(steer_info[target]["latent_value_mean"]-steer_info[source]["latent_value_mean"]),"target_nz_mean":torch.relu(steer_info[target]["latent_value_mean"])}
"""
nz_cnt: 神经元被激活的次数
nz_mean: 神经元被激活后的平均值
nz_mean_pos: 正样本神经元被激活后的平均值
"""
top_k=args.topk_cnt
_,steer_indices=torch.topk(steer_info[f"dif_{target}-{source}_relu"]["latent_frequency"],top_k)



# %%
# 假设 steer_info[f"dif_{b}-{a}_relu"]["latent_frequency"] 是一个 NumPy 数组
nz_cnt = steer_info[f"dif_{target}-{source}_relu"]["latent_frequency"]

# 先获取非零元素的索引
nz_indices = np.nonzero(nz_cnt)
torch.all(nz_cnt == 0)

# %%
steer_info[f"dif_{target}-{source}_relu"]["latent_frequency"].shape


steer_info[f"dif_{target}-{source}_relu"]["latent_value_mean"][steer_indices]# 这里有0,没有负数比较正常


# %%
# steer_info["dif_-pos"]["latent_value_mean"][steer_indices]

# %%
steer_indices

def compute_delta_matrix(sae: SAE, indices: Tensor, nz_mean_val: Tensor, method: str = "val_mul") -> Tensor:
    logging.info(f"Computing steering vectors using method: {method}")
    if method == "mean":
        delta_matrix = torch.mean(sae.W_dec[indices], dim=0)
    elif method == "val_mul":
        delta_matrix = torch.zeros(sae.W_dec.shape[1], device=sae.W_dec.device)
        for idx in indices:
            delta_matrix += nz_mean_val[idx].item() * sae.W_dec[idx]
    else:
        raise ValueError(f"Unknown method: {method}")
    logging.info(f"Steering vectors computed with shape: {delta_matrix.shape}")
    return delta_matrix
if args.mean_type=="dif_mean":
    delta_matrix=compute_delta_matrix(sae,indices=steer_indices,nz_mean_val=steer_info[f"dif_{target}-{source}_relu"]["latent_value_mean"],method="val_mul")
elif args.mean_type=="tar_mean":
    delta_matrix=compute_delta_matrix(sae,indices=steer_indices,nz_mean_val=steer_info[f"dif_{target}-{source}_relu"]["target_nz_mean"],method="val_mul")
else:
    raise ValueError("Unsupported")


model.tokenizer.eos_token

# %%
model.tokenizer

# %%
delta_matrix #理论上这里有正有负比较正常



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
        #     resid_pre[:, :-2, :] += args.alpha * delta_matrix
        # elif steer_type=="gaussian":
        #     # 高斯卷积的方式放缩干预矩阵，
        #     # 这里需要一个高斯核，然后对delta_matrix进行卷积
        #     gaussian_kernel = torch.tensor([1,2,1])
        #     delta_matrix = torch.conv1d(delta_matrix, gaussian_kernel, padding=1)
        #     resid_pre[:, :-1, :] += args.alpha * delta_matrix
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
            max_new_tokens=MAX_NEW_TOKENS,
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



# Example prompt from the selected set
example_prompt = "What really matters is that they know"
# example_prompt=model.tokenizer
# print( all_dataset["val"]["prompt"][:3],all_dataset["val"]["A"][:3])
logging.info(f"Example prompt: {example_prompt}")

# Generate without steering

logging.info("Generating texts **without** steering... ")
generated_texts_no_steer = run_generate(example_prompt, sampling_kwargs,steer_on=False,alpha=0,show_res=True)
logging.info("干预之后的结果")
# bef,aft=args.steer.split("-")
logging.info(f"干预方向{source}->{target},礼貌任务下，neg=impolite，情感任务下 pos=积极情感")
# Generate with steering
# steer_on = True
# alpha = 300
# alpha=args.aplha
logging.info("** Generating texts with steering... Target **")
logging.info(f"form {source} to target")
generated_texts_with_steer = run_generate(
    example_prompt, 
    sampling_kwargs,
    steer_on=True,
    alpha=args.alpha,
    steer_type=args.steer_type,
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
target

# %%
sampling_kwargs['verbose']=False
sampling_kwargs

args.data_size

STEER_TYPE=args.steer_type
ALPHA=args.alpha

# %%


# %%
import sys
if args.debug:
    sys.exit(0)

import copy
# Example prompt from the selected set
import jsonlines
res=[]
params={}
params["params"]={**vars(args),**sampling_kwargs,"max_new_tokens":50,"steer":f"from {source} to {target}"}
params["alpha"]=ALPHA
logging.info(f"Running with alpha: {ALPHA}")
logging.info(f"Running with prompt_type: "+str(params["params"]["steer"]))
# res.append(params)

no_steer_res=[]
steer_res=[]

assert source in prompts,"有这个数据集"
# 打开文件（模式为追加模式 'a'）

with jsonlines.open(os.path.join(output_dir,"params.jsonl"), mode='w') as writer:
    writer.write(params)  # 逐行写入
save_compared=args.save_compared
with jsonlines.open(os.path.join(output_dir,"no_steer_gen_res.jsonl"), mode='w') as nt_file:
    with jsonlines.open(os.path.join(output_dir,"steer_gen_res.jsonl"), mode='w') as t_file: 
        for idx,item in tqdm(enumerate(list(prompts[source])[:1000])):
            prompt=item["prompt"]["text"]
            item["label"]=source

            # 没转向的结果
            if save_compared:
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
            
            # steer_on = True
            steered_texts=run_generate(prompt, 
                                       sampling_kwargs,
                                       steer_on=True,
                                       alpha=ALPHA,
                                       steer_type=STEER_TYPE,
                                       repeat_num=2,
                                       show_res=False
                                       )
            steer_item=copy.deepcopy(item)
            steer_item["generations"]=[]
            for steer_gen in steered_texts:
                steer_item["generations"].append({"text":steer_gen})
            if idx%50==0:
                logging.info(f"from: {source} to: {target} dataset: {source}")
                logging.info(f"alpha: {ALPHA}")
                logging.info(steer_item)
            t_file.write(steer_item)


