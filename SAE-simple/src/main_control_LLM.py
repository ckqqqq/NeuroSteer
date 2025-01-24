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
import numpy as np
import argparse
import logging

from data_preprocess import load_and_prepare_triple_dataset,load_and_prepare_COT_dataset,load_and_prepare_debate_triple_dataset
from utils import load_environment
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


parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='sentiment',choices=['sentiment','cot','polite','toxicity'], help='Task type: sentiment, cot, polite')
parser.add_argument('--layer', type=int, default=6, help='Layer number to analyze')
parser.add_argument('--LLM', type=str, default='gpt2-small', help='LLM model')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
from typing import Union
parser.add_argument('--data_size', type=int, default=-1, help='Data size')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps', 'auto'], help='Device to use')
parser.add_argument('--alpha', type=int, default=500, help='Alpha parameter')
# parser.add_argument('--steer', type=str, default='neg-pos', help='Steering direction')
parser.add_argument('--method', type=str, default='val_mul', choices=['mean', 'val_mul'], help='Method to use')
parser.add_argument('--topk_mean', type=int, default=100, help='Top K mean selection')#目前没有用了
parser.add_argument('--topk_cnt', type=int, default=100, help='Top K count selection')# 主要用这个
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--source', type=str, default='pos', help='Source class')#赞同、积极情感、礼貌、COT、无毒性
parser.add_argument('--target', type=str, default='neg', help='Target class')#不赞同、消极情感、不礼貌、直接推理、无毒性
parser.add_argument('--mean_type', type=str, default="dif_mean",choices=['dif_mean','tar_mean'], help='Mean type')
parser.add_argument('--steer_type', type=str, default="last",choices=['all','last','last2',"gaussian"], help='Steer type')
parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
parser.add_argument('--dataset_path', type=str, default="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5", help='Dataset path')
parser.add_argument('--prompt_path', type=str, default="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k", help='Prompt path')
parser.add_argument('--env_path', type=str, default="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env", help='Environment path')
# 解码参数
parser.add_argument("--temperature", type=float, default=0.9, help='Sampling temperature for generation')
parser.add_argument("--top_p", type=float, default=0.3, help='Top-p (nucleus) sampling parameter')
parser.add_argument("--freq_penalty", type=float, default=1.0, help='Frequency penalty for generation')
# 垃圾args，用布尔会有奇怪的东西
parser.add_argument('--debug', type=int, default=0, choices=[0, 1], help='Debug flag: 0 for False, 1 for True')
parser.add_argument("--save_no_steer", type=int,default=0, choices=[0, 1], help='是否需要比较GPT原先生成的结果')
parser.add_argument("--cache_delta_matrix", type=int,default=0, choices=[0, 1], help='是否需要保存对应任务下干预的delta矩阵和对应信息?')
parser.add_argument("--is_norm_delta_matrix", type=int,default=0, choices=[0, 1], help='是否需要对delta矩阵进行归一化（如果归一化算出来的alpha理论上更体现（单位）数值，但是实际上没多大影响，这玩意和alpha挂钩哦，实验的时候确保统一）')

args = parser.parse_args()


# sampling_kwargs = dict(temperature=args.temperature,top_p=0.3, freq_penalty=1.0)
# 将 sampling_kwargs 直接构建为字典
sampling_kwargs = {
    "temperature": args.temperature,
    "top_p": args.top_p,
    "freq_penalty": args.freq_penalty,
}
sampling_kwargs['verbose']=False
TASK =args.task
STEER_TYPE=args.steer_type
ALPHA=args.alpha
MAX_NEW_TOKENS=50
DATA_SIZE=args.data_size
if DATA_SIZE==-1:
    logging.info("select all data for activation engineering")
    DATA_SIZE="ALL"

OUTPUT_DIR=os.path.join(args.output_dir,f"{TASK}_alpha_{args.alpha}_from_{args.source}_to_{args.target}_datasize_{DATA_SIZE}_layer_{args.layer}_mean_{args.mean_type}_steertype_{args.steer_type}_device_{args.device}_batchsize{args.batch_size}")
os.makedirs(OUTPUT_DIR,exist_ok=True)

# Setup logging
setup_logging(OUTPUT_DIR)
# Save hyperparameters
from utils import params_to_dict
hyperparams = params_to_dict(args,is_print=True)

# %%
# Load Environment Variables
load_environment(args.env_path)
# %%
# args.steer
logging.info("dataset path "+args.dataset_path)
if TASK=="sentiment":
    neg_train_set, pos_train_set, neu_train_set,val_set,test_set = load_and_prepare_triple_dataset(
        args.dataset_path, "sst5",args.seed, DATA_SIZE
    )
elif "cot"==TASK:
    raise ValueError("需要特殊重写，有问题")
    logging.info("COT "*10) # 这个只有llama 支持
    all_dataset=load_and_prepare_COT_dataset(
        args.dataset_path, args.seed, DATA_SIZE
    )
elif "toxicity"==TASK:
    logging.info("toxicity "*10)
    from data_preprocess import load_and_prepare_toxicity_dataset
    neg_train_set, pos_train_set,_,_,_=load_and_prepare_toxicity_dataset(
        args.dataset_path,task=TASK, seed=None, num_samples=DATA_SIZE
    )
elif "polite"==TASK:
    logging.info("polite"*10)
    neg_train_set, pos_train_set, neu_train_set,val_set,test_set=load_and_prepare_triple_dataset(args.dataset_path,"polite", args.seed, DATA_SIZE)
elif "debate"==TASK:
    logging.info("debate"*10)
    neg_train_set, pos_train_set,val_set,test_set=load_and_prepare_debate_triple_dataset(args.dataset_path, args.seed, args.data_size)
    
else:
    raise ValueError("No Supported")

# if DATA_SIZE=="ALL":
#     DATA_SIZE=len(neg_train_set) 这是什么鬼啊
if DATA_SIZE=="ALL":
    DATA_SIZE=min(len(neg_train_set),len(pos_train_set))

# %%pos_train_set[10],neg_train_set[10]

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
    """分析latents支持批次处理

    Args:
        batch_latents (Tensor): _description_
        top_k_mean (int, optional): _description_. Defaults to 100.
        top_k_cnt (int, optional): _description_. Defaults to 100.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: _description_
    """
    SAE_LATENT_SIZE=sae.W_dec.shape[0]
    
    # 计算非0激活在对应位置的激活频率   
    # logging.info("Computing non-zero element counts") 
    lat_freq = (batch_latents != 0).sum(dim=(0, 1))
    # 计算非0激活在对应位置的激活值的和（注意这里不能计算均值）
    # logging.info("Computing sum of non-zero elements")
    lat_val_sum = batch_latents.sum(dim=(0, 1))
    # logging.info("Computing mean of non-zero elements")
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
    lat_freq,lat_val_sum=torch.zeros(SAE_LATENT_SIZE).to("cpu"),torch.zeros(SAE_LATENT_SIZE).to("cpu")# 避免OOM
    # 使用 tqdm 显示进度条
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            # logging.info(f"Batch {i // batch_size + 1}: batch_size {batch_size}")
            try:
                sv_logits, cache = model.run_with_cache(batch_texts, prepend_bos=False, device="cuda")
            except Exception as e:
                logging.error(f"Error processing batch {i // batch_size + 1}: {e}")
                raise ValueError(str([len(i) for i in batch_texts]))
            batch_hidden_states = cache[hook_point]
            # logging.info(f"Batch {i // batch_size + 1}: Hidden states shape: {batch_hidden_states.shape}")

            # logging.info(f"Encoding hidden states for batch {i // batch_size + 1}")
            # 假设 sae.encode 支持批量编码
            batch_latents = sae.encode(batch_hidden_states)  # 形状: (batch_size, latent_dim)
            batch_info=analyze_latents(batch_latents)
            lat_freq=lat_freq+batch_info["latent_frequency"].to("cpu")
            lat_val_sum=lat_val_sum+batch_info["latent_value_sum"].to("cpu")
    lat_val_mean=torch.where(lat_freq != 0, lat_val_sum / lat_freq, torch.tensor(0.0, device="cpu"))
    logging.info(f"Total non-zero element shape: {lat_freq.shape}")
    assert lat_freq.shape[0]==lat_freq.shape[0]==sae.W_dec.shape[0], "sae latent dimension mismatch"
    return {"latent_frequency":lat_freq.to(device),"latent_value_mean":lat_val_mean.to(device)}
# %%
# args.steer

def get_activation_by_steer(texts:list):
    hook_point = sae.cfg.hook_name
    # Compute latents with batch processing
    lat_info=compute_latents(sae, model, texts, hook_point, args.device, args.batch_size)
    return {"latent_value_mean":lat_info["latent_value_mean"],"latent_frequency":lat_info["latent_frequency"]}

# %% [markdown]
# 26000(SAE稀疏神经元)对应的非零激活神经元激活统计信息，和激活值统计信息
# %%

steer_info={}
from functools import partial
if TASK=='polite' or TASK=="sentiment":
    # if args.device=="cpu":
    #     logging.info("CPU处理")
    #     from cpu_utils import get_activation_by_steer_cpu
    #     get_activation_by_steer=partial(get_activation_by_steer_cpu,sae=sae,model=model,device=args.device,batch_size=args.batch_size,top_k_mean=args.topk_mean,top_k_cnt=args.topk_cnt)
    # 现在基本都不炸显存了
    logging.info("from"+args.source+"to"+args.target)
    logging.info(f"positive")
    text=pos_train_set["text"][:DATA_SIZE]
    steer_info["pos"]=get_activation_by_steer(text)
    logging.info(f"negative")
    text=neg_train_set["text"][:DATA_SIZE]
    steer_info["neg"]=get_activation_by_steer(text)
    logging.info(f"neutral")
    text=neu_train_set["text"][:DATA_SIZE]
    steer_info["neu"]=get_activation_by_steer(text)
elif TASK=='debate':
    # if args.device=="cpu":
    #     logging.info("CPU处理,if OOD try this")
    #     from cpu_utils import get_activation_by_steer_cpu
    #     get_activation_by_steer=partial(get_activation_by_steer_cpu,sae=sae,model=model,device=args.device,batch_size=args.batch_size,top_k_mean=args.topk_mean,top_k_cnt=args.topk_cnt)
    logging.info("from"+args.source+"to"+args.target)
    logging.info(f"support")
    text=pos_train_set["text"][:DATA_SIZE]
    steer_info["pos"]=get_activation_by_steer(text)
    logging.info(f"oppose")
    text=neg_train_set["text"][:DATA_SIZE]
    steer_info["opp"]=get_activation_by_steer(text)
elif TASK=='toxicity':
    logging.info("from"+args.source+"to"+args.target)
    logging.info(f"toxic")
    steer_info["pos"]=get_activation_by_steer(
        pos_train_set["text"][:DATA_SIZE]
        )
    steer_info["neg"]=get_activation_by_steer(
        neg_train_set["text"][:DATA_SIZE]
        )
elif "cot"==TASK:
    raise ValueError("BUG")
    if args.steer in "cot-direct":
        texts=all_dataset["train"]["Q+A"][:DATA_SIZE]
        print(type(texts))
        steer_info["direct"]=get_activation_by_steer(texts)
        
        texts=all_dataset["train"]["Q+COT_A"][:DATA_SIZE]
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
assert bool(torch.all((steer_info[target]["latent_value_mean"]-steer_info[source]["latent_value_mean"])==0))==False,"数据库读取有问题"
assert torch.all(steer_info[target]["latent_value_mean"]>=0),"所有SAE的激活需要大于d等于0（maybe）"

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
lat_freq = steer_info[f"dif_{target}-{source}_relu"]["latent_frequency"]
# 先获取非零元素的索引
lat_acti_indices = np.nonzero(lat_freq)
assert torch.all(lat_freq == 0)==False,"latent_frequency全为0元素读取有问题"

# %%
steer_info[f"dif_{target}-{source}_relu"]["latent_frequency"].shape
steer_info[f"dif_{target}-{source}_relu"]["latent_value_mean"][steer_indices]# 这里有0,没有负数比较正常

def compute_delta_matrix(sae: SAE, indices: Tensor, nz_mean_val: Tensor, method: str = "val_mul",is_norm:int=0) -> Tensor:
    assert is_norm in [0,1] and method in ["val_mul"], "Invalid arguments"
    # logging.info(f"Computing steering vectors using method: {method}")
    delta_matrix = torch.zeros(sae.W_dec.shape[1], device=sae.W_dec.device)
    for idx in indices:
        delta_matrix += nz_mean_val[idx].item() * sae.W_dec[idx]
    # logging.info(f"Steering vectors computed with shape: {delta_matrix.shape}")
    if is_norm==1:
        norm = torch.norm(delta_matrix, p='fro')  # 计算 Frobenius 范数
        delta_matrix = delta_matrix / norm
        logging.info(f"Steering vectors normalized with L2 norm: {norm} 归一化矩阵大小")
    return delta_matrix
if args.mean_type=="dif_mean":
    delta_matrix=compute_delta_matrix(sae,indices=steer_indices,nz_mean_val=steer_info[f"dif_{target}-{source}_relu"]["latent_value_mean"],method="val_mul",is_norm=args.is_norm_delta_matrix)
elif args.mean_type=="tar_mean":
    delta_matrix=compute_delta_matrix(sae,indices=steer_indices,nz_mean_val=steer_info[f"dif_{target}-{source}_relu"]["target_nz_mean"],method="val_mul",is_norm=args.is_norm_delta_matrix)
else:
    raise ValueError("Unsupported")


# %%
logging.info("delta_matrix: "+str(delta_matrix)) #理论上这里有正有负比较正常
# %% [markdown]
# # 这里得到的就是delta_matricx
sae.cfg.hook_name
f"blocks.{args.layer}.hook_resid_post"
import einops
steer_cnt=0


def steering_hook(resid_pre, hook,steer_on, alpha, steer_type="last"):
    if resid_pre.shape[1] == 1:
        return
    # 判断是否进行干预
    if steer_on:
        if steer_type == "last":
            # 对最后一个token前的部分应用干预，使用给定的 delta_matrix
            resid_pre[:, :-1, :] += alpha * delta_matrix# best
            # 如果提前干预效果会更好更连贯
        elif steer_type == "gaussian":
            # 使用高斯卷积对输入进行干预
            from utils import half_gaussian_kernel
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
        elif steer_type == "all":
            resid_pre[:, :, :] += alpha * delta_matrix# 全部干预
        elif steer_type == "last2":
            resid_pre[:, :-2, :] += args.alpha * delta_matrix # 提前两个token进行干预
        else:
            raise ValueError("Unknown steering type")

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
    res_str_batch = model.to_string(res)
    if show_res:
        logging.info(f"prompt:*****\n{example_prompt}")
        for idx, text in enumerate(res_str_batch):
            logging.info(f"Generated Text: {idx+1}:\n{text[len(example_prompt):]}")
        
    return res_str_batch


# Example prompt from the selected set
example_prompt = "What really matters is that they know"
logging.info(f"Example prompt: {example_prompt}")

# Generate without steering

logging.info("Generating texts **without** steering... ")
generated_texts_no_steer = run_generate(example_prompt, sampling_kwargs,steer_on=False,alpha=0,show_res=True)
logging.info("干预之后的结果")
# bef,aft=args.steer.split("-")
logging.info(f"干预方向{source}->{target},礼貌任务下，neg=impolite，情感任务下 pos=积极情感")
logging.info("** Generating texts with steering... Target **")
logging.info(f"form {source} to {target}")
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
# # 总结小批量实验进展
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
import copy
# Example prompt from the selected set
import jsonlines

def eval_on_full_data():
    """跑全量实验
    Raises:
        NotImplementedError: _description_
    """
    logging.info("Running on full data")
    
    if TASK=="sentiment":
        logging.info("Out of Domain: Calculate at A dataset, Evaluate at B dataset")
        from data_preprocess import load_and_prepare_sentiment_prompts
        prompts=load_and_prepare_sentiment_prompts(prompt_path=args.prompt_path,task=TASK)
    elif TASK=="politeness":
        logging.info("In Domain: Calculate at A dataset, Evaluate at B dataset")
        # prompts=load_and_prepare_politeness_prompts(pormpt_path=args.prompt_path,sample=args.seed)
        pass
    elif TASK=="toxicity":
        logging.info("Out of Domain: Calculate at A dataset, Evaluate at B dataset")
        from data_preprocess import load_and_prepare_toxicity_prompts 
        prompts=load_and_prepare_toxicity_prompts(prompt_path=args.prompt_path,task=TASK)
    elif TASK=='debate':
        raise NotImplementedError('此处的prompt格式还需要修改')
        logging.info("Out of Domain: Calculate at A dataset, Evaluate at B dataset")
        from data_preprocess import load_and_prepare_debate_prompts 
        prompts=load_and_prepare_debate_prompts(prompt_path=args.prompt_path,task=TASK)
    else:
        raise NotImplementedError("No Supported Task")    
    assert source in prompts,"prompt steer source (pos/neg) not in prompts, please check the data_preprocess section"
    prompts=prompts[source]
    
    param={**vars(args),**sampling_kwargs,"max_new_tokens":50,"steer":f"from {source} to {target}"}
    param["alpha_recheck"]=ALPHA
    logging.info(f"Running with alpha: {ALPHA}")
    logging.info(f"Running with prompt_type: "+str(param["steer"]))

    
    # 打开文件（模式为追加模式 'a'）
    with jsonlines.open(os.path.join(OUTPUT_DIR,"params.jsonl"), mode='w') as writer:
        writer.write(param)  # 逐行写入
    SAVE_COMPARED=args.save_no_steer
    with jsonlines.open(os.path.join(OUTPUT_DIR,"no_steer_gen_res.jsonl"), mode='w') as nt_file:
        with jsonlines.open(os.path.join(OUTPUT_DIR,"steer_gen_res.jsonl"), mode='w') as t_file: 
            for idx,item in tqdm(enumerate(list(prompts))):
                prompt=item["prompt"]["text"]
                item["label"]=source
                # 没转向的结果
                if SAVE_COMPARED:
                    logging.info("Provide No Steer Result")
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
                    logging.info(f"{TASK}: from {source} to {target} prompt_set: {source}")
                    logging.info(steer_item)
                t_file.write(steer_item)


params_to_dict(args,is_print=True)
if args.debug==1:
    logging.info(f"debug mode,show example, no full dataset eval")
elif args.debug==0:
    eval_on_full_data()
else:
    raise ValueError("debug must be 0 or 1")