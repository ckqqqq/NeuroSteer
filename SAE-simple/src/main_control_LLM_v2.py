import argparse
import os
import re
import logging
from typing import Tuple
from sympy import false
from tqdm import tqdm  # For progress bars
from log import setup_logging
import logging
import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
import numpy as np
import argparse
import logging

from data_preprocess import load_and_prepare_triple_dataset,load_and_prepare_COT_dataset,load_and_prepare_debate_triple_dataset, load_and_prepare_polite_dataset
from utils import load_environment
import time
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

parser.add_argument('--task', type=str, default='sentiment',choices=['sentiment','cot','polite','toxicity','debate'], help='Task type: sentiment, cot, polite')
parser.add_argument('--layer', type=int, default=6, help='Layer number to analyze')
parser.add_argument('--LLM', type=str, default='gpt2-small', help='LLM model')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
from typing import Union
parser.add_argument('--data_size', type=int, default=-1, help='Data size')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps', 'auto'], help='Device to use')
parser.add_argument('--alpha', type=float, default=500, help='Alpha parameter')
# parser.add_argument('--steer', type=str, default='neg-pos', help='Steering direction')
parser.add_argument('--method', type=str, default='val_mul', choices=['mean', 'val_mul'], help='Method to use')
parser.add_argument('--topk_mean', type=int, default=100, help='Top K mean selection')#目前没有用了
parser.add_argument('--topk_cnt', type=int, default=100, help='Top K count selection')# 主要用这个
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--source', type=str, default='pos', help='Source class')#赞同、积极情感、礼貌、COT、无毒性
parser.add_argument('--target', type=str, default='neg', help='Target class')#不赞同、消极情感、不礼貌、直接推理、无毒性
parser.add_argument('--prompt_source', type=str, default="",choices=['pos','neu','neg'], help='数据集prompt的极性')
parser.add_argument('--prompt_data_size', type=int, default=-1, help='用于评估的prompt的条数，默认-1为全部')
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
parser.add_argument("--example_prompt",type=str,default="I feel",help='Example prompt eg "I FEEL"')
# 垃圾args，用布尔会有奇怪的东西
parser.add_argument('--debug', type=int, default=0, choices=[0, 1], help='Debug flag: 0 for False, 1 for True')
parser.add_argument("--save_no_steer", type=int,default=0, choices=[0, 1], help='是否需要比较GPT原先生成的结果')
parser.add_argument("--is_norm_delta_matrix", type=int,default=0, choices=[0, 1], help='是否需要对delta矩阵进行归一化（如果归一化算出来的alpha理论上更体现（单位）数值，但是实际实验结果表明没多大影响，这玩意和alpha挂钩哦，实验的时候确保统一）')
parser.add_argument("--use_cache", type=int,default=0, choices=[0, 1], help='是否需要保存steer_info?')

parser.add_argument("--repeat_num", type=int,default=2, choices=[0, 1,2,3,4,5], help='重复生成每个prompt的次数')
parser.add_argument("--gen_batch_size",type=int,default=1, help='批量生成的时候生成batch大小，基于显卡调整')
# 注意：不去使用则覆写！
args = parser.parse_args()

####################################################################################setup
# sampling_kwargs = dict(temperature=args.temperature,top_p=0.3, freq_penalty=1.0)
# 将 sampling_kwargs 直接构建为字典
sampling_kwargs = {
    "temperature": args.temperature,
    "top_p": args.top_p,
    "freq_penalty": args.freq_penalty,
}


sampling_kwargs['verbose']=False

EXAMPLE_PROMPT_LIST=str(args.example_prompt).split("|")
assert isinstance(EXAMPLE_PROMPT_LIST,list) and isinstance(EXAMPLE_PROMPT_LIST[0],str),"EXAMPLE_PROMPT_LIST必须是list[str]"
assert args.example_prompt!="","输入测试prompt"
logging.info(f"Example prompt: {EXAMPLE_PROMPT_LIST}")


TASK =args.task
STEER_TYPE=args.steer_type
SOURCE=args.source
TARGET=args.target
# 调整样本正负性在这里调整 从负样本到正样本还是从正样本()到负样本
# pos 代表积极情绪
# neg 代表消极情绪
ALPHA=args.alpha
MAX_NEW_TOKENS=50
GEN_BATCH_SIZE = args.gen_batch_size

SAVE_NO_STEER=args.save_no_steer
DATA_SIZE=args.data_size
if DATA_SIZE==-1:
    logging.info("select all data for activation engineering")
    DATA_SIZE="ALL"

CACHE_DIR=os.path.join(args.output_dir,f"{args.LLM}_{TASK}_layer_{args.layer}_datasize_{DATA_SIZE}_batchsize{args.batch_size}_topK_{args.topk_cnt}")
OUTPUT_DIR=os.path.join(CACHE_DIR,f"alpha_{args.alpha}_from_{args.source}_to_{args.target}_prompt_{args.prompt_source}_mean_{args.mean_type}_steertype_{args.steer_type}_device_{args.device}")
## 注意，一定要小心CACHE_DIR的路径！！
os.makedirs(OUTPUT_DIR,exist_ok=True)

# Setup logging
setup_logging(OUTPUT_DIR)
# Save hyperparameters
from utils import params_to_dict
hyperparams = params_to_dict(args,is_print=True)
# Load Environment Variables
load_environment(args.env_path)
# %%
####################################################################### task
logging.info("dataset path "+args.dataset_path)
if TASK=="sentiment":
    neg_train_set, pos_train_set, neu_train_set,val_set,test_set = load_and_prepare_triple_dataset(
        dataset_path=args.dataset_path, 
        seed=args.seed, 
        dataset_name="sst5",
    )
elif "cot"==TASK:
    logging.info("COT "*10) # 这个只有llama 支持
    neg_train_set, pos_train_set,_,_,_=load_and_prepare_COT_dataset(
        dataset_path=args.dataset_path, 
        seed=args.seed, 
    )
elif "toxicity"==TASK:
    logging.info("toxicity "*10)
    from data_preprocess import load_and_prepare_toxicity_dataset
    neg_train_set, pos_train_set,_,_,_=load_and_prepare_toxicity_dataset(
        args.dataset_path,
        task=TASK, 
        seed=None
    )
    print(neg_train_set)
elif "polite"==TASK:
    logging.info("polite"*10)
    pos_train_set, neg_train_set, neu_train_set, polite_test_set = load_and_prepare_polite_dataset(args.dataset_path, args.seed)
elif "debate"==TASK:
    logging.info("debate"*10)
    pos_train_set, neg_train_set,val_set,test_set = load_and_prepare_debate_triple_dataset(args.dataset_path, args.seed)
else:
    raise ValueError("No Supported")




logging.info(f"Loading Model Loading SAE for layer {args.layer} {args.LLM}")
if "llama" in args.LLM:
    logging.info(f"Loading model: {args.LLM}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="meta-llama/Meta-Llama-3.1-8B",
        sae_id=f"blocks.{args.layer}.hook_resid_pre",
        device=args.device
    )
    model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", device=args.device)
    logging.info(f"model architecture for {args.LLM} {model}")

elif "gemma-2-2b" in args.LLM:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = "gemma-scope-2b-pt-res-canonical",
        sae_id = "layer_0/width_16k/canonical",
        device=args.device)
    logging.info(f"Loading model: {args.LLM}")
    model = HookedTransformer.from_pretrained(args.LLM, device=args.device)
elif "gemma-2b" in args.LLM:
    logging.info(f"Loading model: {args.LLM}")
    sae, cfg_dict, _ = SAE.from_pretrained(
        release=f"{args.LLM}-res-jb", sae_id=f"blocks.{args.layer}.hook_resid_pre", device=args.device
    )
    model = HookedTransformer.from_pretrained(args.LLM, device=args.device)
    logging.info(f"model architecture for {args.LLM} {model}")
elif "gpt2-small" in args.LLM:
    logging.info(f"Loading model: {args.LLM} SAE gpt2-small-res-jb")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=f"{args.LLM}-res-jb",
        sae_id=f"blocks.{args.layer}.hook_resid_pre",
        device=args.device
    )
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # 设置填充标记为 EOS token
    tokenizer.pad_token = tokenizer.eos_token
    model = HookedTransformer.from_pretrained(args.LLM, device=args.device,tokenizer=tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained(args.LLM)
else:
    raise ValueError("No Supported")
logging.info(f"model architecture for {args.LLM} {model} {tokenizer}")
##########################################################################Collection
start_train_time=time.time()
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
    
    # logging.info("Computing non-zero element counts") 
    lat_freq = (batch_latents != 0).sum(dim=(0, 1))# 计算非0激活在对应latent位置的激活频率
    # 第一维度 batch_size 第二维度 seq_len 第三维度 sae_latent_size 
    # logging.info("Computing sum of non-zero elements")
    lat_val_sum = batch_latents.sum(dim=(0, 1))# 计算非0激活在对应latent位置的激活值的和（注意这里不能计算均值）
    # 第一维度 batch_size 第二维度 seq_len 第三维度 sae_latent_size
    # logging.info("Computing mean of non-zero elements")
    assert batch_latents.shape[-1]==SAE_LATENT_SIZE==lat_val_sum.shape[0]==lat_freq.shape[0], "Latent dimension mismatch"
    return {"latent_frequency":lat_freq,"latent_value_sum":lat_val_sum}
def SAE_encoding(sae: SAE, model: HookedTransformer, texts: list, hook_point: str, device: str, batch_size: int) -> list:
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
    lat_freq,lat_val_sum=torch.zeros(SAE_LATENT_SIZE).to("cpu"),torch.zeros(SAE_LATENT_SIZE).to("cpu")# 避免OOM
    # 使用 tqdm 显示进度条
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            # logging.info(f"Batch {i // batch_size + 1}: batch_size {batch_size}")
            try:
                sv_logits, cache = model.run_with_cache(batch_texts, prepend_bos=True, device=args.device)
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
    
    assert lat_freq.shape[0]==lat_val_mean.shape[0]==sae.W_dec.shape[0], "sae latent dimension mismatch"
    return {"latent_frequency":lat_freq.to(device),"latent_value_mean":lat_val_mean.to(device)}
# %%
# args.steer

def get_activation_by_steer(texts:list):
    hook_point = sae.cfg.hook_name
    # Compute latents with batch processing
    lat_info=SAE_encoding(sae, model, texts, hook_point, args.device, args.batch_size)
    return {"latent_value_mean":lat_info["latent_value_mean"],"latent_frequency":lat_info["latent_frequency"]}

# %% [markdown]
# 26000(SAE稀疏神经元)对应的非零激活神经元激活统计信息，和激活值统计信息


# 示例使用
def compute_steer_info():
    global DATA_SIZE
    # 这里是原先的代码逻辑，计算 steer_info
    steer_info = {}
    steer_polar_list=[str(args.source),(args.target)]# eg ['pos','neu']
    steer_lens={"pos":len(pos_train_set["text"]),"neg":len(neg_train_set["text"])}
    if TASK == 'polite' or TASK == "sentiment":
        logging.info(f":>> {TASK} : from " + args.source + " to " + args.target)
        # 加入neu
        steer_lens["neu"]=len(neu_train_set["text"])
        if DATA_SIZE=="ALL":
            DATA_SIZE=min([steer_lens[steer] for steer in steer_polar_list])
        
            
        if 'pos' in steer_polar_list:
            logging.info(f"positive")
            steer_info["pos"] = get_activation_by_steer(pos_train_set["text"][:DATA_SIZE])
        if 'neg' in steer_polar_list:
            logging.info(f"negative")
            steer_info["neg"] = get_activation_by_steer(neg_train_set["text"][:DATA_SIZE])
        if 'neu' in steer_polar_list:
            logging.info(f"neutral")
            steer_info["neu"] = get_activation_by_steer(neu_train_set["text"][:DATA_SIZE])
    else:
        # 不包含neu的处理
        if DATA_SIZE=="ALL":
            DATA_SIZE=min([steer_lens[steer] for steer in steer_polar_list])
            
        if  TASK == 'debate':
            logging.info(":> Debate :from" + args.source + "to" + args.target)
            logging.info(f"support")
            steer_info["pos"]=get_activation_by_steer(pos_train_set["text"][:DATA_SIZE])
            logging.info(f"oppose")
            steer_info["neg"]=get_activation_by_steer(neg_train_set["text"][:DATA_SIZE])
        elif TASK == 'toxicity':
            logging.info(f":) Toxicity :from" + args.source + "to" + args.target)

            steer_info["pos"] = get_activation_by_steer(pos_train_set["text"][:DATA_SIZE])
            steer_info["neg"] = get_activation_by_steer(neg_train_set["text"][:DATA_SIZE])
        elif "cot" == TASK:
            logging.info(":) COT: from" + args.source + "to" + args.target)
            steer_info["pos"] = get_activation_by_steer(pos_train_set["text"][:DATA_SIZE])
            steer_info["neg"] = get_activation_by_steer(neg_train_set["text"][:DATA_SIZE])
        else:
            raise ValueError("Non support Task")
        
    assert DATA_SIZE>100,"训练数据量太少了，看看数据集"
    setattr(args, 'real_data_size_for_train', DATA_SIZE)
    return steer_info

# 缓存到文件里面
from utils import load_or_cache_steer_info
steer_info = load_or_cache_steer_info(CACHE_DIR=CACHE_DIR,args=args,cache_filename=f"steer_info_cache_of_{args.LLM}_l{args.layer}.pkl", compute_func=compute_steer_info)

end_train_time=time.time()

# %%
assert TARGET in steer_info.keys() and SOURCE in steer_info.keys(),str(steer_info.keys())+"请检查source和target是否正确"
assert bool(torch.all((steer_info[TARGET]["latent_value_mean"]-steer_info[SOURCE]["latent_value_mean"])==0))==False,"数据库读取有问题,请检查很可能neg pos写混了"
assert torch.all(steer_info[TARGET]["latent_value_mean"]>=0),"所有SAE的激活需要大于d等于0（maybe）"

logging.info(f"转向方向 dif_{TARGET}-{SOURCE}_relu")
steer_info[f"dif_{TARGET}-{SOURCE}_relu"]={"latent_frequency":torch.relu(steer_info[TARGET]["latent_frequency"]-steer_info[SOURCE]["latent_frequency"]),"latent_value_mean":torch.relu(steer_info[TARGET]["latent_value_mean"]-steer_info[SOURCE]["latent_value_mean"]),"target_nz_mean":torch.relu(steer_info[TARGET]["latent_value_mean"])}

"""
nz_cnt: 神经元被激活的次数
nz_mean: 神经元被激活后的平均值
nz_mean_pos: 正样本神经元被激活后的平均值
"""
top_k=args.topk_cnt
_,steer_indices=torch.topk(steer_info[f"dif_{TARGET}-{SOURCE}_relu"]["latent_frequency"],top_k)

# %%
# 假设 steer_info[f"dif_{b}-{a}_relu"]["latent_frequency"] 是一个 NumPy 数组
# lat_freq = steer_info[f"dif_{TARGET}-{SOURCE}_relu"]["latent_frequency"]
# 先获取非零元素的索引
# lat_acti_indices = np.nonzero(lat_freq)
assert torch.all(steer_info[f"dif_{TARGET}-{SOURCE}_relu"]["latent_frequency"] == 0)==False,"latent_frequency全为0元素读取有问题"
logging.info("sae cfg.hook_name 挂载名称: "+str(sae.cfg.hook_name)) #挂载名称
# %%
steer_info[f"dif_{TARGET}-{SOURCE}_relu"]["latent_frequency"].shape
steer_info[f"dif_{TARGET}-{SOURCE}_relu"]["latent_value_mean"][steer_indices]# 这里有0,没有负数比较正常


def SAE_decoding(sae: SAE, indices: Tensor, latent_value_mean: Tensor, method: str = "val_mul",is_norm:int=0) -> Tensor:
    assert is_norm in [0,1] and method in ["val_mul"], "Invalid arguments"
    delta_h = torch.zeros(sae.W_dec.shape[1], device=sae.W_dec.device)
    for idx in indices:
        delta_h += latent_value_mean[idx].item() * sae.W_dec[idx]
    # logging.info(f"Steering vectors computed with shape: {delta_matrix.shape}")
    if is_norm==1:
        norm = torch.norm(delta_h, p='fro')  # 计算 Frobenius 范数
        delta_h = delta_h / norm
        logging.info(f"Steering vectors normalized with L2 norm: {norm} 归一化矩阵大小")
    return delta_h

if args.mean_type=="dif_mean":
    delta_h=SAE_decoding(sae,indices=steer_indices,latent_value_mean=steer_info[f"dif_{TARGET}-{SOURCE}_relu"]["latent_value_mean"],method="val_mul",is_norm=args.is_norm_delta_matrix)
elif args.mean_type=="tar_mean":
    delta_h=SAE_decoding(sae,indices=steer_indices,latent_value_mean=steer_info[f"dif_{TARGET}-{SOURCE}_relu"]["target_nz_mean"],method="val_mul",is_norm=args.is_norm_delta_matrix)
else:
    raise ValueError("Unsupported")


# %%
if args.debug==1:
    logging.info("delta_matrix: "+str(delta_h[:5])) #理论上这里有正有负比较正常
# %% [markdown]
# # 这里得到的就是delta_matricx



# Example prompt from the selected set
# example_prompt = "What really matters is that they know"
# example_prompt=""" Q: Cody goes to the store and buys $40 worth of stuff.  The taxes were 5%.  After taxes, he got an $8 discount.  Cody and his friend split the final price equally. How much did Cody pay?
# A:"""


########################################################### apply_control
# Generate without steering

from intervention_generation import run_generate
logging.info("Generating texts **without** steering... ")
logging.info("无转向结果")

with torch.no_grad():
    generated_texts_no_steer = run_generate(
        prompts=EXAMPLE_PROMPT_LIST, 
        sampling_kwargs=sampling_kwargs,
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        MAX_NEW_TOKENS=MAX_NEW_TOKENS,
        repeat_num=3,
        steer_type="",
        steer_on=False,
        alpha=0,
        delta_h=None,
        show_res=True)
logging.info("干预之后的结果")
# bef,aft=args.steer.split("-")
logging.info(f"干预方向{SOURCE}->{TARGET},礼貌任务下，neg=impolite，情感任务下 pos=积极情感")
logging.info("** Generating texts with steering... Target **")
logging.info(f"form {SOURCE} to {TARGET}")
logging.info("转向结果")
generated_texts_with_steer = run_generate(
    prompts=EXAMPLE_PROMPT_LIST, 
    sampling_kwargs=sampling_kwargs,
    sae=sae,
    model=model,
    tokenizer=tokenizer,
    MAX_NEW_TOKENS=MAX_NEW_TOKENS,
    repeat_num=3,
    steer_on=True,
    alpha=ALPHA,
    steer_type=STEER_TYPE,
    delta_h=delta_h,
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
    elif TASK=="polite":
        logging.info("In Domain: Calculate at A train dataset, Evaluate at A test dataset")
        from data_preprocess import load_and_prepare_polite_prompts
        prompts=load_and_prepare_polite_prompts(test_set=polite_test_set)
    elif TASK=="toxicity":
        logging.info("Out of Domain: Calculate at A dataset, Evaluate at B dataset")
        from data_preprocess import load_and_prepare_toxicity_prompts 
        prompts=load_and_prepare_toxicity_prompts(prompt_path=args.prompt_path,task=TASK)
    elif TASK=="debate":
        logging.info("Out of Domain: Calculate at A dataset, Evaluate at B dataset")
        from data_preprocess import load_and_prepare_debate_prompts
        prompts=load_and_prepare_debate_prompts(prompt_path=args.prompt_path,task=TASK)
    else:
        raise NotImplementedError("No Supported Task")    
    print(prompts)
    assert "neg" in prompts,"prompt steer source (pos/neg) not in prompts, please check the data_preprocess section"
    if args.prompt_source!="":
        logging.info(f"prompt的极性是{args.prompt_source}")
        prompts=prompts[args.prompt_source]
    else:
        prompts=prompts[SOURCE]
    
    param={**vars(args),**sampling_kwargs,"max_new_tokens":50,"steer":f"from {SOURCE} to {TARGET}"}
    param["alpha_recheck"]=ALPHA
    logging.info(f"Running with alpha: {ALPHA}")
    logging.info(f"Running with prompt_type: "+str(param["steer"]))

    
    # 打开文件（模式为追加模式 'a'）
    with jsonlines.open(os.path.join(OUTPUT_DIR,"params.jsonl"), mode='w') as writer:
        writer.write(param)  # 逐行写入
    if args.prompt_data_size!=-1:# 用于小批量网格搜索
        prompts = prompts.select(range(args.prompt_data_size))  # Correct slicing for datase
        logging.warning("截取prompt_datasize"+str(len(prompts)))
    
    # 新增批次大小参数（假设从args获取）

    with jsonlines.open(os.path.join(OUTPUT_DIR,"no_steer_gen_res.jsonl"), mode='w') as no_steer_f:
        with jsonlines.open(os.path.join(OUTPUT_DIR,"steer_gen_res.jsonl"), mode='w') as steer_f: 
            # 分批次处理prompts
            for batch_idx in tqdm(range(0, len(prompts), GEN_BATCH_SIZE)):
                # batch = prompts[batch_idx:batch_idx+GEN_BATCH_SIZE]
                batch = prompts.select(range(batch_idx, min(batch_idx + GEN_BATCH_SIZE, len(prompts))))
                batch_prompts = [item["prompt"]["text"] for item in batch]
                
                # 无转向生成
                if SAVE_NO_STEER == 1:
                    with torch.no_grad():
                        no_steer_gen_texts_batch = run_generate(
                            prompts=batch_prompts,  # 传入批次prompts
                            sampling_kwargs=sampling_kwargs,
                            sae=sae,
                            model=model,
                            tokenizer=tokenizer,
                            MAX_NEW_TOKENS=MAX_NEW_TOKENS,
                            repeat_num=args.repeat_num,
                            steer_type="",
                            steer_on=False,
                            alpha=0,
                            delta_h=None,
                            show_res=False
                        )
                    
                    # 写入无转向结果
                    for i, item in enumerate(batch):
                        no_steer_item = copy.deepcopy(item)
                        no_steer_item["generations"] = [{"text": text} for text in no_steer_gen_texts_batch[i]]
                        no_steer_f.write(no_steer_item)

                # 转向生成
                steered_texts_batch = run_generate(
                    prompts=batch_prompts, 
                    sampling_kwargs=sampling_kwargs,
                    sae=sae,
                    model=model,
                    tokenizer=tokenizer,
                    MAX_NEW_TOKENS=MAX_NEW_TOKENS,
                    repeat_num=args.repeat_num,
                    steer_on=True,
                    alpha=ALPHA,
                    steer_type=STEER_TYPE,
                    delta_h=delta_h,
                    show_res=False
                )
                
                # 写入转向结果
                for i, item in enumerate(batch):# 遍历每个batch中的元素
                    steer_item = copy.deepcopy(item)
                    steer_item["generations"] = [{"text": text} for text in steered_texts_batch[i]]
                    global_idx = batch_idx + i
                    if global_idx % 20 == 0:
                        logging.info(f"{TASK}: from {SOURCE} to {TARGET} prompt_set: {SOURCE}")
                        logging.info(steer_item)
                    steer_f.write(steer_item)

if args.debug==1:
    logging.info(f"debug mode,show example, no full dataset eval")
elif args.debug==0:
    if SAVE_NO_STEER==1:
        logging.info("Provide No Steer Result 提供无干预对照样本")
    eval_on_full_data()
else:
    raise ValueError("debug must be 0 or 1")
# %%
logging.info("训练时间"+str(end_train_time-start_train_time))
params = params_to_dict(args, is_print=True)
logging.info(f"{TASK}:{SOURCE}->{TARGET}")
with jsonlines.open(os.path.join(OUTPUT_DIR, "params.jsonl"), mode='w') as param_file:
    param_file.write(params)