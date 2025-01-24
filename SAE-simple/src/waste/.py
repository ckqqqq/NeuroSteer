# %%
!export HF_ENDPOINT=https://hf-mirror.com
!export CUDA_VISIBLE_DEVICES=0

# %%
# package import
from torch import Tensor
from transformer_lens import utils
from functools import partial
from jaxtyping import Int, Float
import torch
# device setup
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

# %%
device="cpu"

# %%
from dotenv import load_dotenv
load_dotenv("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env")

# %%
from transformer_lens import HookedTransformer
import numpy as np
import torch
import plotly_express as px

from transformer_lens import HookedTransformer

# Model Loading

from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list

# Virtual Weight / Feature Statistics Functions
from sae_lens.analysis.feature_statistics import (
    get_all_stats_dfs,
    get_W_U_W_dec_stats_df,
)

# Enrichment Analysis Functions
from sae_lens.analysis.tsea import (
    get_enrichment_df,
    manhattan_plot_enrichment_scores,
    plot_top_k_feature_projections_by_token_and_category,
)
from sae_lens.analysis.tsea import (
    get_baby_name_sets,
    get_letter_gene_sets,
    generate_pos_sets,
    get_test_gene_sets,
    get_gene_set_from_regex,
)
import os
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
# model = HookedTransformer.from_pretrained(
#     "tiny-stories-1L-21M"
# )  # This will wrap huggingface models and has lots of nice utilities.


model = HookedTransformer.from_pretrained("gpt2-small",device=device)

# %%
from sae_lens import SAE
layer=5

gpt2_small_sparse_autoencoders = {}
gpt2_small_sae_sparsities = {}


sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id=f"blocks.{layer}.hook_resid_pre",  # won't always be a hook point
    device=device
)
gpt2_small_sparse_autoencoders[f"blocks.{layer}.hook_resid_pre"] = sae
gpt2_small_sae_sparsities[f"blocks.{layer}.hook_resid_pre"] = sparsity

# %%
from datasets import load_dataset
dataset = load_dataset("/home/ckqsudo/code2024/0dataset/emotional_classify/multiclass-sentiment-analysis")

# %%
dataset["train"]= dataset['train'].shuffle(seed=42)  # seed 用于固定随机性



# %%
train_data_text=dataset["train"]# 假设数据集是训练集（train），筛选 labels = 1 的数据
neg_train_set = dataset['train'].filter(lambda example: example['label'] == 0).select(range(1000))

pos_train_set=dataset['train'].filter(lambda example: example['label'] == 2).select(range(1000))

# %%
neu_train_set=dataset['train'].filter(lambda example: example['label'] == 1).select(range(1000))

# %%
len(neg_train_set),len(pos_train_set)

# %%
neg_train_set["text"][1],pos_train_set["text"][1]

# %%
hook_point = sae.cfg.hook_name

# %%
sv_logits, cache = model.run_with_cache(pos_train_set["text"][:1000], prepend_bos=True,device=device)

# %%
sv_logits.shape

# %%
batch_hidden_states=cache[hook_point]

# %%
batch_hidden_states.device

# %%
batch_hidden_states.shape

# %%
import torch

# 定义一个三维 Tensor
tensor = torch.tensor([
    [
    [0, 1, 0, 2],
    [3, 0, 0, 4],
    [0, 0, 0, 0]],

                       [[5, 0, 6, 0],
                        [0, 7, 0, 8],
                        [9, 0, 0, 0]]])

# 统计在 A 和 B 两个维度上不为 0 的元素，输出 C 维度的向量
nonzero_counts_C = (tensor != 0).sum(dim=(0, 1))
# # 1. 统计非零元素的数量
# latents_count = (batch_latents != 0).sum(dim=(0, 1))

# 打印结果
print(nonzero_counts_C,tensor.shape)

# %%
from matplotlib import axis
import torch
batch_latents=[]
for data_idx in range(0,batch_hidden_states.shape[0]):
    hidden_state=batch_hidden_states[0]
    # hidden_state.to("cuda")
    latents = sae.encode(hidden_state)
    batch_latents.append(latents)
batch_latents=torch.stack(batch_latents,axis=0)


# %%
# import torch

# # 定义矩阵
# matrix = torch.zeros(4,2)
# matrix[2,1]=4
# matrix[3,1]=9
# matrix[1,0]=-1
# # 统计每一列中不为 0 的元素的个数
# nonzero_counts = (matrix != 0).sum(dim=0)

# # 打印结果
# print(nonzero_counts)

# %%
batch_latents.shape

# %%
batch_latents.shape

# %%

# 1. 统计非零元素的数量
act_cnt = (batch_latents != 0).sum(dim=(0, 1))

# 2. 计算非零元素的总和
nz_sum = torch.where(batch_latents != 0, batch_latents, torch.tensor(0.0)).sum(dim=(0, 1))

# 3. 计算非零元素的均值
nz_mean = torch.where(act_cnt != 0, nz_sum / act_cnt, torch.tensor(0.0))

# %%
torch.max(act_cnt )

# %%
nz_mean[12555]

# %%
nz_act_val, nz_val_indices= torch.topk(nz_mean, 100)

# %%
nz_act_val,nz_val_indices

# %%
nz_val_indices

# %%
nz_cnt, cnt_indices = torch.topk(act_cnt, 200)

# nonzero_indices = torch.nonzero(latents_count > 0).squeeze()

# %%
overlap_indices=nz_val_indices[torch.isin(nz_val_indices,cnt_indices)]

# %%
len(overlap_indices)

# %%
import torch

# 创建两个 Tensor
tensor1 = torch.tensor([1, 2, 3, 4, 5])
tensor2 = torch.tensor([4, 5, 6, 7, 8,9])

# 检查 tensor1 中的元素是否在 tensor2 中
mask = torch.isin(tensor1, tensor2)

# 提取重复元素
intersection = tensor1[mask]

# 打印结果
print("重复元素:", intersection,mask.shape)

# %%


# %%
def check_acts_by_index(prompt,acts_idx,hook_point):
    sv_logits, cache = model.run_with_cache(prompt, prepend_bos=True)
    # 转换为对应的令牌表示
    tokens = model.to_tokens(prompt)
    print(tokens)

    # get the feature activations from our SAE 获取特定位置的中间状态,并对中间状态进行编码，得到特征激活
    sv_feature_acts = sae.encode(cache[hook_point])
    print(sv_feature_acts.shape)
    return sv_feature_acts[:,:,acts_idx-1:acts_idx+1]


# %%


# %%
sae.W_dec.shape

# %%
method="val_mul"

# %%
def steering_vectors(method):
    if method=="mean":
        steering_vectors=Tensor.mean(sae.W_dec[overlap_indices],axis=0)
    elif method=="val_mul":
        steering_vectors=torch.zeros(768)
        for i,important_idx in enumerate(overlap_indices):
            assert nz_mean[important_idx]>2
            steering_vectors+=nz_mean[important_idx]*sae.W_dec[important_idx]
    return steering_vectors
            
# d_hidden=sae.W_dec.shape[1]
# d_latent=sae.W_dec.shape[0]

# %%
mean_steering_vec=steering_vectors("mean")
mul_steering_vec=steering_vectors("val_mul")

# %%
"欧几里得距离",torch.norm(mean_steering_vec - mul_steering_vec, p=2)

# %%
import torch.nn.functional as F
"余弦",F.cosine_similarity(mean_steering_vec.unsqueeze(0), mul_steering_vec.unsqueeze(0))

# %%
steering_vectors=mul_steering_vec
print("注意你最后选择的干预向量",method)


# %%
# 

from ast import Raise


def steering_hook(resid_pre, hook):
    if resid_pre.shape[1] == 1:
        return

    # position = steer_prompt_id.shape[1]
    # raise ValueError(f"steering{resid_pre.shape}")
    if steering_on:
        # using our steering vector and applying the coefficient
        # position=steer_prompt_id.shape[-1]
        resid_pre[:, :, :] += coeff * steering_vectors


def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            stop_at_eos=True,  # avoids a bug on MPS
            input=tokenized,
            max_new_tokens=50,
            do_sample=True,
            **kwargs,
        )
    return result
def run_generate(example_prompt):
    model.reset_hooks()
    editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook)]
    res = hooked_generate(
        [example_prompt] * 3, editing_hooks, seed=None, **sampling_kwargs
    )# batch=3, e

    # Print results, removing the ugly beginning of sequence token
    res_str = model.to_string(res[:, 1:])
    print(("\n\n" + "-" * 80 + "\n\n").join(res_str))

# %%
sampling_kwargs = dict(temperature=1.0, top_p=0.5, freq_penalty=1.0)

# %%
example_prompt=neu_train_set["text"][1]

# %%
example_prompt

# %%


# %%
steering_on=False
example_prompt="I feel"
coeff=0
run_generate(example_prompt=example_prompt)

# %%
steering_on=True
# example_prompt="hey! I feel"
coeff=10
run_generate(example_prompt=example_prompt)


