import argparse
import os
import re
import json
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


parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='debate',choices=['sentiment','cot','polite','debate'], help='Task type: sentiment, cot, polite, debate')
parser.add_argument('--layer', type=int, default=6, help='Layer number to analyze')
parser.add_argument('--LLM', type=str, default='gpt2-small', help='LLM model')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--data_size', type=str, default='ALL', help='Data size')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps', 'auto'], help='Device to use')
parser.add_argument('--alpha', type=int, default=100, help='Alpha parameter')
parser.add_argument('--steer', type=str, default='opp-sup', help='Steering direction')
parser.add_argument('--method', type=str, default='val_mul', choices=['mean', 'val_mul'], help='Method to use')
parser.add_argument('--topk_mean', type=int, default=100, help='Top K mean selection')
parser.add_argument('--topk_cnt', type=int, default=100, help='Top K count selection')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--source', type=str, default='sup', help='Source class')
parser.add_argument('--target', type=str, default='opp', help='Target class')
parser.add_argument('--mean_type', type=str, default='dif_mean', help='Mean type')
parser.add_argument('--steer_type', type=str, default='last', help='Steer type')
parser.add_argument('--debug', type=bool, default=False, help='Debug flag')
parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
parser.add_argument('--dataset_path', type=str, default="/home/ckqsudo/code2024/0dataset/baseline-acl/data/debate/StanceSentences", help='Dataset path')
parser.add_argument('--prompt_path', type=str, default="/home/ckqsudo/code2024/0dataset/baseline-acl/data/debate/ibm_debate", help='Prompt path')
parser.add_argument('--env_path', type=str, default="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env", help='Environment path')
parser.add_argument("--save_compared", type=bool, default=False, help='Save compared')

args = parser.parse_args()


sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)
sampling_kwargs['verbose']=False
TASK =args.task
STEER_TYPE=args.steer_type
ALPHA=args.alpha
MAX_NEW_TOKENS=50


output_dir=os.path.join(args.output_dir,f"alpha_{args.alpha}_from_{args.source}_to_{args.target}_datasize_{args.data_size}_layer_{args.layer}_mean_{args.mean_type}_steertype_{args.steer_type}_device_{args.device}_batchsize{args.batch_size}")
os.makedirs(output_dir,exist_ok=True)

# Setup logging
setup_logging(output_dir)
# Save hyperparameters
hyperparams = vars(args)
# Log hyperparameters
logging.info("Hyperparameters:")
for key, value in hyperparams.items():
    logging.info(f"  {key}: {value}")


load_environment(args.env_path)
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
elif TASK=="debate":
    logging.info("debate"*10)
    sup_train_set, opp_train_set,val_set,test_set=load_and_prepare_debate_triple_dataset(args.dataset_path, args.seed, args.data_size)
else:
    raise ValueError("No Supported")

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
    lat_freq,lat_val_sum=torch.zeros(SAE_LATENT_SIZE).to("cpu"),torch.zeros(SAE_LATENT_SIZE).to("cpu")# 避免OOM
    # 使用 tqdm 显示进度条
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            logging.info(f"Batch {i // batch_size + 1}: batch_size {batch_size}")
            try:
                sv_logits, cache = model.run_with_cache(batch_texts, prepend_bos=False, device="cuda")
            except Exception as e:
                logging.error(f"Error processing batch {i // batch_size + 1}: {e}")
                raise ValueError(str([len(i) for i in batch_texts]))
            batch_hidden_states = cache[hook_point]
            logging.info(f"Batch {i // batch_size + 1}: Hidden states shape: {batch_hidden_states.shape}")

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

def get_activation_by_steer(texts:list):
    hook_point = sae.cfg.hook_name
    # Compute latents with batch processing
    lat_info=compute_latents(sae, model, texts, hook_point, args.device, args.batch_size)
    return {"latent_value_mean":lat_info["latent_value_mean"],"latent_frequency":lat_info["latent_frequency"]}

steer_info={}
from functools import partial
if TASK=='polite' or TASK=="sentiment":
    if args.device=="cpu":
        logging.info("CPU处理")
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
elif TASK=='debate':
    if args.device=="cpu":
        logging.info("CPU处理")
        from cpu_utils import get_activation_by_steer_cpu
        get_activation_by_steer_cpu=partial(get_activation_by_steer_cpu,sae=sae,model=model,device=args.device,batch_size=args.batch_size,top_k_mean=args.topk_mean,top_k_cnt=args.topk_cnt)
    logging.info("from"+args.source+"to"+args.target)
    logging.info(f"support")
    text=sup_train_set["text"]
    steer_info["sup"]=get_activation_by_steer(text)
    logging.info(f"oppose")
    text=opp_train_set["text"]
    steer_info["opp"]=get_activation_by_steer(text)
    
source=args.source
target=args.target
# 调整样本正负性在这里调整 从负样本到正样本还是从正样本()到负样本
# pos 代表积极情绪
# neg 代表消极情绪

logging.info(f"转向方向 dif_{target}-{source}_relu")
steer_info[f"dif_{target}-{source}_relu"]={"latent_frequency":torch.relu(steer_info[target]["latent_frequency"]-steer_info[source]["latent_frequency"]),"latent_value_mean":torch.relu(steer_info[target]["latent_value_mean"]-steer_info[source]["latent_value_mean"]),"target_nz_mean":torch.relu(steer_info[target]["latent_value_mean"])}

top_k=args.topk_cnt
_,steer_indices=torch.topk(steer_info[f"dif_{target}-{source}_relu"]["latent_frequency"],top_k)

# 假设 steer_info[f"dif_{b}-{a}_relu"]["latent_frequency"] 是一个 NumPy 数组
lat_freq = steer_info[f"dif_{target}-{source}_relu"]["latent_frequency"]
# 先获取非零元素的索引
lat_acti_indices = np.nonzero(lat_freq)


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

def run_generate(example_prompt,sampling_kwargs,steer_on,alpha,steer_type="last",repeat_num=1,show_res=False):
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
    if show_res:
        for idx, text in enumerate(res_str):
            logging.info(f"Generated Text: {idx+1}:\n{text}")
        
    return res_str


from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()
deepseek_api_key = os.getenv('OPENAI_API_KEY')
def get_deepseek_eval(text):
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant, you need to help me determine the stance of a given sentence, whether it conveys a supportive or an opposing tone. Your answer can only be one of the two words 'support' or 'oppose'."},
            {"role": "user", "content": text},
        ],
        stream=False
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def load_and_prepare_debate_prompts(prompt_path:str,task:str):
    assert task in ["debate"],"请输入正确的任务"
    prompts = load_dataset(prompt_path)
    sup_train_set = prompts['test'].filter(lambda example: example['label'] == 'support')
    opp_train_set = prompts['test'].filter(lambda example: example['label'] == 'oppose')
    return sup_train_set['text'],opp_train_set['text']


sup_prompt, opp_prompt = load_and_prepare_debate_prompts(prompt_path='/home/ckqsudo/code2024/0dataset/baseline-acl/data/debate/StanceSentences',task='debate')

final_eval_results = []
for text in sup_prompt:
    origin_text = text
    text = text[:len(text)//2] if len(text) > 30 else text
    generated_text_no_steer = run_generate(text, sampling_kwargs,steer_on=False,alpha=0,show_res=True)
    generated_text_with_steer = run_generate(text, sampling_kwargs,steer_on=True,alpha=args.alpha,steer_type=args.steer_type,show_res=True)
    no_steer_eval = get_deepseek_eval(generated_text_no_steer[0])
    with_steer_eval = get_deepseek_eval(generated_text_with_steer[0])
    final_eval_results.append({"origin_text": origin_text,
                              "no_steer_text": generated_text_no_steer[0],
                              "with_steer_text": generated_text_with_steer[0],
                              "no_steer_eval": no_steer_eval,
                              "with_steer_eval": with_steer_eval})

opp_num_no_steer, opp_num_with_steer = 0,0
for item in final_eval_results:
    if 'oppose' in item['no_steer_eval']:
        opp_num_no_steer += 1
    if 'oppose' in item['with_steer_eval']:
        opp_num_with_steer += 1
    
    
with open (f'/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/results/debate_eval_results/eval_results_{args.alpha}_{args.steer_type}_from{opp_num_no_steer}_to{opp_num_with_steer}.json','w') as f:
    json.dump(final_eval_results,f,ensure_ascii=False,indent=4)