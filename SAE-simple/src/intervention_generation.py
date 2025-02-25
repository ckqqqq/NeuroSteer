
# f"blocks.{args.layer}.hook_resid_post"
import einops
import logging
import torch
from transformer_lens import HookedTransformer
steer_cnt=0
from sae_lens import SAE
from transformers import AutoTokenizer
def steering_hook(resid_pre, hook,steer_on, alpha, delta_h,steer_type="last"):
    if resid_pre.shape[1] == 1:
        return
    # 判断是否进行干预
    if steer_on:
        if steer_type == "last":
            # 对最后一个token前的部分应用干预，使用给定的 delta_matrix
            resid_pre[:, :-1, :] += alpha * delta_h# best
            # 如果提前干预效果会更好更连贯
        elif steer_type == "gaussian": # fail exploration
            # 使用高斯卷积对输入进行干预
            from utils import half_gaussian_kernel
            d_m=torch.clone(delta_h)
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
            resid_pre[:, :, :] += alpha * delta_h# 全部干预
        elif steer_type == "last2":
            resid_pre[:, :-2, :] += alpha * delta_h # 提前两个token进行干预
        else:
            raise ValueError("Unknown steering type")

def hooked_generate(prompt_batch,MAX_NEW_TOKENS,tokenizer,model, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        # kwargs["pad_token_id"]=tokenizer.pad_token_id  # 确保填充 token 正确
        tokenized = model.to_tokens(prompt_batch,prepend_bos=True)
        # tokenized = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True)
        result = model.generate(
            # stop_at_eos=True,  # avoids a bug on MPS
            prepend_bos=True,
            eos_token_id=tokenizer.eos_token_id,
            input=tokenized,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            **kwargs,
        )
        
    return result
from functools import partial


def run_generate(prompts:list, sampling_kwargs, steer_on:bool, alpha:float,delta_h,model:HookedTransformer,sae:SAE,tokenizer:AutoTokenizer,MAX_NEW_TOKENS:int, steer_type="last", repeat_num=3, show_res=False):
    model.reset_hooks()
    if steer_on:
        steering_hook_fn = partial(steering_hook, steer_on=steer_on, alpha=alpha, steer_type=steer_type,delta_h=delta_h)
        editing_hooks = [(sae.cfg.hook_name, steering_hook_fn)]
    else:
        logging.info("无干预")
        editing_hooks = []
    assert isinstance(prompts,list) and isinstance(prompts[0],str),"example_prompts必须是list[str]"
    # 构建重复采样prompt列表（保持原始顺序） 简单来说就是保存顺序*repeat_num
    repeated_prompts = [
        p for p in prompts
        for _ in range(repeat_num)
    ]
    
    # 批次生成
    res = hooked_generate(
        prompt_batch=repeated_prompts,
        fwd_hooks=editing_hooks,
        seed=None,
        MAX_NEW_TOKENS=MAX_NEW_TOKENS,
        tokenizer=tokenizer,
        model=model,
        **sampling_kwargs
    )
    
    # 转换并分组结果
    # res_str_batch = model.to_string(res)
    # 草，这是减少EOS的关键，似乎EOS没啥影响
    res_str_batch=model.tokenizer.batch_decode(res, clean_up_tokenization_spaces=False,skip_special_tokens=True)
    
    # 简单来说就是对字典基于repeat_num进行切片
    grouped_results = [
        res_str_batch[i*repeat_num : (i+1)*repeat_num] 
        for i in range(len(prompts))
    ]
    if show_res:
        logging.info(f"当前批次共处理{len(prompts)}个prompt")
        for prompt_idx, (prompt, gens) in enumerate(zip(prompts, grouped_results)):
            logging.info(f"Prompt {prompt_idx+1}: |{prompt[:60]}|")
            for gen_idx, gen in enumerate(gens):
                logging.info(f"生成 {gen_idx+1}: |{gen[len(prompt):][:]}|")
    
    return grouped_results