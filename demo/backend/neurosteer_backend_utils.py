

from src.denoising import denosing
from src.intervention_generation import run_generate, SAE_decoding

from src.model_preprocess import load_models
from src.utils import load_or_cache_neuron_info

from sae_lens import SAE
import torch
import os
from demo.backend.cache_config import discover_caches
def check_gpu():
    if torch.cuda.is_available():
        print("GPU可用！")
        return "cuda:0"
    else:
        print("GPU不可用！")
        return "cpu"
def load_neuron_info(model_name="gpt2-small", layer=6):
    task_info = []
    for task_name, cache_path in discover_caches(model_name, layer).items():
        neuron_info = load_or_cache_neuron_info(
            cache_dir=str(cache_path.parent), cache_filename=cache_path.name,
            use_cache=True, compute_func=None,
        )
        task_info.append({"model": model_name, "layer": layer, "task": task_name,
                          "neuron_info": neuron_info})
    return task_info


def model_preprocess(LLM = "gpt2-small",layer = 6):
    model, sae, tokenizer = load_models(LLM=LLM, layer=layer, device=check_gpu())
    return model,sae,tokenizer


def delta_h(sae:SAE,LLM, layer: int):
    task_info = load_neuron_info(LLM, layer)
    assert LLM == task_info[0]["model"] and task_info[0]["layer"] == layer, (
        "model and layer must be the same, check path"
    )
    delta_h_info = {}
    for t in task_info:
        # neuron_info={}
        task_name = t["task"]
        for target, source in [("pos", "neg"), ("neg", "pos")]:
            dif_res = denosing(
                t["neuron_info"], target=target, source=source, top_k=100
            )
            delta_h = SAE_decoding(
                sae=sae,
                target_neuron_indices=dif_res["topk_freq_dif"],
                latent_value_mean=dif_res["latent_value_mean"],
                is_norm=0,
            )
            delta_h_info[f"{task_name}_{target}-{source}"] = delta_h
    print(delta_h_info.keys())
    return delta_h_info




def neuron_steer_generate(alphas: dict, prompt: str,model,sae,tokenizer,delta_h_info):
    
    sampling_kwargs = {
        "temperature": 0.9,
        "top_p": 0.3,
        "freq_penalty": 1.0,
        "verbose": False,
    }
    max_new_tokens= 30

    real_delta_h = None
    real_alpha = 1
    steer_on = False
    for k, alpha in alphas.items():
        if alpha != 0:
            if real_delta_h is None:
                real_delta_h = delta_h_info[k] * alpha
            else:
                real_delta_h += delta_h_info[k] * alpha
            steer_on = True

    res = run_generate(
        prompts=[prompt],
        sampling_kwargs=sampling_kwargs,
        steer_on=steer_on,
        alpha=real_alpha,
        delta_h=real_delta_h,
        model=model,
        sae=sae,
        tokenizer=tokenizer,
        MAX_NEW_TOKENS=max_new_tokens,
        steer_type="all",
        repeat_num=1,
        show_res=True,
    )
    print(res)
    return res[0][0]



