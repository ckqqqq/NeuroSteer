

from src.denoising import denosing
from src.intervention_generation import run_generate, SAE_decoding

from src.model_preprocess import load_models
from src.utils import load_or_cache_neuron_info

from sae_lens import SAE
import torch
import os
def check_gpu():
    if torch.cuda.is_available():
        print("GPU可用！")
        return "cuda:0"
    else:
        print("GPU不可用！")
        return "cpu"
def load_neuron_info(demo_neuron_cache_floder):
    caches_path=os.listdir(demo_neuron_cache_floder)
    assert len(caches_path)==4,"four neuron caches are needed, including sentiment, toxicity, stance, and polite"
    print(caches_path)
    task_info=[]
    for cache_path in caches_path:
        info={}
        info["model"]=cache_path.split("_")[0]
        assert cache_path.split("_")[2]=="layer",("cache_path must contain layer")
        assert cache_path.split("_")[-2]=="topK",("cache_path must contain topK") 
        info["layer"]=int(cache_path.split("_")[3])
        info["task"]=cache_path.split("_")[1]
        info["topK"]=int(cache_path.split("_")[-1])
        neuron_cache=[i for i in os.listdir(os.path.join(demo_neuron_cache_floder,cache_path)) if i.endswith(".pkl")]
        assert len(neuron_cache)==1,"only one neuron cache is needed"
        file=neuron_cache[0]
        path=os.path.join(demo_neuron_cache_floder,cache_path)
        neuron_info=load_or_cache_neuron_info(cache_dir=path,cache_filename=file,use_cache=True,compute_func=None)
        info["neuron_info"]=neuron_info
        task_info.append(info)
        
    assert len(set([(i["layer"],i["model"]) for i in task_info[:]]))==1,"layer and model must be the same"
    return task_info


def model_preprocess(LLM = "gpt2-small",layer = 6):
    model, sae, tokenizer = load_models(LLM=LLM, layer=layer, device=check_gpu())
    return model,sae,tokenizer


def delta_h(sae:SAE,LLM, layer: int):
    task_info = load_neuron_info(
        demo_neuron_cache_floder="/home/ckqsudo/code2024/CKQ_ACL2024/NeuroSteer/results/demo/demo_v1"
    )
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
        MAX_NEW_TOKENS=30,
        steer_type="all",
        repeat_num=1,
        show_res=True,
    )
    print(res)
    return res[0][0]



