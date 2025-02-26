import torch
from src.utils import load_or_cache_neuron_info
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


# import os
import pickle
from functools import wraps

def cache_delta_h_results(cache_dir="./.cache"):
    """
    缓存装饰器，基于 LLM 和 layer 参数自动缓存结果
    缓存文件路径：./.cache/<LLM>_layer<layer>.pkl
    """
    def decorator(func):
        @wraps(func)
        def wrapper(sae, LLM: str, layer: int, *args, **kwargs):
            # 创建缓存目录（如果不存在）
            os.makedirs(cache_dir, exist_ok=True)
            
            # 生成安全的文件名（替换特殊字符）
            safe_LLM = "".join(c if c.isalnum() else "_" for c in LLM)
            cache_path = os.path.join(
                cache_dir,
                f"{safe_LLM}_layer{layer}_delta_h.pkl"
            )
            
            # 检查缓存是否存在
            if os.path.exists(cache_path):
                print(f"Loading cached results from {cache_path}")
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            
            # 计算并缓存结果
            print(f"Computing new results (no cache found at {cache_path})")
            result = func(sae, LLM, layer, *args, **kwargs)
            
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Results cached to {cache_path}")
            
            return result
        return wrapper
    return decorator

# def test():
#     load_neuron_info(demo_neuron_cache_floder="")
# test()