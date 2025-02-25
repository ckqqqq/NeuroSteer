import os
from dotenv import load_dotenv
import logging
# %%
import os
import pickle
# import partial
def load_or_cache_neuron_info(CACHE_DIR,args,cache_filename, compute_func):
    """
    加载或计算 steer_info 的缓存机制

    :param cache_filename: 缓存文件的名称
    :param compute_func: 计算 steer_info 的函数
    :return: steer_info 字典
    """
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    # 检查缓存文件是否存在
    if args.use_cache==1 and os.path.exists(cache_path):
        logging.info(f"从缓存 {cache_path} 中加载 steer_info")
        import time
        time.sleep(3)
        with open(cache_path, 'rb') as f:
            steer_info = pickle.load(f)
    else:
        if args.use_cache:
            logging.info(f"强制覆写 {cache_path}"+"，重新计算 steer_info")
        else:
            logging.info(f"缓存 {cache_path} 不存在"+"，缓存 steer_info")
        steer_info = compute_func()
        # 将计算结果保存到缓存文件中 (反复cache)
        with open(cache_path, 'wb') as f:
            pickle.dump(steer_info, f)
        logging.info(f"steer_info 已保存到缓存 {cache_path}")
    return steer_info


def params_to_dict(args,is_print=True):
    hyperparams=vars(args)
    # Log hyperparameters
    if is_print:
        logging.info("Show Hyperparameters: \n\n")
        for key, value in hyperparams.items():
            logging.info(f"  {key}: {value}")
    return hyperparams

def load_environment(env_path: str):
    load_dotenv(env_path)
    hf_endpoint = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
    logging.info(f"HF_ENDPOINT: {hf_endpoint}")
    
import torch.nn.functional as F
import torch

def half_gaussian_kernel(half_len):
    """返回高斯掩码

    Args:
        half_len (_type_): _description_

    Returns:
        _type_: _description_
    """
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