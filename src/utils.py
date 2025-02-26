import os
from dotenv import load_dotenv
import logging

# %%
import pickle
import torch


# import partial
def load_or_cache_neuron_info(cache_dir, use_cache: bool, cache_filename, compute_func):
    """
    加载或计算 neuron_info 的缓存机制

    :param cache_filename: 缓存文件的名称
    :param compute_func: 计算 neuron_info 的函数
    :return: neuron_info 字典
    """
    cache_path = os.path.join(cache_dir, cache_filename)
    # 检查缓存文件是否存在
    if use_cache and os.path.exists(cache_path):
        logging.info(f"从缓存 {cache_path} 中加载 neuron_info")
        import time

        time.sleep(3)
        with open(cache_path, "rb") as f:
            neuron_info = pickle.load(f)
    else:
        if use_cache:
            logging.info(f"强制覆写 {cache_path}" + "，重新计算 neuron_info")
        else:
            logging.info(f"缓存 {cache_path} 不存在" + "，缓存 neuron_info")
        assert compute_func is not None, "警告，强制覆写cache，check your code"
        neuron_info = compute_func()
        # 将计算结果保存到缓存文件中 (反复cache)
        with open(cache_path, "wb") as f:
            pickle.dump(neuron_info, f)
        logging.info(f"neuron_info 已保存到缓存 {cache_path}")
    return neuron_info


def params_to_dict(args, is_print=True):
    # 将参数转换为字典
    hyperparams = vars(args)
    # Log hyperparameters
    if is_print:
        logging.info("Show Hyperparameters: \n\n")
        for key, value in hyperparams.items():
            logging.info(f"  {key}: {value}")
    return hyperparams


def load_environment(env_path: str):
    load_dotenv(env_path)
    hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    logging.info(f"HF_ENDPOINT: {hf_endpoint}")


def half_gaussian_kernel(half_len):
    """返回高斯掩码

    Args:
        half_len (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 设置均值和标准差
    cov_len = 2 * half_len
    mean = cov_len // 2  # 正态分布的均值
    std = cov_len / 6  # 设置标准差，可以根据需要调整

    # 创建正态分布
    x = torch.arange(cov_len, dtype=torch.float32)
    kernel = torch.exp(-0.5 * ((x - mean) / std) ** 2)
    # print(kernel)

    # 仅保留正态分布的前半部分（右侧值设置为0）
    kernel[int(cov_len // 2) :] = 0  # 保留前半部分，右半部分置为零

    # 归一化，确保总和为 1
    kernel = kernel / kernel.sum()
    return kernel[:half_len]
