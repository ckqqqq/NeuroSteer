import os
from dotenv import load_dotenv
import logging
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