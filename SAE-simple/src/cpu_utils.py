import torch
from torch import Tensor
from sae_lens import SAE
import tqdm
from typing import Tuple
import logging
def compute_latents_cpu(sae: SAE, model, texts: list, hook_point: str, device: str, batch_size: int) -> list:
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
    logging.info("Running model with cache to obtain hidden states")
    batch_latents = []

    # 使用 tqdm 显示进度条
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        sv_logits, cache = model.run_with_cache(batch_texts, prepend_bos=False, device=device)
        batch_hidden_states = cache[hook_point]
        logging.info(f"Batch {i // batch_size + 1}: Hidden states shape: {batch_hidden_states.shape}")

        logging.info(f"Encoding hidden states for batch {i // batch_size + 1}")
        # 假设 sae.encode 支持批量编码
        latents = sae.encode(batch_hidden_states)  # 形状: (batch_size, latent_dim)
        batch_latents.append(latents)
        

    logging.info(f"Total batches processed: {len(batch_latents)}")
    return batch_latents

def analyze_latents_cpu(batch_latents: Tensor, top_k_mean: int = 100, top_k_cnt: int = 100) -> Tuple[Tensor, Tensor, Tensor]:
    """
    分析潜在表示（latents）
    Args:
        batch_latents (Tensor): 批次的潜在表示。
        top_k_mean (int, optional): 基于均值选择的 top-k 索引的数量。默认为 100。
        top_k_cnt (int, optional): 基于计数选择的 top-k 索引的数量。默认为 100。

    Returns:
        Tuple[Tensor, Tensor, Tensor]: 包含非零元素的均值、计数和重叠索引的元组。
    """
    logging.info("Computing non-zero element counts")
    act_cnt = (batch_latents != 0).sum(dim=(0, 1))

    logging.info("Computing sum of non-zero elements")
    nz_sum = torch.where(batch_latents != 0, batch_latents, torch.tensor(0.0, device=batch_latents.device)).sum(dim=(0, 1))

    logging.info("Computing mean of non-zero elements")
    nz_mean = torch.where(act_cnt != 0, nz_sum / act_cnt, torch.tensor(0.0, device=batch_latents.device))

    logging.info("Selecting top-k indices based on nz_mean")
    nz_act_val, nz_val_indices = torch.topk(nz_mean, top_k_mean)
    logging.info(f"Top {top_k_mean} nz_mean values selected.")

    logging.info("Selecting top-k indices based on act_cnt")
    nz_cnt, cnt_indices = torch.topk(act_cnt, top_k_cnt)
    logging.info(f"Top {top_k_cnt} act_cnt values selected.")

    # logging.info("Finding overlapping indices between nz_mean and act_cnt top-k")
    # overlap_mask = torch.isin(nz_val_indices, cnt_indices)
    # overlap_indices = nz_val_indices[overlap_mask]
    # logging.info(f"Number of overlapping indices: {len(overlap_indices)}")
    # overlap_indices=overlap_indices
    return nz_mean, act_cnt,None

def get_activation_by_steer_cpu(texts:list,sae,model,device,batch_size,topk_mean,topk_cnt):
    """
    根据给定的文本列表获取激活值。
    arg:
    texts (list): 输入文本列表。
    return:
    dict: 包含非零均值和计数的字典。
    """
    hook_point = sae.cfg.hook_name

    # Compute latents with batch processing
    batch_latents = compute_latents_cpu(sae, model, texts, hook_point, device, batch_size)
    # 计算第二个维度的最大值
    max_dim1 = max(latent.shape[1] for latent in batch_latents)  # 第二个维度的最大值
    logging.info(f"最大长度:{max_dim1}")
    # 对每个 Tensor 进行填充（仅填充第二个维度）
    padded_latents_right = [
        torch.nn.functional.pad(latent, (0, 0, 0, max_dim1 - latent.size(1)), "constant", 0)
        for latent in batch_latents
    ]

    batch_latents_concatenated = torch.cat(padded_latents_right, dim=0)
    logging.info(f"Concatenated batch latents shape: {batch_latents_concatenated.shape}")

    # Analyze latents 
    nz_mean, act_cnt, _ = analyze_latents_cpu(batch_latents_concatenated, top_k_mean=topk_mean, top_k_cnt=topk_cnt)
    return {"latent_value_mean":nz_mean,"latent_frequency":act_cnt}