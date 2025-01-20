import torch
from scipy.stats import gaussian_kde

def autoencoder_loss(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    latent_activations: torch.Tensor,
    l1_weight: float,
) -> torch.Tensor:
    """选中的代码定义了一个名为 autoencoder_loss 的函数，该函数用于计算自编码器（Autoencoder）的损失。自编码器是一种无监督学习算法，用于学习数据的压缩表示。这个函数结合了两种损失：归一化均方误差（Normalized Mean Squared Error, NMSE）和归一化L1损失（Normalized L1 Loss），并通过一个权重参数 l1_weight 来平衡这两种损失。
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param l1_weight: weight of L1 loss
    :return: loss (shape: [1])
    """
    return (
        normalized_mean_squared_error(reconstruction, original_input)
        + normalized_L1_loss(latent_activations, original_input) * l1_weight
    )


def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    均方差!!!!
    """
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1) / (original_input**2).mean(dim=1)
    ).mean()

def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss (shape: [1])
    """
    return (latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)).mean()

epsilon=1e-10
import numpy as np
# 定义KL散度损失函数
from sklearn.neighbors import KernelDensity 

def kde_estimator(samples):
    samples_np = samples.detach().cpu().numpy()
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(samples_np)
    return kde

def kl_loss(positive_samples, generated_samples):
    kde_positive = kde_estimator(positive_samples)
    kde_generated = kde_estimator(generated_samples)
    
    generated_samples_np = generated_samples.detach().cpu().numpy()
    log_p = torch.tensor(kde_positive.score(generated_samples_np), dtype=torch.float32).to(positive_samples.device)
    log_q = torch.tensor(kde_generated.score(generated_samples_np), dtype=torch.float32).to(positive_samples.device)
    
    # 过滤掉NaN值
    mask = torch.isfinite(log_p) & torch.isfinite(log_q)
    log_p = log_p[mask]
    log_q = log_q[mask]
    
    kl_div = (log_p - log_q).mean()
    return kl_div