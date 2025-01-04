# from typing import Callable, Any

# import torch
from ast import Tuple
from typing import Callable,Any
import torch
import torch.nn as nn
import torch.nn.functional as F



def layer_norm(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """选中的代码定义了一个名为 LN 的函数，用于对输入的张量 x 进行层归一化（Layer Normalization）操作。层归一化是一种在神经网络中常用的归一化技术，用于稳定训练过程并加速收敛。

    Args:
        x (torch.Tensor): _description_
        eps (float, optional): _description_. Defaults to 1e-5.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
    """
    x_mean = x.mean(dim=-1, keepdim=True)
    #归一化的第一步是将数据中心化（减去均值），使得数据的均值为 0。
    x = x - x_mean
    # 归一化的第二步是将数据缩放到一个固定的范围内，通常是将标准差缩放到 1。
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    # 了层归一化的操作加上一个小的正数 eps，通过减去均值并除以标准差
    return x, x_mean, std

import torch
import torch.nn as nn


class JumpReLUSAE(nn.Module):
    """这是Jump ReLU SAE，不同于传统的ReLU函数，他的设置了一个阈值用于激活函数，当输入大于阈值时，激活函数输出为输入值，否则输出为0。（不过这里需要层归一化吗）

    Args:
        nn (_type_): _description_
    """
    def __init__(self, n_input, n_sae):
        """
        ### 存在的优化方向
        * +
        Args:
            d_model (_type_): 这玩意就是需要编码的模型隐藏层的维度
            d_sae (_type_): 这个就是需要编码的SAE的维度，需要升维度的玩意
        """
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__()
        self.encoder_w = nn.Parameter(torch.zeros(n_input, n_sae)) #编码器矩阵
        self.decoder_w = nn.Parameter(torch.zeros(n_sae, n_input)) #解码器矩阵
        self.threshold = nn.Parameter(torch.zeros(n_sae))# 就连这个阈值都是学出来的
        self.encoder_bias = nn.Parameter(torch.zeros(n_sae))
        self.decoder_bias = nn.Parameter(torch.zeros(n_input))

    def encode(self, input_activations):
        # 编码器的线性变换  
        pre_acts = input_activations @ self.encoder_w + self.encoder_bias
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts) # relu 可以将负数变成0，这样就可以实现mask 的效果
        # 
        return acts

    def decode(self, acts):
        return acts @ self.decoder_w + self.decoder_bias

    def forward(self, acts):
        latents = self.encode(acts)
        recon = self.decode(latents)
        return recon


class TopKSAE(nn.Module):
    """当然可以作为普通SAE使用，如果传入的activation_fn不指定的话
    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """
    def __init__(
        self, n_sae: int, n_input: int,is_layer_norm: bool = False,is_weight_tied: bool=False,
        activation_fn: Callable = nn.ReLU()
    ) -> None:
        super.__init__()
        self.pre_bias=nn.Parameter(torch.zeros(n_input))# 可以学习的参数encoder前的bias
        self.sae_bias=nn.Parameter(torch.zeros(n_sae))# 可学习的参数
        self.activation_fn=activation_fn# 激活函数
        self.encoder=nn.Linear(n_input,n_sae,bias=False)
        if is_weight_tied==False:
            self.decoder=nn.Linear(n_sae,n_input,bias=False)
        else:
            self.decoder=TiedEncoderTranspose(self.encoder)
        
        self.is_layer_norm=is_layer_norm
    
        
    def pre_process(self,x:torch.Tensor)->tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, x_mean, std = layer_norm(x)
        return x, dict(x_mean=x_mean, std=std)
    
    def pre_activation(self,x:torch.Tensor)->torch.Tensor:
        x=x-self.pre_bias
        # 首先使其均值为0
        latents_pre_activations=F.linear(
            input=x,weight=self.encoder.weight,bias=self.encoder)
        return latents_pre_activations
    def encode(self,x:torch.Tensor)->torch.Tensor:
        x,x_info=self.pre_process(x)
        pre_activations=self.pre_activation(x)# 预激活
        latents=self.activation_fn(pre_activations)# 使用一般relu或者topk获取latents表示
        return pre_activations,latents,x_info
    
    def decode(self,x:torch.Tensor,info:dict[str,Any]|None=None)->torch.Tensor:
        recons=self.decoder(x)+self.pre_bias
        if self.normalize:
            assert info is not None
            recons=recons*info["std"]+info["x_mean"]
            # 层归一化
        return recons
    
    def forward(self,x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        pre_activations,latents,x_info=self.encode(x)
        recons=self.decode(latents,x_info)
        return pre_activations,latents,recons
    def check_params(self):
        """检查参数是否正确,尤其是topk函数必须使用layer normalized的参数
        """
        if isinstance(self.activation_fn,nn.ReLU):
            print("ReLU")
        elif isinstance(self.activation_fn,TopK):
            assert self.is_layer_norm==True,"TopK () must use layer normalized parameters to make the dicrete processing continous"
        else:
            raise ValueError("activation_fn must be ReLU or TopK")


class TiedEncoderTranspose(nn.Module):
    """这是一个自定义的权重绑定函数，传入encoder并绑定之，用于绑定encoder 和 decoder，绑定技巧在于防止过拟合，节省较大维度时候的参数量，帮助学习对称性。但是基于论文第17页，他们只在初始化的时候绑定了    

    Args:
        nn (_type_): _description_
    """

    def __init__(self, linear: nn.Linear):
        """为什么不用 nn.Linear(n_latents, n_inputs, bias=False)

        Args:
            linear (nn.Linear): _description_
        """
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class TopK(nn.Module):
    """自定义的激活函数 
    只保留前k个最大的激活值，其余置为0。论文中使用的k的值是dmodel/2的平方
    Args:
        nn (_type_): _description_
    """
    def __init__(self, k: int=2048*2048/2/2, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result        