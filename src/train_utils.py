
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 设置自动求导异常检测（可选）
torch.autograd.set_detect_anomaly(True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.init as init
# 确保使用中文字体（可选）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 适用于Mac，Windows用户可尝试 'SimHei'

# 自定义数据集
class VectorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.float()
        self.labels = labels.float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, output_size=128):
        super(MLP, self).__init__()
        # Encoder部分
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Decoder部分
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化网络的权重和偏置，使得网络在没有训练的情况下尽量使输入和输出相同
        """
        # 对每一层进行初始化
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                # 使用正交初始化初始化权重
                init.orthogonal_(module.weight)
                # 将偏置初始化为0
                if module.bias is not None:
                    init.zeros_(module.bias)
        
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# # 定义MMD损失函数
# class MMDLoss(nn.Module):
#     def __init__(self, kernel_mul=2.0, kernel_num=10, fix_sigma=None):
#         super(MMDLoss, self).__init__()
#         self.kernel_num = kernel_num
#         self.kernel_mul = kernel_mul
#         self.fix_sigma = fix_sigma

#     def guassian_kernel(self, source, target):
#         n_samples = int(source.size()[0]) + int(target.size()[0])
#         total = torch.cat([source, target], dim=0)
#         total0 = total.unsqueeze(0).repeat(total.size(0), 1, 1)
#         total1 = total.unsqueeze(1).repeat(1, total.size(0), 1)
#         L2_distance = ((total0 - total1) ** 2).sum(2)  # [n_samples, n_samples]

#         if self.fix_sigma:
#             bandwidth = self.fix_sigma
#         else:
#             bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
#         bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
#         bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
#         kernel_val = [torch.exp(-L2_distance / (2 * bw)) for bw in bandwidth_list]
#         return sum(kernel_val)  # [n_samples, n_samples]

#     def forward(self, source, target):
#         '''
#         source: Reconstructed negative samples [n, d]
#         target: Positive samples [m, d]
#         '''
#         batch_size = int(source.size()[0])
#         kernels = self.guassian_kernel(source, target)
#         XX = kernels[:batch_size, :batch_size]
#         YY = kernels[batch_size:, batch_size:]
#         XY = kernels[:batch_size, batch_size:]
#         YX = kernels[batch_size:, :batch_size]
#         loss = torch.mean(XX + YY - XY - YX)
#         return loss

# # 自定义损失函数
# class CustomLoss(nn.Module):
#     def __init__(self, mmd_loss, alpha=1.0):
#         super(CustomLoss, self).__init__()
#         self.mse = nn.MSELoss()
#         self.mmd_loss = mmd_loss
#         self.alpha = alpha  # 权重因子

#     def forward(self, outputs, targets, labels):
#         # 标签为1的样本使用MSE损失
#         if (labels == 1).sum() > 0:
#             loss_positive = self.mse(outputs[labels == 1], targets[labels == 1])
#         else:
#             loss_positive = torch.tensor(0.0).to(outputs.device)

#         # 标签为0的样本使用MMD损失
#         if (labels == 0).sum() > 0:
#             reconstructed_neg = outputs[labels == 0]
#             # 获取正样本的重建结果（在同一batch中）
#             reconstructed_pos = targets[labels == 1]
#             if reconstructed_pos.size(0) > reconstructed_neg.size(0):
#                 indices = torch.randperm(reconstructed_pos.size(0))[:reconstructed_neg.size(0)]
#                 reconstructed_pos = reconstructed_pos[indices]
#             elif reconstructed_neg.size(0) > reconstructed_pos.size(0):
#                 indices = torch.randperm(reconstructed_neg.size(0))[:reconstructed_pos.size(0)]
#                 reconstructed_neg = reconstructed_neg[indices]
#             loss_negative = self.mmd_loss(reconstructed_neg, reconstructed_pos)
#         else:
#             loss_negative = torch.tensor(0.0).to(outputs.device)

#         # 总损失为正样本损失和负样本损失的加权和
#         loss = loss_positive + self.alpha * loss_negative
#         return loss


# 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self, positive_mean, alpha=0.00001):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.positive_mean = positive_mean
        self.alpha = alpha  # 权重因子

    def forward(self, outputs, targets, labels):
        # 标签为1的样本使用MSE损失
        loss_positive = self.mse(outputs[labels == 1], targets[labels == 1])

        # 标签为0的样本使用与正样本均值的距离作为损失
        if (labels == 0).sum() > 0:
            loss_negative = torch.mean(torch.norm(outputs[labels == 0] - self.positive_mean, dim=1))
        else:
            loss_negative = torch.tensor(0.0).to(outputs.device)

        # 总损失为正样本损失和负样本损失的加权和
        loss = loss_positive + self.alpha * loss_negative
        return loss

# 训练函数
def train_model(data, labels, epochs=50, batch_size=64, learning_rate=1e-3, alpha=1.0):
    dataset = VectorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP()
    model = model.to(data.device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 计算正样本的均值，用于负样本的损失计算
    positive_data = data[labels == 1]
    if len(positive_data) == 0:
        raise ValueError("没有正样本（标签为1）的数据。")
    positive_mean = positive_data.mean(dim=0)
    positive_mean = positive_mean.to(data.device)

    criterion = CustomLoss(positive_mean, alpha=alpha)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(data.device)
            batch_labels = batch_labels.to(data.device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_data.size(0)

        epoch_loss /= len(dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

    return model

# 统计量打印函数
def print_statistics(tensor, name):
    mean = torch.mean(tensor, dim=0).cpu().numpy()
    std = torch.std(tensor, dim=0).cpu().numpy()
    print(f'{name} - Mean: {mean[:5]}... , Std: {std[:5]}...')

# 可视化函数：PCA
def visualize_distributions_PCA(target_positive, origin_neg, generated_neg, generated_pos):
    """
    使用PCA将高维数据降维到2维，并绘制分布图。
    """
    pca = PCA(n_components=2)
    # 结合所有数据进行PCA
    combined = np.vstack((target_positive, origin_neg, generated_neg, generated_pos))
    pca.fit(combined)

    target_positive_pca = pca.transform(target_positive)
    origin_neg_pca = pca.transform(origin_neg)
    generated_neg_pca = pca.transform(generated_neg)
    generated_pos_pca = pca.transform(generated_pos)

    plt.figure(figsize=(12, 6))

    # 绘制正样本
    plt.scatter(target_positive_pca[:, 0], target_positive_pca[:, 1], 
                c='blue', label='Target Positive', alpha=0.5, s=10)

    # 绘制原始负样本
    plt.scatter(origin_neg_pca[:, 0], origin_neg_pca[:, 1],
                c='yellow', label='Original Negative', alpha=0.5, s=10)

    # 绘制生成的负样本
    plt.scatter(generated_neg_pca[:, 0], generated_neg_pca[:, 1], 
                c='red', label='Generated Negative', alpha=0.5, s=10)

    # 绘制生成的正样本
    plt.scatter(generated_pos_pca[:, 0], generated_pos_pca[:, 1], 
                c='green', label='Generated Positive', alpha=0.5, s=10)

    plt.legend()
    plt.title('PCA Projection of Samples')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
    plt.savefig('pca.png')
    
from sae import JumpReLUSAE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

def train_v6(data_np, epochs=50, batch_size=32, learning_rate=0.001, n_sae=2048, val_ratio=0.2, device='cuda'):
    """
    训练一个SAE模型，目标是重建每个输入向量，并在训练过程中划分验证集以评估模型性能。
    
    Parameters:
    - data_np (numpy.ndarray): 输入数据 (训练数据)
    - epochs (int): 训练的轮数
    - batch_size (int): 每个批次的数据量
    - learning_rate (float): 学习率
    - n_sae (int): SAE的内部编码维度
    - val_ratio (float): 验证集所占比例（0到1之间）
    - device (str): 训练使用的设备（'cpu' 或 'cuda'）
    
    Returns:
    - sae_model (nn.Module): 训练好的SAE模型
    - history (dict): 包含训练和验证损失的历史记录
    """
    # 转换为 PyTorch tensor
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    
    # 创建数据集
    dataset = TensorDataset(data_tensor, data_tensor)  # 输入和目标都是data_tensor
    
    # 计算验证集大小
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    # 划分为训练集和验证集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化SAE模型
    sae_model = JumpReLUSAE(n_input=data_tensor.shape[1], n_sae=n_sae).to(device)
    
    # 定义损失函数（均方误差，MSE）
    criterion = nn.MSELoss()
    
    # 定义优化器（Adam）
    optimizer = optim.Adam(sae_model.parameters(), lr=learning_rate)
    
    # 历史记录
    history = {'train_loss': [], 'val_loss': []}
    
    # 训练过程
    for epoch in range(1, epochs + 1):
        sae_model.train()
        running_loss = 0.0
        
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            recon_data = sae_model(batch_data)
            
            # 计算损失
            loss = criterion(recon_data, batch_data)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_data.size(0)
        
        # 计算平均训练损失
        epoch_train_loss = running_loss / train_size
        history['train_loss'].append(epoch_train_loss)
        
        # 验证过程
        sae_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data, _ in val_loader:
                val_data = val_data.to(device)
                recon_val = sae_model(val_data)
                loss_val = criterion(recon_val, val_data)
                val_loss += loss_val.item() * val_data.size(0)
        
        # 计算平均验证损失
        epoch_val_loss = val_loss / val_size
        history['val_loss'].append(epoch_val_loss)
        
        # 打印损失信息
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
    
    return sae_model


# 可视化函数：t-SNE
def visualize_distributions_tsne(target_positive, origin_neg, generated_neg, generated_pos):
    """
    使用t-SNE将高维数据降维到2维，并绘制分布图。
    """
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    # 结合所有数据进行t-SNE
    combined = np.vstack((target_positive, origin_neg, generated_neg, generated_pos))
    tsne_results = tsne.fit_transform(combined)

    # 划分降维后的数据
    n_positive = target_positive.shape[0]
    n_origin_neg = origin_neg.shape[0]
    n_generated_neg = generated_neg.shape[0]

    target_positive_tsne = tsne_results[:n_positive]
    origin_neg_tsne = tsne_results[n_positive:n_positive + n_origin_neg]
    generated_neg_tsne = tsne_results[n_positive + n_origin_neg:n_positive + n_origin_neg + n_generated_neg]
    generated_pos_tsne = tsne_results[n_positive + n_origin_neg + n_generated_neg:]

    plt.figure(figsize=(12, 6))

    # 绘制正样本
    plt.scatter(target_positive_tsne[:, 0], target_positive_tsne[:, 1], 
                c='blue', label='Target Positive', alpha=0.6, s=10)
    # 绘制原始负样本
    plt.scatter(origin_neg_tsne[:, 0], origin_neg_tsne[:, 1],
                c='orange', label='Original Negative', alpha=0.6, s=10)

    # 绘制生成的负样本
    plt.scatter(generated_neg_tsne[:, 0], generated_neg_tsne[:, 1], 
                c='red', label='Generated Negative', alpha=0.2, s=10)

    # 绘制生成的正样本
    plt.scatter(generated_pos_tsne[:, 0], generated_pos_tsne[:, 1], 
                c='green', label='Generated Positive', alpha=0.2, s=10)

    plt.legend()
    plt.title('t-SNE Projection of Samples')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
    plt.savefig('t-SNE.png')

# def train_v5(data_np, labels_np,layer,head,epochs=50):
    # 设置随机种子
    # torch.manual_seed(42)
    # np.random.seed(42)

    # 数据生成参数
    # num_samples = 4560
    # input_dim = 128
    # num_positive = num_samples // 2  # 假设正负样本各占一半
    # num_negative = num_samples - num_positive

    # # 正样本生成：均值为1，协方差为单位矩阵
    # mu_pos = np.ones(input_dim)
    # sigma_pos = np.eye(input_dim)
    # positive_data = np.random.multivariate_normal(mean=mu_pos, cov=sigma_pos, size=num_positive)

    # # 负样本生成：均值为-1，协方差为单位矩阵
    # mu_neg = -1 * np.ones(input_dim)
    # sigma_neg = np.eye(input_dim)
    # negative_data = np.random.multivariate_normal(mean=mu_neg, cov=sigma_neg, size=num_negative)

    # 合并数据
    # data_np = np.vstack((positive_data, negative_data))
    # labels_np = np.hstack((np.ones(num_positive), np.zeros(num_negative)))

    # # 转换为Tensor
    # data_tensor = torch.tensor(data_np, dtype=torch.float32)
    # labels_tensor = torch.tensor(labels_np, dtype=torch.float32)
    
    # # 将数据移动到GPU（如果可用）
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data_tensor = data_tensor.to(device)
    # labels_tensor = labels_tensor.to(device)
    

    # # 训练模型
    # model = train_model(data_tensor, labels_tensor, epochs=epochs, batch_size=64, learning_rate=1e-3, alpha=10.0)
    # model.to(device)

    # # 保存模型
    # torch.save(model.state_dict(), f'/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/src/saes/version1/sea_{layer}_{head}.pth')

    # # 生成重建样本
    # model.eval()
    # with torch.no_grad():
    #     all_outputs = model(data_tensor)

    # # 分离生成的正负样本
    # generated_pos = all_outputs[labels_tensor == 1]
    # generated_neg = all_outputs[labels_tensor == 0]

    # # 获取原始正负样本
    # all_positive = data_tensor[labels_tensor == 1]
    # all_negative = data_tensor[labels_tensor == 0]

    # # 打印统计量
    # print("\nStatistics of Generated Negative Samples:")
    # print_statistics(generated_neg, "Generated Negative")

    # print("\nStatistics of Generated Positive Samples:")
    # print_statistics(generated_pos, "Generated Positive")

    # print("\nStatistics of Target Positive Samples:")
    # print_statistics(all_positive, "Target Positive")

    # # 转换为numpy数组用于可视化
    # all_positive_np = all_positive.cpu().numpy()
    # all_negative_np = all_negative.cpu().numpy()
    # generated_neg_np = generated_neg.cpu().numpy()
    # generated_pos_np = generated_pos.cpu().numpy()

    # # 可视化样本分布（使用PCA）
    # visualize_distributions_PCA(all_positive_np, all_negative_np, generated_neg_np, generated_pos_np)

    # # 可视化样本分布（使用t-SNE）
    # visualize_distributions_tsne(all_positive_np, all_negative_np, generated_neg_np, generated_pos_np)

# if __name__ == "__main__":
#     main()
