import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 设置自动求导异常检测（可选）
torch.autograd.set_detect_anomaly(True)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 正确实现RBF核的MMD函数
def mmd(x, y, kernel='rbf', gamma=1.0):
    if kernel == 'rbf':
        # 计算x和x的距离
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
        dist_xx = x_norm + x_norm.t() - 2 * torch.mm(x, x.t())
        K_xx = torch.exp(-gamma * dist_xx)
        
        # 计算y和y的距离
        y_norm = (y ** 2).sum(dim=1).unsqueeze(1)
        dist_yy = y_norm + y_norm.t() - 2 * torch.mm(y, y.t())
        K_yy = torch.exp(-gamma * dist_yy)
        
        # 计算x和y的距离
        dist_xy = x_norm + y_norm.t() - 2 * torch.mm(x, y.t())
        K_xy = torch.exp(-gamma * dist_xy)
        
        # 计算MMD
        m = x.size(0)
        n = y.size(0)
        
        # 去除对角线元素
        sum_K_xx = (K_xx.sum() - torch.diag(K_xx).sum()) / (m * (m - 1))
        sum_K_yy = (K_yy.sum() - torch.diag(K_yy).sum()) / (n * (n - 1))
        sum_K_xy = K_xy.mean()
        
        mmd_value = sum_K_xx + sum_K_yy - 2 * sum_K_xy
        return mmd_value
    else:
        raise NotImplementedError("仅支持RBF核")

# 数据准备
def prepare_data():
    # 生成1000个128维的正负样本，均值为0，标准差为1
   # 设置不同的均值
    negative_mean = 0.0
    positive_mean = 2.0  # 正样本均值偏移为2.0，使其与负样本分布明显不同
    
    # 生成1000个128维的负样本和正样本，均值不同，标准差相同
    negative_samples = np.random.randn(1000, 128) + negative_mean
    positive_samples = np.random.randn(1000, 128) + positive_mean
    
    # 转换为PyTorch张量
    negative_samples = torch.from_numpy(negative_samples).float()
    positive_samples = torch.from_numpy(positive_samples).float()
    
    # 划分训练集和验证集
    train_size = 800
    val_size = 200
    train_negative = negative_samples[:train_size]
    train_positive = positive_samples[:train_size]
    val_negative = negative_samples[train_size:train_size+val_size]
    val_positive = positive_samples[train_size:train_size+val_size]
    
    return (train_negative, train_positive), (val_negative, val_positive)

# 创建DataLoader
def create_dataloaders(train_data, val_data, batch_size=32):
    train_negative, train_positive = train_data
    val_negative, val_positive = val_data
    
    # 创建训练集的DataLoader
    train_dataset_neg = TensorDataset(train_negative)
    train_loader_neg = DataLoader(train_dataset_neg, batch_size=batch_size, shuffle=True)
    
    train_dataset_pos = TensorDataset(train_positive)
    train_loader_pos = DataLoader(train_dataset_pos, batch_size=batch_size, shuffle=True)
    
    # 创建验证集的DataLoader
    val_dataset_neg = TensorDataset(val_negative)
    val_loader_neg = DataLoader(val_dataset_neg, batch_size=batch_size, shuffle=False)
    
    val_dataset_pos = TensorDataset(val_positive)
    val_loader_pos = DataLoader(val_dataset_pos, batch_size=batch_size, shuffle=False)
    
    return (train_loader_neg, train_loader_pos), (val_loader_neg, val_loader_pos)

# 定义综合MMD损失函数
def combined_mmd_loss(generated_neg, generated_pos, target_pos):
    # MMD损失：负样本转换后与正样本分布的差异
    loss_neg = mmd(generated_neg, target_pos, kernel='rbf', gamma=1.0)
    # MMD损失：正样本转换后与自身分布的差异
    loss_pos = mmd(generated_pos, target_pos, kernel='rbf', gamma=1.0)
    # 综合损失，可以根据需要调整权重
    combined_loss = loss_neg + loss_pos
    return combined_loss

# 定义综合L1损失函数
def combined_l1_loss(generated_neg, generated_pos, target_pos, l1_criterion):
    # L1损失：负样本转换后与正样本的差异
    loss_neg = l1_criterion(generated_neg, target_pos)
    # L1损失：正样本转换后与正样本的差异
    loss_pos = l1_criterion(generated_pos, target_pos)
    # 综合损失，可以根据需要调整权重
    combined_loss = loss_neg + loss_pos
    return combined_loss

def main():
    # 准备数据
    train_data, val_data = prepare_data()
    (train_loader_neg, train_loader_pos), (val_loader_neg, val_loader_pos) = create_dataloaders(train_data, val_data, batch_size=32)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型参数
    input_dim = 128
    hidden_dim = 256
    output_dim = 128
    
    # 初始化模型并移动到设备
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 定义L1损失函数
    l1_criterion = nn.L1Loss()
    
    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # 确保训练集的负样本和正样本DataLoader长度一致
        for (batch_neg, batch_pos) in zip(train_loader_neg, train_loader_pos):
            batch_neg = batch_neg[0].to(device)
            batch_pos = batch_pos[0].to(device)
            
            optimizer.zero_grad()
            # 负样本转换
            generated_neg = model(batch_neg)
            # 正样本保持
            generated_pos = model(batch_pos)
            # 计算综合损失
            loss = combined_l1_loss(generated_neg, generated_pos, batch_pos,l1_criterion=l1_criterion)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_neg.size(0)
        train_loss /= len(train_loader_neg.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_neg, batch_pos) in zip(val_loader_neg, val_loader_pos):
                batch_neg = batch_neg[0].to(device)
                batch_pos = batch_pos[0].to(device)
                
                # 负样本转换
                generated_neg = model(batch_neg)
                # 正样本保持
                generated_pos = model(batch_pos)
                # 计算综合损失
                loss = combined_mmd_loss(generated_neg, generated_pos, batch_pos)
                val_loss += loss.item() * batch_neg.size(0)
            val_loss /= len(val_loader_neg.dataset)
        
        # 打印epoch统计
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # 更新学习率
        scheduler.step()
    
    # 计算最终的综合MMD
    model.eval()
    with torch.no_grad():
        all_negative = train_data[0].to(device)
        all_positive = train_data[1].to(device)
        generated_neg = model(all_negative)
        generated_pos = model(all_positive)
        final_mmd_neg = mmd(generated_neg, all_positive, kernel='rbf', gamma=1.0)
        final_mmd_pos = mmd(generated_pos, all_positive, kernel='rbf', gamma=1.0)
        final_mmd = final_mmd_neg + final_mmd_pos
    print(f'Final MMD: {final_mmd.item():.6f}')

if __name__ == "__main__":
    main()
