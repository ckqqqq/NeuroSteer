'''
CCS的实现方法重构版
可以在其官方数据集上验证CCS方法，但是感觉准确率不高，可能是数据集设置的问题
'''

from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression

cache_dir = "/home/ckqsudo/code2024/SZK_ACL2024_test/Uncertainty/method1/cache" # 模型缓存目录

# 获取解码器隐藏状态
def get_decoder_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    给定解码器模型和输入文本，获取指定层的隐藏状态。
    :param model: 预训练的解码器模型
    :param tokenizer: 分词器
    :param input_text: 输入文本
    :param layer: 指定的层，默认最后一层
    :return: 隐藏状态的 NumPy 数组 (hidden_dim,)
    """
    input_ids = tokenizer(input_text + tokenizer.eos_token, return_tensors="pt").input_ids.to(model.device)  # 分词并添加EOS符号
    with torch.no_grad():  # 禁用梯度计算
        output = model(input_ids, output_hidden_states=True)  # 模型前向传播，获取隐藏状态
    return output.hidden_states[layer][0, -1].detach().cpu().numpy()  # 返回指定层最后一个词的隐藏状态

# 格式化IMDB文本
def format_imdb(text, label):
    """
    将文本和标签（0为负面，1为正面）格式化为零样本提示语。
    :param text: 输入文本
    :param label: 文本标签
    :return: 格式化后的提示语
    """
    sentiment = "negative" if label == 0 else "positive"  # 根据标签选择情感
    return f"The following movie review expresses a {sentiment} sentiment:\n{text}"

# 批量获取隐藏状态
def get_hidden_states_many_examples(model, tokenizer, data, n=100000):
    """
    从数据集中随机抽取n个样本，提取正负面格式下的隐藏状态。
    :param model: 预训练模型
    :param tokenizer: 分词器
    :param data: 数据集
    :param n: 样本数量
    :return: 负面隐藏状态、正面隐藏状态及真实标签
    """
    model.eval()  # 切换为评估模式
    neg_hs, pos_hs, labels = [], [], []  # 初始化结果存储
    for _ in tqdm(range(n)):  # 遍历n个样本
        while True:  # 确保样本文本长度合适
            idx = np.random.randint(len(data))  # 随机选择样本
            text, label = data[idx]["content"], data[idx]["label"]
            if len(tokenizer(text)) < 400:  # 控制文本长度
                break
        # 获取负面和正面隐藏状态
        neg_hs.append(get_decoder_hidden_states(model, tokenizer, format_imdb(text, 0)))
        pos_hs.append(get_decoder_hidden_states(model, tokenizer, format_imdb(text, 1)))
        labels.append(label)  # 存储真实标签
    return np.stack(neg_hs), np.stack(pos_hs), np.array(labels)  # 转为NumPy数组后返回

# 加载数据、模型、分词器
data = load_dataset("amazon_polarity",cache_dir=cache_dir)["test"]  # 加载情感分类数据集的测试集
tokenizer = AutoTokenizer.from_pretrained("model_path", cache_dir=cache_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("model_path", cache_dir=cache_dir, trust_remote_code=True).cuda()

# 提取隐藏状态和标签
neg_hs, pos_hs, y = get_hidden_states_many_examples(model, tokenizer, data)

print("hidden_states_extracted")

# 数据集划分
n = len(y)
neg_train, neg_test = neg_hs[:n // 2], neg_hs[n // 2:]  # 负面样本
pos_train, pos_test = pos_hs[:n // 2], pos_hs[n // 2:]  # 正面样本
y_train, y_test = y[:n // 2], y[n // 2:]  # 标签

print("dataset_split")

# 特征拼接
x_train = np.hstack([neg_train, pos_train, neg_train - pos_train])
x_test = np.hstack([neg_test, pos_test, neg_test - pos_test])

# 特征标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 使用随机森林模型
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
print(f"Random Forest accuracy: {rf.score(x_test, y_test)}")


# 定义多层感知机探针
class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)  # 输入层到隐藏层
        self.linear2 = nn.Linear(100, 1)  # 隐藏层到输出层

    def forward(self, x):
        h = F.relu(self.linear1(x))  # ReLU激活
        return torch.sigmoid(self.linear2(h))  # 输出层使用Sigmoid激活

# CCS探针类
class CCS:
    def __init__(self, x0, x1, nepochs=10000, ntries=10, lr=1e-3, batch_size=10,
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        self.var_normalize = var_normalize  # 是否方差归一化
        self.x0 = self.normalize(x0)  # 标准化负面样本
        self.x1 = self.normalize(x1)  # 标准化正面样本
        self.d = self.x0.shape[-1]  # 特征维度
        self.nepochs, self.ntries = nepochs, ntries  # 训练轮数与尝试次数
        self.lr, self.batch_size = lr, batch_size  # 学习率与批量大小
        self.verbose, self.device = verbose, device  # 调试模式与设备
        self.weight_decay = weight_decay  # 权重衰减
        self.linear = linear  # 是否使用线性探针
        self.initialize_probe()  # 初始化探针模型

    def initialize_probe(self):
        """
        初始化探针模型：线性或MLP探针。
        """
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())  # 线性探针
        else:
            self.probe = MLPProbe(self.d)  # MLP探针
        self.probe.to(self.device)  # 将模型移动到设备上

    def normalize(self, x):
        """
        数据标准化：减去均值，若启用方差归一化则再除以标准差。
        """
        x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            x /= (x.std(axis=0, keepdims=True) + 1e-6)
        return x

    def train(self):
        """
        使用对比损失训练探针模型。
        """
        x0, x1 = map(lambda t: torch.tensor(t, dtype=torch.float32, device=self.device), (self.x0, self.x1))
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # AdamW优化器
        for epoch in range(self.nepochs):  # 迭代多个epoch
            permutation = torch.randperm(len(x0))  # 打乱样本顺序
            for i in range(0, len(x0), self.batch_size):  # 批量训练
                x0_batch = x0[permutation[i:i+self.batch_size]]
                x1_batch = x1[permutation[i:i+self.batch_size]]
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)  # 预测概率
                loss = self.get_loss(p0, p1)  # 计算CCS损失
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def get_loss(self, p0, p1):
        """
        计算CCS损失：信息性损失 + 一致性损失。
        """
        info_loss = (torch.min(p0, p1) ** 2).mean()  # 信息性损失
        cons_loss = ((p0 - (1 - p1)) ** 2).mean()  # 一致性损失
        return info_loss + cons_loss

    def evaluate(self, x0_test, x1_test, y_test):
        """
        评估模型准确率。
        """
        x0, x1 = map(self.normalize, (x0_test, x1_test))  # 测试数据标准化
        x0, x1 = map(lambda t: torch.tensor(t, dtype=torch.float32, device=self.device), (x0, x1))
        with torch.no_grad():
            p0, p1 = self.probe(x0), self.probe(x1)  # 获取概率
        avg_confidence = 0.5 * (p0 + (1 - p1))  # 平均置信度
        predictions = (avg_confidence.cpu().numpy() < 0.5).astype(int).flatten()
        acc = (predictions == y_test).mean()  # 计算准确率
        return max(acc, 1 - acc)  # 返回最大值以处理标签翻转的情况

# CCS训练与评估
ccs = CCS(neg_train, pos_train)
ccs.train()
print(f"CCS accuracy: {ccs.evaluate(neg_test, pos_test, y_test)}")
