'''
仿照CCS方法写的标签八分类方法，但是目前还有一定问题，探针趋向于只预测其中某一个标签
但是基本思想应该没差，可以基于这段代码进一步修改来实现方法一
'''
from tqdm import tqdm
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report  # 用于计算分类报告
from sklearn.preprocessing import LabelEncoder  # 用于将多分类标签编码为数字

# 数据加载与模型初始化
cache_dir = "/home/szk/szk_2024/1124"  # 模型缓存目录
# data = load_dataset("custom_dataset", cache_dir=cache_dir)["test"]  # 加载自定义的多分类情感数据集
with open("/home/szk/szk_2024/1124/cache.json", "r", encoding="utf-8") as f:
    data = json.load(f)


tokenizer = AutoTokenizer.from_pretrained(
    "/home/szk/szk_2024/ChatGLM3/chatglm3-6b", cache_dir=cache_dir, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "/home/szk/szk_2024/ChatGLM3/chatglm3-6b", cache_dir=cache_dir, trust_remote_code=True
).cuda()


# 定义分类标签
categories = [
    "Question", "Others", "Providing Suggestions", "Affirmation and Reassurance",
    "Self-disclosure", "Reflection of feelings", "Information", "Restatement or Paraphrasing"
]


# 获取解码器隐藏状态
def get_decoder_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    给定解码器模型和输入文本，获取指定层的隐藏状态。
    """
    input_ids = tokenizer(input_text + tokenizer.eos_token, return_tensors="pt").input_ids.to(model.device)  # 分词并添加EOS符号
    with torch.no_grad():  # 禁用梯度计算
        output = model(input_ids, output_hidden_states=True)  # 模型前向传播，获取隐藏状态
    return output.hidden_states[layer][0, -1].detach().cpu().numpy()  # 返回指定层最后一个词的隐藏状态

# 格式化文本
def format_text(text, label):
    """
    将文本和多分类标签格式化为零样本提示语。
    """
    return f"In the following conversation, the supporter uses the strategy of {label} in the last sentence response to the seeker:\n\n{text}"

# 批量获取隐藏状态
def get_hidden_states_many_examples(model, tokenizer, data, n=1000):
    """
    从数据集中随机抽取n个样本，提取对应分类下的隐藏状态。
    """
    model.eval()  # 切换为评估模式
    all_hs, labels = [], []  # 初始化结果存储
    for _ in tqdm(range(n)):  # 遍历n个样本
        idx = np.random.randint(len(data))  # 随机选择样本
        text, label = data[idx]["chat_history"]+data[idx]["predict_utterance"], data[idx]["predict_strategy_label"]
        hs = get_decoder_hidden_states(model, tokenizer, format_text(text, label))  # 获取隐藏状态
        all_hs.append(hs)
        labels.append(label)  # 存储真实标签
    return np.stack(all_hs), np.array(labels)  # 转为NumPy数组后返回

# 提取隐藏状态和标签
n_samples = 1000  # 样本数量
print(f"Extracting hidden states from {n_samples} samples...")
hidden_states, labels = get_hidden_states_many_examples(model, tokenizer, data, n=n_samples)


# 标签编码
label_encoder = LabelEncoder()
label_encoder.fit(categories)  # 将分类标签转为数字编码
encoded_labels = label_encoder.transform(labels)  # 将标签转为数值编码


# 数据集划分
n = int(len(encoded_labels) * 0.8)
hs_train, hs_test = hidden_states[:n], hidden_states[n:]  # 特征划分
y_train, y_test = encoded_labels[:n], encoded_labels[n:]  # 标签划分


class MLPProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512)  # 输入层到第一个隐藏层，增加节点数
        self.linear2 = nn.Linear(512, 256)       # 第二个隐藏层
        self.linear3 = nn.Linear(256, 128)       # 第三个隐藏层
        self.linear4 = nn.Linear(128, num_classes)  # 输出层

    def forward(self, x):
        h = F.relu(self.linear1(x))  # ReLU激活
        h = F.relu(self.linear2(h))  # 第二层ReLU激活
        h = F.relu(self.linear3(h))  # 第三层ReLU激活
        return self.linear4(h)  # 最后一层输出logits



import torch
import torch.nn.functional as F

def custom_loss(logits, labels, lambda_penalty=0.5):
    """
    自定义损失函数，结合了交叉熵损失、对错误预测的惩罚以及概率和惩罚。

    Args:
        logits (torch.Tensor): 模型输出的logits（未经过softmax），形状为 (batch_size, num_classes)
        labels (torch.Tensor): 真实标签，形状为 (batch_size,)
        lambda_penalty (float): 错误预测惩罚的权重，控制错误预测样本的惩罚强度

    Returns:
        torch.Tensor: 计算出的损失值
    """
    
    # 1. 交叉熵损失：确保正确类别的概率尽可能高
    ce_loss = F.cross_entropy(logits, labels)
    probs = F.softmax(logits, dim=-1)  # 将logits转化为概率

    # 2. 错误预测惩罚：对预测错误的样本加大损失
    incorrect_mask = (torch.argmax(probs, dim=-1) != labels)  # 错误预测的样本
    incorrect_penalty = torch.sum(incorrect_mask.float())  # 错误样本的数量
    error_loss = incorrect_penalty * lambda_penalty  # 错误样本的惩罚

    # 3. 概率和惩罚：确保所有类别的预测概率之和接近1
    probs_sum = torch.sum(probs, dim=-1)  # 计算每个样本所有类别概率的和
    probability_sum_loss = torch.mean(torch.abs(probs_sum - 1.0))  # 计算和与1的差距并取平均

    # 4. 最终损失：结合交叉熵损失、正则化项、错误预测惩罚和概率和惩罚
    total_loss = ce_loss  + error_loss + probability_sum_loss

    return total_loss


class CCS:
    def __init__(self, x, y, num_classes, nepochs=500, lr=1e-5, batch_size=100, device="cuda"):
        # 将 NumPy 数组转换为 PyTorch Tensor，并确保数据类型正确
        self.x = torch.tensor(x, dtype=torch.float32, device=device)  # 特征转换为 FloatTensor
        self.y = torch.tensor(y, dtype=torch.long, device=device)  # 标签转换为 LongTensor
        self.num_classes = num_classes
        self.nepochs = nepochs  # 训练轮数
        self.lr = lr  # 学习率
        self.batch_size = batch_size  # 批量大小
        self.device = device  # 设备
        self.model = MLPProbe(x.shape[1], num_classes).to(device)  # 初始化MLP探针
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)  # 添加权重衰减项

    def train(self):
        """
        训练多层感知机探针。
        """
        self.model.train()
        # 使用自定义损失函数
        for epoch in tqdm(range(self.nepochs)):
            permutation = torch.randperm(self.x.size(0))  # 打乱样本顺序
            for i in range(0, self.x.size(0), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = self.x[indices], self.y[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)  # 获取模型输出

                # 使用自定义损失函数计算损失
                loss = custom_loss(outputs, batch_y)  # 计算自定义损失

                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

                if i % 100 == 0:  # 每100个batch输出一次日志
                    print(f"Epoch [{epoch+1}/{self.nepochs}], Step [{i}/{self.x.size(0)}], Loss: {loss.item():.4f}")

    def evaluate(self, x_test, y_test):
        """
        评估多层感知机探针性能，并返回每个样本的内容、正确标签、预测标签和置信度。
        """
        self.model.eval()
        x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device)  # 特征转换为 FloatTensor
        y_test = torch.tensor(y_test, dtype=torch.long, device=self.device)  # 标签转换为 LongTensor
        results = []  # 存储结果

        with torch.no_grad():
            outputs = self.model(x_test)  # 获取输出概率
            probs, preds = torch.max(outputs, dim=1)  # 获取最大概率和预测标签

            for i in range(len(preds)):
                true_label = label_encoder.inverse_transform([y_test[i].item()])[0]  # 获取真实标签
                pred_label = label_encoder.inverse_transform([preds[i].item()])[0]  # 获取预测标签
                confidence = probs[i].item()  # 获取预测的置信度

                # 保存样本内容、真实标签、预测标签和置信度
                results.append({
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": confidence
                })

        return results


# CCS训练与评估
print("Training CCS model...")
ccs = CCS(hs_train, y_train, num_classes=len(categories))
ccs.train()

print("Evaluating the model...")
evaluation_results = ccs.evaluate(hs_test, y_test)  # 获取评估结果

# 输出每个样本的内容、正确标签、预测标签、置信度
for result in evaluation_results[:20]:  # 只打印前10个结果进行调试
    print(f"True Label: {result['true_label']}\nPredicted Label: {result['pred_label']}\nConfidence: {result['confidence']:.4f}\n")

# 输出MLP探针的分类报告
y_pred_ccs = [result['pred_label'] for result in evaluation_results]
y_true_ccs = [result['true_label'] for result in evaluation_results]
print("MLPProbe Classification report:")
print(classification_report(y_true_ccs, y_pred_ccs, target_names=label_encoder.classes_, zero_division=0))

