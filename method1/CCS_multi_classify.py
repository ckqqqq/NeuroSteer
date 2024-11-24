'''
仿照CCS方法写的标签八分类方法，但是目前还有一定问题，探针趋向于只预测其中某一个标签
但是基本思想应该没差，可以基于这段代码进一步修改来实现方法一
'''


from tqdm import tqdm  # 用于显示进度条
import copy  # 用于对象深拷贝
import json
import numpy as np  # 用于数值计算
import torch  # 用于深度学习计算
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # PyTorch功能性模块
from datasets import load_dataset  # 用于加载标准化数据集
from transformers import AutoTokenizer, AutoModelForCausalLM  # 用于加载预训练模型和分词器
from sklearn.linear_model import LogisticRegression  # 逻辑回归分类器
from sklearn.metrics import classification_report  # 用于计算分类报告
from sklearn.preprocessing import LabelEncoder  # 用于将多分类标签编码为数字

print("init!")

# 数据加载与模型初始化
cache_dir = "/home/ckqsudo/code2024/SZK_ACL2024_test/Uncertainty/method1/cache"  # 模型缓存目录
# data = load_dataset("custom_dataset", cache_dir=cache_dir)["test"]  # 加载自定义的多分类情感数据集
with open("cache.json", "r", encoding="utf-8") as f:
    data = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(
    "model_path", cache_dir=cache_dir, trust_remote_code=True
)  # 加载分词器
model = AutoModelForCausalLM.from_pretrained(
    "model_path", cache_dir=cache_dir, trust_remote_code=True
).cuda()  # 加载模型并移动到GPU

print("model_loaded")

# 定义分类标签
categories = [
    "Question", "Others", "Providing Suggestions", "Affirmation and Reassurance",
    "Self-disclosure", "Reflection of feelings", "Information", "Restatement or Paraphrasing"
]

# 初始化标签编码器
label_encoder = LabelEncoder()
label_encoder.fit(categories)  # 将分类标签转为数字编码

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

# 格式化文本
def format_text(text, label):
    """
    将文本和多分类标签格式化为零样本提示语。
    :param text: 输入文本
    :param label: 文本标签
    :return: 格式化后的提示语
    """
    return f"You are a professional counselor, and your task is to help me analyze the psychotherapy strategies used by the supporter in a conversation by selecting the letter that represents the strategies used by the supporter in the conversation.\nThe following is the content of the conversation:\n{text}\n\nThe correct label is:\n{text}"

# 批量获取隐藏状态
def get_hidden_states_many_examples(model, tokenizer, data, n=100):
    """
    从数据集中随机抽取n个样本，提取对应分类下的隐藏状态。
    :param model: 预训练模型
    :param tokenizer: 分词器
    :param data: 数据集
    :param n: 样本数量
    :return: 样本隐藏状态及其标签
    """
    model.eval()  # 切换为评估模式
    all_hs, labels = [], []  # 初始化结果存储
    for _ in tqdm(range(n)):  # 遍历n个样本
        idx = np.random.randint(len(data))  # 随机选择样本
        # text, label = data[idx]["content"], data[idx]["label"]
        text, label = data[idx]["chat_history"], data[idx]["predict_strategy_label"]
        hs = get_decoder_hidden_states(model, tokenizer, format_text(text, label))  # 获取隐藏状态
        all_hs.append(hs)
        labels.append(label)  # 存储真实标签
    return np.stack(all_hs), np.array(labels)  # 转为NumPy数组后返回

# 提取隐藏状态和标签
n_samples = 1000  # 样本数量
hidden_states, labels = get_hidden_states_many_examples(model, tokenizer, data, n=n_samples)
encoded_labels = label_encoder.transform(labels)  # 将标签转为数值编码

print("hidden_states_extracted")

# 数据集划分
n = len(encoded_labels)
hs_train, hs_test = hidden_states[:n // 2], hidden_states[n // 2:]  # 特征划分
y_train, y_test = encoded_labels[:n // 2], encoded_labels[n // 2:]  # 标签划分

print("dataset_split")

# 定义多层感知机探针（修改为多分类）
class MLPProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 100)  # 输入层到隐藏层
        self.linear2 = nn.Linear(100, num_classes)  # 隐藏层到输出层

    def forward(self, x):
        h = F.relu(self.linear1(x))  # ReLU激活
        return F.softmax(self.linear2(h), dim=-1)  # 输出层，使用softmax激活以得到类别概率

# CCS探针类
class CCS:
    def __init__(self, x, y, num_classes, nepochs=1000, lr=1e-3, batch_size=32, device="cuda"):
        self.x = torch.tensor(x, dtype=torch.float32, device=device)  # 特征
        self.y = torch.tensor(y, dtype=torch.long, device=device)  # 标签
        self.num_classes = num_classes
        self.nepochs = nepochs  # 训练轮数
        self.lr = lr  # 学习率
        self.batch_size = batch_size  # 批量大小
        self.device = device  # 设备
        self.model = MLPProbe(x.shape[1], num_classes).to(device)  # 初始化MLP探针
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  # 优化器

    def train(self):
        """
        训练多层感知机探针。
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()  # 损失函数（交叉熵损失，适用于多分类任务）
        for epoch in tqdm(range(self.nepochs)):
            permutation = torch.randperm(self.x.size(0))  # 打乱样本顺序
            for i in range(0, self.x.size(0), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = self.x[indices], self.y[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)  # 获取模型输出
                loss = criterion(outputs, batch_y)  # 计算交叉熵损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

    def evaluate(self, x_test, y_test, data):
        """
        评估多层感知机探针性能，并返回每个样本的内容、正确标签、预测标签和置信度。
        """
        self.model.eval()
        x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device)
        y_test = torch.tensor(y_test, dtype=torch.long, device=self.device)
        results = []  # 存储结果

        with torch.no_grad():
            outputs = self.model(x_test)  # 获取输出概率
            probs, preds = torch.max(outputs, dim=1)  # 获取最大概率和预测标签

            for i in range(len(preds)):
                sample_text = data[i]["chat_history"]  # 获取样本内容
                true_label = label_encoder.inverse_transform([y_test[i].item()])[0]  # 获取真实标签
                pred_label = label_encoder.inverse_transform([preds[i].item()])[0]  # 获取预测标签
                confidence = probs[i].item()  # 获取预测的置信度

                # 保存样本内容、真实标签、预测标签和置信度
                results.append({
                    "text": sample_text,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": confidence
                })

        return results

# CCS训练与评估
ccs = CCS(hs_train, y_train, num_classes=len(categories))
ccs.train()
evaluation_results = ccs.evaluate(hs_test, y_test, data)  # 获取评估结果

# 输出每个样本的内容、正确标签、预测标签、置信度
for result in evaluation_results:
    print(f"Text: {result['text']}\nTrue Label: {result['true_label']}\nPredicted Label: {result['pred_label']}\nConfidence: {result['confidence']:.4f}\n")

# 输出MLP探针的分类报告
y_pred_ccs = [result['pred_label'] for result in evaluation_results]
y_true_ccs = [result['true_label'] for result in evaluation_results]
print("MLPProbe Classification report:")
print(classification_report(y_true_ccs, y_pred_ccs, target_names=label_encoder.classes_, zero_division=0))
