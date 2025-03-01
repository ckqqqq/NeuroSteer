# NeuroSteer: Regulating LM Behavior via SAE Neurons: 

## Description

NeuroSteer is a plug-and-play framework for dynamic control of language model behaviors (e.g., sentiment, politeness, toxicity) **without domain-specific training**. By leveraging Sparse AutoEncoder (SAE) feature-space interventions, it activates target-related neurons, extracts feature residuals, and modulates hidden states to steer output distributions. Experiments show NeuroSteer achieves SOTA on four benchmarks, balancing generation quality and behavioral control. It enables rapid domain adaptation in seconds using hundreds of examples, surpassing fine-tuning. Layer-wise interventions also reveal insights into concept representation and feature vector combinations in LMs. We release our model, code, demo, and steering vectors for the NLP research community.
### What can NeuroSteer do?

<p align="center">
  <img src=https://ckqqqq-qiker-image-service.oss-cn-beijing.aliyuncs.com/typora-image/demo_main_1.7b.gif/>
</p>

* Quantitatively regulate LLM behaviors in any tasks

### How does NeuroSteer work?

<p align="center">
  <img src=doc/method1.gif />
</p>

* Adjusting LLMs output via activating SAE neurons

### Quick Start
* To DO
### Deploy Our Demo
DEMO Backend:

```bash
uvicorn demo.backend.main_fastapp_backend:app
```



DEMO Frontend:

```bash
streamlit run demo/frontend/main_streamlit_frontend.py
```
### Reproduce NeuroSteer

```bash
git clone
cd NeuroSteer
pip install -r requirements.txt # SOON  
pip install -e .
cd ./src/scripts
chmod 777 demo_prepare_all.sh
./demo_prepare_all.sh

# all reproduction scripts are in src/scripts  change your path for reproduction
```
### Apply NeuroSteer to your task. 

* Prepare your dataset for your task, which includes negative examples and positive examples, eg.:
    * Complex reasoning text and simple reasoning text
    * Happy image and Sad image
    * Confused emojis and Happy emojis
    * ...(It doesn't necessarily have to be a binary opposition, but there needs to be a distinction in same dataset. )
* Modify the bashs in src/scripts/

## 🚀 News
- **2025/02/27**: 📣 [Colab notebooks] Demo is released. Feel free to try!

- **2025/02/28**: 📣 NeuroSteer Code release.[Demo website](https://auffusion.github.io/) and 

- **2025/03/01**: 📣 NeuroSteer website release. All steering vectors are released in [Hugging Face]. 

## NeuroSteer Model Family

| Model Name                 | Model                                                                                                    |
|----------------------------|------------------------------------------------------------------------------------------------------------------------ |
| GPT2-NeuroSteer                  |               todo                  |
| Gamma-2-2b-NeuroSteer             |                todo       |
| Multi-Media-NeuroSteer  |                       |

# TODO

- [x] Publish github page.
- [x] Publish demo and website.
- [x] Publish Auffusion and Auffusion-Full checkpoints.
- [x] Add README documantation.
- [x] Support gamma-2--2b and gpt-2.
- [x] deploy and demo.
- [ing] Explore multi-media steering.
- [ ] Scale to llama-3 .
- [ ] rebuttal.


## 📚 Citation
Please consider citing the following article if you found our work useful:
[~under review~]

<!-- 
第一次使用

```bash
git clone https://github.com/ckqqqq/Uncertainty.git
将你的文件复制到这个文件夹下

```

和团队其他人合并

```bash
git pull # 拉取别人的代码，默认自动合并，如果有冲突，vscode会有提醒，请手动合并
git add . # 将所在文件夹下的所有的文件 添加跟踪
git commit -m "simple English" # 你要提交的消息
# git branch -M main # 第一次需要使用
# git remote add origin https://github.com/ckqqqq/Uncertainty.git 第一次需要使用，如果是clone下来的不用
git push -u origin main # 将本地的main分支提交到远程 origin 分支上，不要强制提交，记得开setproxy
```

个人新分支的创建与合并

```bash
# 各人也可以开一个自己的name_dev分支用于个人开发，随后合并到主分支上，便于最终代码的维护
git checkout main    # 切换到 main 分支
git pull origin main # 拉取最新的 main 分支代码

git checkout -b name_dev # 创建并切换到 name_dev

git add .         # 暂存所有修改
git commit -m "描述你更改的消息" # 提交更改

git checkout main  # 切换回 main 分支
git merge name_dev # 将 name_dev 合并到 main

git push origin main # 将本地 main 分支推送到远程

git branch -d name_dev # 删除本地的临时分支
git push origin --delete name_dev # 删除远程的临时分支
```

## 实验计划

https://hqejk4h3h1.feishu.cn/wiki/BabzwVlApiYvslk9cjac8511n4g?from=from_copylink -->
