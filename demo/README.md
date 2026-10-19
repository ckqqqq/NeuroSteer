# NeuroSteer demo：目录、用途与启动

## data、log、demo 在哪里？

| 内容 | 路径 | 当前状态 |
| --- | --- | --- |
| 数据预处理脚本 | `src/data/` | 仓库只有毒性和立场数据相关脚本／说明，不包含完整数据集 |
| 历史生成结果 | `results/generated_texts_*.md` | 可以直接阅读，用来分析成功和失败案例；不是实时生成 |
| 日志工具 | `src/log.py` | 训练／实验程序把 `execution.log` 写入所指定的输出目录 |
| 新的 demo 运行日志 | `logs/demo/backend.log`、`logs/demo/frontend.log` | 使用本目录启动脚本后生成 |
| 原服务器日志位置 | `demo/4090deploy/docker_log/` | 旧 README 提及的位置；当前仓库 没有这些服务器日志 |
| 后端 | `demo/backend/main_fastapp_backend.py` | FastAPI，默认 `127.0.0.1:8000` |
| 前端 | `demo/frontend/main_streamlit_frontend.py` | Streamlit，默认 `127.0.0.1:8501` |
| 神经元统计缓存 | `results/demo/demo_v1/` | 已从 Git 历史恢复四组；完整模型／SAE 权重需单独加载 |
| 演示动图 | `doc/demo_main_1.7b.gif` | 可用于查看历史交互方式 |

## 它还有什么用途？

它是 GPT-2 文本续写与 SAE 行为干预的研究演示。输入同一个文本前缀，可比较基线与情感、毒性、立场、礼貌干预，也支持组合干预。它不是现代指令模型聊天产品；页面记录聊天历史，但请求只发送本轮输入。滑块沿用旧代码的系数尺度，不等同于论文中的归一化 alpha。随机采样也会造成输出差异，单次对比不能单独证明干预效果。

四组任务缓存已恢复，来源与校验值见 `results/demo/RECOVERY.md`。缺失模型资源或加载失败时 API 返回 503，不会伪造推理结果。真实推理已在 DGX Spark 通过加载与生成测试，详见 [DGX 部署说明](deploy/spark/README.md)；这不等同于复现论文指标。

## 本地启动页面和 API

在仓库根目录，使用 Python 3.11 或 3.12 建议环境（本次 UI/API 在 Python 3.13 上验证）：

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r demo/requirements.txt
bash demo/run_backend.sh
```

另一个终端激活同一个环境后：

```bash
bash demo/run_fronted.sh
```

浏览器访问 `http://127.0.0.1:8501`，后端状态在 `http://127.0.0.1:8000/health`。端口不再使用旧的 10478。自定义端口时，请同时设置 `NEUROSTEER_PORT` 和前端的 `NEUROSTEER_BACKEND_URL`。用 Ctrl-C 停止服务。

## 恢复真实推理

1. 仓库已恢复以下四个任务的统计缓存。每个目录必须恰好包含一个 `*neuron_info*.pkl`：

```text
results/demo/demo_v1/
  gpt2-small_sentiment_layer_6_..._topK_100/*neuron_info*.pkl
  gpt2-small_toxicity_layer_6_..._topK_100/*neuron_info*.pkl
  gpt2-small_debate_layer_6_..._topK_100/*neuron_info*.pkl
  gpt2-small_polite_layer_6_..._topK_100/*neuron_info*.pkl
```

也可以设置 `NEUROSTEER_CACHE_DIR=/absolute/path/to/demo_v1`。这是项目自行计算的任务统计文件，不能用通用 SAE 权重替代。只加载你自己生成或可信来源的 pickle 文件。

2. 在独立研究环境安装推理依赖（原仓库记录 SAE Lens 5.5.2、TransformerLens 2.15.0、Transformers 4.47.1）。原 `requirements.txt` 中 `torch==2.5.0+cu121` 仅适用于相应 CUDA 环境，不适用于 macOS；按目标机器安装匹配的 PyTorch 版本。还需要 `python-dotenv`、`einops` 等原研究依赖。DGX Spark 的已验证环境版本见 `deploy/spark/requirements.txt`；其他平台尚未实机验证。
3. 启动后端。缓存齐全后才会尝试从 Hugging Face 下载 GPT-2 与 `gpt2-small-res-jb`、layer 6 的 SAE。没有 CUDA 时旧模型加载器走 CPU，性能需实际测量。只有 `/health` 返回 200/ready，才表示模型和八个方向的向量均已加载。

原 `src/scripts/demo_prepare_all.sh` 仍含旧服务器数据集路径、Python 路径和 CUDA 编号，不能在本机原样运行；恢复缓存优先于重跑全部数据处理。没有原始数据时，这次修复不能重新生成与论文相同的向量。

## 验证 API 与前端连接

```bash
python -m pip install httpx
python -m unittest discover -s demo/tests -v
```

这些测试覆盖缺失缓存、无效输入、任务方向映射、网络故障和 API 分发；用桩函数验证分发的测试不代表真实模型推理通过。

## VPS 与公网访问

参见 [部署配置](deploy/README.md)。页面通过反向代理开放，推理 API 保留在内部网络；轻量镜像只包含 UI/API，真实推理仍需准备缓存与匹配的研究环境。
