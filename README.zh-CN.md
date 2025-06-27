# MiniMamba：Mamba 状态空间语言模型的极简 PyTorch 实现

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/github/stars/Xinguang/MiniMamba?style=for-the-badge"/>
</p>

**MiniMamba** 是 [Mamba](https://arxiv.org/abs/2312.00752) 架构的纯 PyTorch 简洁实现。它基于 **选择性状态空间模型（S6）**，可用于高效的序列建模任务（如语言建模）。

本项目专为教育、易读性与移植性设计：
- ✅ 无需自定义 CUDA kernel
- ✅ 支持 CPU、CUDA 和 Apple MPS
- ✅ 结构清晰，便于理解与扩展

> 📂 GitHub 仓库：[github.com/Xinguang/MiniMamba](https://github.com/Xinguang/MiniMamba)

---

## ✨ 项目特点

- 🧠 **纯 PyTorch 实现**：适配任何支持 PyTorch 的平台
- 📦 **结构简洁**：模块独立，便于调试与教学
- ⚡ **高效推理**：支持缓存机制的自回归生成
- 🧪 **测试完善**：附带 `pytest` 单元测试
- 📖 **适合学习**：理想的研究和教学项目

---

## 📦 安装方法

你可以通过以下两种方式安装 MiniMamba：

### ✅ 方式一：通过 PyPI 官方源安装（推荐）

```bash
pip install minimamba
```

### 💻 方式二：从源码安装（用于开发或最新代码）

```bash
git clone https://github.com/Xinguang/MiniMamba.git
cd MiniMamba
pip install -e .
```

> 注：`-e` 表示“可编辑模式”，源码改动可立即生效。

> ✅ 依赖要求：
>
> * Python ≥ 3.8
> * PyTorch ≥ 1.12
> * pytest（用于测试）

---

## 🚀 快速开始

运行示例脚本：

```bash
python examples/run_mamba_example.py
```

输出示例：

```
✅ Using device: MPS (Apple Silicon)
Total model parameters: 26,738,688
Input shape: torch.Size([2, 128])
Output shape: torch.Size([2, 128, 10000])
Inference time: 0.1524 seconds
```

---

## 🛠️ 使用示例

```python
import torch
from minimamba import Mamba, MambaConfig

# 1. 使用 MambaConfig 类定义模型配置
config = MambaConfig(
    d_model=512,
    n_layer=6,
    vocab_size=10000,
    d_state=16,
    d_conv=4,
    expand=2,
)

# 2. 使用配置对象初始化模型
model = Mamba(config=config)

# 3. 构造输入
input_ids = torch.randint(0, config.vocab_size, (2, 128))
logits = model(input_ids)

# 注意：为了性能，输出词表大小可能被填充
print(logits.shape)  # torch.Size([2, 128, 10008])
```

### 🔁 自回归推理（支持缓存）

```python
class InferenceCache:
    def __init__(self):
        self.seqlen_offset = 0
        self.key_value_memory_dict = {}

inference_params = InferenceCache()

# 模拟逐 token 生成
input1 = torch.randint(0, config.vocab_size, (1, 1))
logits1 = model(input1, inference_params=inference_params)
inference_params.seqlen_offset += 1

input2 = torch.randint(0, config.vocab_size, (1, 1))
logits2 = model(input2, inference_params=inference_params)
```

---

## 🧪 运行测试

使用 `pytest` 执行所有单元测试：

```bash
pytest tests/
```

包含以下测试用例：

* ✅ 模型结构是否构建成功
* ✅ 输出形状是否正确
* ✅ 零长度序列是否能正确处理

---

## 📂 项目结构

```
MiniMamba/
├── minimamba/              # 模型核心模块
│   ├── config.py           # MambaConfig 配置类
│   ├── model.py            # 完整模型定义
│   ├── block.py            # MambaBlock（带残差）
│   ├── s6.py               # S6 状态空间层
│   ├── norm.py             # RMSNorm 实现
│   └── __init__.py
│
├── examples/
│   └── run_mamba_example.py
│
├── tests/
│   └── test_mamba.py       # 单元测试
│
├── requirements.txt
├── setup.py
├── README.md
├── README.zh-CN.md
├── README.ja.md
└── LICENSE
```

---

## 🧠 模型原理简述

Mamba 是一种基于状态空间模型（SSM）的架构，它能够：

* 以 **线性时间复杂度** 处理长序列（相比 Transformer 的二次复杂度）
* 使用选择性扫描操作（Selective Scan）压缩状态信息
* 有效建模长程依赖，内存与计算效率更优

本实现包含：

* ✅ `S6`：核心状态空间扫描层
* ✅ `MambaBlock`：预归一化 + 残差结构
* ✅ `Mamba`：嵌入 + 多层堆叠 + 输出头

---

## 📄 开源协议

本项目使用 [MIT License](./LICENSE) 开源，允许自由使用与修改。

---

## 🙏 致谢

本项目参考并致敬以下作品：

* 论文：[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
  作者：Albert Gu 与 Tri Dao
* 官方实现：[state-spaces/mamba](https://github.com/state-spaces/mamba)

衷心感谢原作者的卓越贡献！

---

## 🌐 其他语言版本

* [English](./README.md)
* [日本語](./README.ja.md)
