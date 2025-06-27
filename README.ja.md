# MiniMamba：Mamba 状態空間言語モデルの最小 PyTorch 実装

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/github/stars/Xinguang/MiniMamba?style=for-the-badge"/>
</p>

**MiniMamba** は、[Mamba](https://arxiv.org/abs/2312.00752) アーキテクチャの最小・簡潔な PyTorch 実装です。**Selective State Space Model（S6）** をベースとし、効率的なシーケンスモデリング（言語モデリングなど）を実現します。

この実装は以下を目的に設計されています：
- ✅ 教育用途に最適なシンプル設計
- ✅ カスタム CUDA カーネル不要
- ✅ CPU / CUDA / Apple MPS に完全対応

> 📂 GitHub リポジトリ：[github.com/Xinguang/MiniMamba](https://github.com/Xinguang/MiniMamba)

---

## ✨ 特徴

- 🧠 **完全 PyTorch 実装**：読みやすく改造しやすい
- ⚙️ **自己完結型**：軽量で依存も少ない
- 🔁 **自動回帰生成対応**：キャッシュを活用した高速生成
- 🧪 **テスト同梱**：`pytest` による単体テスト
- 📘 **学習に最適**：論文理解や構造解析におすすめ

---

## 📦 インストール方法

MiniMamba は次の 2 通りの方法でインストールできます：

### ✅ 方法1：PyPI からインストール（推奨）

```bash
pip install minimamba
```

### 💻 方法2：ソースコードからインストール（開発・最新版向け）

```bash
git clone https://github.com/Xinguang/MiniMamba.git
cd MiniMamba
pip install -e .
```

> 補足：`-e` は「編集可能モード」で、ソースコードの変更が即時反映されます。

> ✅ 必要要件：
>
> * Python ≥ 3.8
> * PyTorch ≥ 1.12
> * pytest（テスト実行に使用）

---

## 🚀 クイックスタート

以下のスクリプトで、モデルの初期化と前向き推論を確認できます：

```bash
python examples/run_mamba_example.py
```

出力例：

```
✅ Using device: MPS (Apple Silicon)
Total model parameters: 26,738,688
Input shape: torch.Size([2, 128])
Output shape: torch.Size([2, 128, 10000])
Inference time: 0.1524 seconds
```

---

## 🛠️ 使用例

```python
import torch
from minimamba import Mamba, MambaConfig

# 1. MambaConfig クラスでモデル構成を定義
config = MambaConfig(
    d_model=512,
    n_layer=6,
    vocab_size=10000,
    d_state=16,
    d_conv=4,
    expand=2,
)

# 2. 構成オブジェクトでモデルを初期化
model = Mamba(config=config)

# 3. ダミー入力
input_ids = torch.randint(0, config.vocab_size, (2, 128))
logits = model(input_ids)

# 注意：パフォーマンスのため、出力の語彙サイズはパディングされる場合があります
print(logits.shape)  # torch.Size([2, 128, 10008])
```

### 🔁 自動回帰生成（キャッシュ付き）

```python
class InferenceCache:
    def __init__(self):
        self.seqlen_offset = 0
        self.key_value_memory_dict = {}

inference_params = InferenceCache()

# 1トークンずつの生成シミュレーション
input1 = torch.randint(0, config.vocab_size, (1, 1))
logits1 = model(input1, inference_params=inference_params)
inference_params.seqlen_offset += 1

input2 = torch.randint(0, config.vocab_size, (1, 1))
logits2 = model(input2, inference_params=inference_params)
```

---

## 🧪 テスト実行

`pytest` によりテスト可能：

```bash
pytest tests/
```

含まれるテスト：

* ✅ モデル構築の検証
* ✅ 出力サイズの確認
* ✅ 長さ 0 の入力処理の確認

---

## 📁 プロジェクト構成

```
MiniMamba/
├── minimamba/              # モデル本体
│   ├── config.py           # MambaConfig 構成クラス
│   ├── model.py            # Mamba モデルクラス
│   ├── block.py            # MambaBlock（正規化 + 残差）
│   ├── s6.py               # S6 状態空間層
│   ├── norm.py             # RMSNorm 実装
│   └── __init__.py
│
├── examples/
│   └── run_mamba_example.py
│
├── tests/
│   └── test_mamba.py
│
├── requirements.txt
├── setup.py
├── README.md
├── README.zh-CN.md
├── README.ja.md
└── LICENSE
```

---

## 🧠 モデル解説

Mamba アーキテクチャは、状態空間モデル（SSM）を使って長距離依存を効率的に扱う新しいアプローチです：

* **線形時間複雑度**（Transformer の O(n²) に対し O(n)）
* 選択的状態スキャン（Selective Scan）により情報を圧縮・伝播
* トークン生成時に内部状態をキャッシュし再利用可能

含まれる主要コンポーネント：

* `S6`：選択的状態空間スキャン層
* `MambaBlock`：正規化 + 残差付きの処理ユニット
* `Mamba`：埋め込み + 層積み重ね + 出力ヘッド

---

## 📄 ライセンス

このプロジェクトは [MIT License](./LICENSE) に基づきオープンソースとして公開されています。

---

## 🙏 謝辞

本リポジトリは以下の研究・実装に基づいて作成されました：

* 論文：[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
  著者：Albert Gu、Tri Dao
* 公式実装：[state-spaces/mamba](https://github.com/state-spaces/mamba)

素晴らしい研究と公開に感謝します！

---

## 🌐 他言語版

* [English](./README.md)
* [简体中文](./README.zh-CN.md)
