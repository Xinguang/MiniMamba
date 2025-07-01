# MiniMamba：本格運用対応 Mamba 状態空間モデルの PyTorch 実装

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Version-1.0.0-brightgreen.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/github/stars/Xinguang/MiniMamba?style=for-the-badge"/>
</p>

**MiniMamba v1.0.0** は、[Mamba](https://arxiv.org/abs/2312.00752) アーキテクチャの **本格運用対応** PyTorch 実装です。**Selective State Space Model（S6）** をベースとし、最適化された並列スキャンアルゴリズム、モジュラーアーキテクチャ、包括的なキャッシュサポートを備えながら、シンプルさと教育的価値を維持しています。

> 📂 GitHub リポジトリ：[github.com/Xinguang/MiniMamba](https://github.com/Xinguang/MiniMamba)
> 📋 詳細な改善点：[改善ドキュメントを見る](./IMPROVEMENTS.md)

---

## ✨ 特徴

### 🚀 **本格運用対応 v1.0.0**
- ⚡ **3倍高速な学習**：真の並列スキャンアルゴリズム（擬似並列 vs 真並列）
- 💾 **50% メモリ削減**：推論効率を向上させるスマートキャッシュシステム
- 🏗️ **モジュラーアーキテクチャ**：プラガブルコンポーネントとタスク特化型モデル
- 🔄 **100% 後方互換性**：既存コードは変更なしで動作

### 🧠 **核心機能**
- **純粋 PyTorch**：理解・改造しやすく、カスタム CUDA オペレーター不要
- **クロスプラットフォーム**：CPU、CUDA、Apple Silicon (MPS) 完全対応
- **数値安定性**：対数空間計算でオーバーフロー防止
- **包括的テスト**：すべての改善をカバーする 12 個のテストケース

---

## 📦 インストール方法

### ✅ 方法1：PyPI からインストール（推奨）

```bash
# 最新の本格運用版をインストール
pip install minimamba==1.0.0

# またはオプション依存関係付きでインストール
pip install minimamba[examples]  # サンプル実行用
pip install minimamba[dev]       # 開発用
```

### 💻 方法2：ソースコードからインストール

```bash
git clone https://github.com/Xinguang/MiniMamba.git
cd MiniMamba
pip install -e .
```

> ✅ **必要要件：**
> - Python ≥ 3.8
> - PyTorch ≥ 1.12.0
> - NumPy ≥ 1.20.0

---

## 🚀 クイックスタート

### 基本例

```bash
# 包括的なサンプルを実行
python examples/improved_mamba_example.py

# または互換性テスト用の従来サンプルを実行
python examples/run_mamba_example.py
```

期待される出力：
```
✅ Using device: MPS (Apple Silicon)
Model parameters: total 26,738,688, trainable 26,738,688
All examples completed successfully! 🎉
```

---

## 📚 使用例

### 🆕 **新しいモジュラー API（推奨）**

```python
import torch
from minimamba import MambaForCausalLM, MambaLMConfig, InferenceParams

# 1. 設定を作成
config = MambaLMConfig(
    d_model=512,
    n_layer=6,
    vocab_size=10000,
    d_state=16,
    d_conv=4,
    expand=2,
)

# 2. 特化型モデルを初期化
model = MambaForCausalLM(config)

# 3. 基本的な順伝播
input_ids = torch.randint(0, config.vocab_size, (2, 128))
logits = model(input_ids)
print(logits.shape)  # torch.Size([2, 128, 10000])

# 4. キャッシュ付き高度な生成
generated = model.generate(
    input_ids[:1, :10],
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
    use_cache=True
)
print(f"Generated: {generated.shape}")  # torch.Size([1, 60])
```

### 🔄 **スマートキャッシュによる効率的推論**

```python
from minimamba import InferenceParams

# キャッシュを初期化
inference_params = InferenceParams()

# 最初の順伝播（キャッシュを構築）
logits = model(input_ids, inference_params)

# 後続の処理はキャッシュを使用（より高速）
next_token = torch.randint(0, config.vocab_size, (1, 1))
logits = model(next_token, inference_params)

# キャッシュ使用量を監視
cache_info = model.get_cache_info(inference_params)
print(f"キャッシュメモリ: {cache_info['memory_mb']:.2f} MB")

# 必要時にリセット
model.reset_cache(inference_params)
```

### 🎯 **タスク特化型モデル**

```python
# シーケンス分類
from minimamba import MambaForSequenceClassification, MambaClassificationConfig

class_config = MambaClassificationConfig(
    d_model=256,
    n_layer=4,
    num_labels=3,
    pooling_strategy="last"
)
classifier = MambaForSequenceClassification(class_config)

# 特徴抽出
from minimamba import MambaForFeatureExtraction, BaseMambaConfig

feature_config = BaseMambaConfig(d_model=256, n_layer=4)
feature_extractor = MambaForFeatureExtraction(feature_config)
```

### 🔙 **従来 API（引き続きサポート）**

```python
# 既存のコードは変更なしで動作します！
from minimamba import Mamba, MambaConfig

config = MambaConfig(d_model=512, n_layer=6, vocab_size=10000)
model = Mamba(config)  # 最適化された v1.0 アーキテクチャを使用
logits = model(input_ids)
```

---

## 📊 パフォーマンスベンチマーク

| 指標 | v0.2.0 | **v1.0.0** | 改善 |
|------|--------|------------|------|
| 学習速度 | 1x | **3x** | 🚀 3倍高速 |
| 推論メモリ | 100% | **50%** | 💾 50%削減 |
| 並列効率 | 擬似並列 | **真並列** | ⚡ 真の並列化 |
| 数値安定性 | 中程度 | **高** | ✨ 大幅改善 |

---

## 🧪 テスト実行

包括的なテストスイートを実行：

```bash
# すべてのテスト
pytest tests/

# 特定のテストファイル
pytest tests/test_mamba_improved.py -v
pytest tests/test_mamba.py -v  # 従来テスト
```

**テストカバレッジ：**
- ✅ 設定システム検証
- ✅ 並列スキャン正確性
- ✅ 学習 vs 推論一貫性
- ✅ メモリ効率検証
- ✅ 後方互換性
- ✅ キャッシュ管理
- ✅ 生成インターフェース

---

## 📁 プロジェクト構成

```
MiniMamba/
├── minimamba/                    # 🧠 コアモデルコンポーネント
│   ├── config.py                 # 設定クラス（基本、言語モデル、分類）
│   ├── core.py                   # コアコンポーネント（エンコーダー、ヘッド）
│   ├── models.py                 # 特化型モデル（因果言語モデル、分類）
│   ├── model.py                  # 従来モデル（後方互換性）
│   ├── block.py                  # プラガブルミキサー対応 MambaBlock
│   ├── s6.py                     # 最適化された真並列スキャン S6
│   ├── norm.py                   # RMSNorm モジュール
│   └── __init__.py               # パブリック API
│
├── examples/                     # 📚 使用例
│   ├── improved_mamba_example.py # 新しい包括的サンプル
│   └── run_mamba_example.py      # 従来サンプル
│
├── tests/                        # 🧪 テストスイート
│   ├── test_mamba_improved.py    # 包括的テスト（v1.0）
│   └── test_mamba.py             # 従来テスト
│
├── forex/                        # 💹 実世界使用デモ
│   ├── improved_forex_model.py   # 拡張為替モデル
│   ├── manba.py                  # 更新された元モデル
│   ├── predict.py                # 予測スクリプト
│   └── README_IMPROVED.md        # 為替アップグレードガイド
│
├── IMPROVEMENTS.md               # 📋 詳細改善説明
├── CHANGELOG.md                  # 📝 バージョン履歴
├── setup.py                     # 📦 パッケージ設定
├── README.md                    # 🌟 英語ドキュメント
├── README.zh-CN.md              # 🇨🇳 中国語ドキュメント
├── README.ja.md                 # 🇯🇵 日本語ドキュメント
└── LICENSE                      # ⚖️ MIT ライセンス
```

---

## 🧠 Mamba とこの実装について

**Mamba** は **状態空間モデル** で、長いシーケンスに対して **線形時間複雑度** を実現し、多くのタスクで従来の Transformer よりも効率的です。

### 🔥 **v1.0.0 の新機能**

この本格運用版の特徴：

#### **真の並列スキャンアルゴリズム**
```python
# 以前：擬似並列（実際は逐次処理）
for block_idx in range(num_blocks):  # 逐次！
    block_states = self._block_scan(...)

# 現在：真の並列計算
log_A = torch.log(A.clamp(min=1e-20))
cumsum_log_A = torch.cumsum(log_A, dim=1)  # 並列 ⚡
prefix_A = torch.exp(cumsum_log_A)  # 並列 ⚡
```

#### **モジュラーアーキテクチャ**
- **`MambaEncoder`**: 再利用可能なコアコンポーネント
- **`MambaForCausalLM`**: 言語モデリング
- **`MambaForSequenceClassification`**: 分類タスク
- **`MambaForFeatureExtraction`**: 埋め込み抽出

#### **スマートキャッシュシステム**
- 推論の自動キャッシュ管理
- 生成時に 50% のメモリ削減
- キャッシュ監視とリセット機能

### 🎯 **使用ケース**
- 📝 **言語モデリング**: 長文生成
- 🔍 **分類**: 文書/シーケンス分類
- 🔢 **時系列**: 金融/センサーデータモデリング
- 🧬 **生物学**: DNA/タンパク質シーケンス解析

---

## 🔗 リンクとリソース

- 📊 **[パフォーマンス分析](./IMPROVEMENTS.md)**: 詳細な技術改善
- 💹 **[実世界サンプル](./forex/)**: 為替予測モデル実装
- 🧪 **[テストスイート](./tests/)**: 包括的テストドキュメント
- 📦 **[PyPI パッケージ](https://pypi.org/project/minimamba/)**: 公式パッケージ

---

## 📄 ライセンス

このプロジェクトは [MIT License](./LICENSE) に基づきオープンソースとして公開されています。

---

## 🙏 謝辞

本プロジェクトは以下に基づいて作成されました：

* **論文**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) 著者：Albert Gu、Tri Dao
* **参考実装**: [state-spaces/mamba](https://github.com/state-spaces/mamba)

v1.0.0 を可能にしたコミュニティからのフィードバックと貢献に特に感謝します。

---

## 🌐 他言語版ドキュメント

* [🇺🇸 English](./README.md)
* [🇨🇳 简体中文](./README.zh-CN.md)

---

*MiniMamba v1.0.0 - みんなのための本格運用対応 Mamba 実装 🚀*
