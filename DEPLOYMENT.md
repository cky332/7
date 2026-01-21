# I³-MRec 完整部署指南

本文档详细介绍如何从零开始部署和运行 I³-MRec 项目。

## 目录

1. [环境配置](#1-环境配置)
2. [数据集准备](#2-数据集准备)
3. [项目结构说明](#3-项目结构说明)
4. [运行项目](#4-运行项目)
5. [参数说明](#5-参数说明)
6. [常见问题](#6-常见问题)

---

## 1. 环境配置

### 方法一：使用 Conda (推荐)

```bash
# 1. 创建新的 conda 环境
conda env create -f environment.yml

# 2. 激活环境
conda activate i3mrec

# 3. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 方法二：手动配置 Conda 环境

```bash
# 1. 创建 Python 3.9 环境
conda create -n i3mrec python=3.9 -y

# 2. 激活环境
conda activate i3mrec

# 3. 安装 PyTorch (根据你的 CUDA 版本选择)
# CUDA 11.7
conda install pytorch==1.13.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 或者 CPU 版本
# conda install pytorch==1.13.0 torchvision torchaudio cpuonly -c pytorch

# 4. 安装其他依赖
conda install numpy=1.26.4 numba=0.60.0 scipy pandas tqdm -c conda-forge

# 5. 安装 pip 依赖
pip install tensorboardX faiss-cpu
# 如果有 GPU: pip install faiss-gpu
```

### 方法三：使用 pip (需要已安装 Python 3.9)

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 2. 安装依赖
pip install -r requirements.txt
```

### 验证环境

运行以下命令验证所有依赖是否正确安装：

```bash
python -c "
import torch
import numpy as np
import numba
import scipy
import pandas
import tqdm
import tensorboardX
import faiss

print('所有依赖已正确安装!')
print(f'PyTorch 版本: {torch.__version__}')
print(f'NumPy 版本: {np.__version__}')
print(f'Numba 版本: {numba.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 设备: {torch.cuda.get_device_name(0)}')
"
```

---

## 2. 数据集准备

### 数据集下载

项目支持三个数据集：

| 数据集 | 模态 | 下载链接 |
|--------|------|----------|
| Baby | 图像 + 文本 | [Google Drive](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing) |
| Clothing | 图像 + 文本 | [Google Drive](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing) |
| TikTok | 图像 + 文本 + 音频 | [Google Drive](https://drive.google.com/drive/folders/11wEn5k1Kzusj1GkdAlCcfS3GbBWGzFpX?usp=drive_link) |

### 数据集目录结构

下载后，将数据放入 `Data` 文件夹，目录结构如下：

```
项目根目录/
├── Data/
│   ├── baby/
│   │   ├── baby.inter          # 用户-商品交互数据
│   │   ├── image_feat.npy      # 图像特征 (CNN 提取)
│   │   └── text_feat.npy       # 文本特征 (Sentence-Transformers 提取)
│   │
│   ├── clothing/
│   │   ├── clothing.inter
│   │   ├── image_feat.npy
│   │   └── text_feat.npy
│   │
│   └── tiktok/
│       ├── tiktok.inter
│       ├── image_feat.npy
│       ├── text_feat.npy
│       └── audio_feat.npy      # 音频特征 (仅 TikTok)
│
├── main.py
├── model.py
└── ...
```

### 数据文件说明

1. **`{dataset}.inter`**: 用户-商品交互文件 (TSV 格式)
   - 包含字段: `userID`, `itemID`, `x_label`
   - `x_label`: 0=训练集, 1=验证集, 2=测试集

2. **`image_feat.npy`**: 图像特征矩阵
   - 形状: `(num_items, image_dim)`
   - 来源: CNN (如 ResNet) 提取的特征

3. **`text_feat.npy`**: 文本特征矩阵
   - 形状: `(num_items, text_dim)`
   - 来源: Sentence-Transformers 提取的特征

4. **`audio_feat.npy`** (仅 TikTok): 音频特征矩阵
   - 形状: `(num_items, audio_dim)`

### 自动生成的文件

首次运行时，程序会自动从 `.inter` 文件生成以下文件：

- `train.txt`: 训练集用户-商品交互
- `val.txt`: 验证集
- `test.txt`: 测试集
- `s_pre_adj_mat.npz`: 预计算的邻接矩阵

---

## 3. 项目结构说明

```
项目根目录/
├── main.py             # 主程序入口，参数解析和训练流程
├── model.py            # 模型定义 (MILK_model, MGCN)
├── session.py          # 训练/验证/测试会话管理
├── dataset_loader.py   # 多模态数据加载器
├── enviroment.py       # 环境配置 (设备、日志、路径)
├── evaluation.py       # 评估指标计算 (HR, Recall, NDCG)
├── criterion.py        # 损失函数 (BPR, InfoNCE, MSE)
├── mi_estimators.py    # 互信息估计器 (CLUB)
├── tool.py             # 工具函数
├── environment.yml     # Conda 环境配置
├── requirements.txt    # pip 依赖
└── Data/               # 数据集目录
```

---

## 4. 运行项目

### 快速开始

```bash
# 激活环境
conda activate i3mrec

# 运行 Baby 数据集 (完全模态)
python main.py --dataset baby --exp_mode ff

# 运行 Clothing 数据集 (缺失模态)
python main.py --dataset clothing --exp_mode mm --missing_rate 0.3
```

### 三种实验模式

| 模式 | 参数 | 说明 |
|------|------|------|
| Full-Full | `--exp_mode ff` | 完整训练 + 完整测试 |
| Full-Missing | `--exp_mode fm` | 完整训练 + 缺失测试 |
| Missing-Missing | `--exp_mode mm` | 缺失训练 + 缺失测试 |

### Baby 数据集完整命令

```bash
# Full Training Full Test
python main.py --dataset baby \
    --max_info_coeff 1e-3 \
    --min_info_coeff 1e-5 \
    --reg_coeff 1e-3 \
    --penalty_coeff 300 \
    --lr 1e-3 \
    --exp_mode ff

# Full Training Missing Test
python main.py --dataset baby \
    --max_info_coeff 1e-3 \
    --min_info_coeff 1e-5 \
    --reg_coeff 1e-3 \
    --penalty_coeff 300 \
    --lr 1e-3 \
    --missing_rate 0.3 \
    --exp_mode fm

# Missing Training Missing Test
python main.py --dataset baby \
    --max_info_coeff 1e-3 \
    --min_info_coeff 1e-5 \
    --reg_coeff 1e-3 \
    --penalty_coeff 300 \
    --lr 1e-3 \
    --missing_rate 0.3 \
    --exp_mode mm
```

### Clothing 数据集完整命令

```bash
# Full Training Full Test
python main.py --dataset clothing \
    --max_info_coeff 1e-2 \
    --min_info_coeff 1e-5 \
    --reg_coeff 1e-2 \
    --penalty_coeff 1 \
    --lr 1e-2 \
    --exp_mode ff

# Full Training Missing Test
python main.py --dataset clothing \
    --max_info_coeff 1e-2 \
    --min_info_coeff 1e-6 \
    --reg_coeff 1e-2 \
    --penalty_coeff 1 \
    --missing_rate 0.3 \
    --lr 1e-2 \
    --exp_mode fm

# Missing Training Missing Test
python main.py --dataset clothing \
    --max_info_coeff 1e-2 \
    --min_info_coeff 1e-6 \
    --reg_coeff 1e-2 \
    --penalty_coeff 1 \
    --missing_rate 0.3 \
    --lr 1e-2 \
    --exp_mode mm
```

### TikTok 数据集完整命令

```bash
# Full Training Full Test (3模态: 图像+文本+音频)
python main.py --dataset tiktok \
    --max_info_coeff 1e-3 \
    --min_info_coeff 1e-5 \
    --reg_coeff 1e-3 \
    --penalty_coeff 100 \
    --lr 1e-3 \
    --exp_mode ff
```

---

## 5. 参数说明

### 基本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `baby` | 数据集名称: baby, clothing, tiktok |
| `--exp_mode` | `ff` | 实验模式: ff, fm, mm |
| `--missing_rate` | `0.3` | 模态缺失比例 (0.0-1.0) |
| `--seed` | `2020` | 随机种子 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--free_emb_dimension` | `64` | 嵌入维度 |
| `--batch_size` | `2048` | 批次大小 |
| `--epoch` | `200` | 训练轮次 |
| `--lr` | `0.001` | 学习率 |

### 正则化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--reg_coeff` | `1e-3` | L2 正则化系数 |
| `--penalty_coeff` | `300` | 环境差异惩罚系数 |
| `--max_info_coeff` | `1e-3` | 最大化互信息系数 |
| `--min_info_coeff` | `1e-5` | 最小化互信息系数 |

### 设备参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_gpu` | `True` | 是否使用 GPU |
| `--device_id` | `cuda:0` | GPU 设备 ID |

---

## 6. 常见问题

### Q1: CUDA out of memory

```bash
# 减小批次大小
python main.py --dataset baby --batch_size 1024 --exp_mode ff
```

### Q2: 没有 GPU

```bash
# 使用 CPU 运行 (较慢)
python main.py --dataset baby --use_gpu False --exp_mode ff
```

### Q3: 数据集文件找不到

确保数据集放置在正确位置:
```
Data/{dataset_name}/{dataset_name}.inter
Data/{dataset_name}/image_feat.npy
Data/{dataset_name}/text_feat.npy
```

### Q4: 查看训练日志

训练日志保存在:
```
exp_report/{dataset}/{suffix}/log/
```

### Q5: TensorBoard 可视化

```bash
tensorboard --logdir exp_report/{dataset}/tensorboard/
```

---

## 输出说明

训练完成后，会输出以下指标 (k=10, 20):

- **HR@k** (Hit Rate): 命中率，推荐列表中是否包含正确商品
- **Recall@k**: 召回率，正确商品被推荐的比例
- **NDCG@k**: 归一化折扣累积收益，考虑排序位置的推荐质量

---

## 联系方式

如有问题，请参考原论文或在 GitHub 提交 Issue。
