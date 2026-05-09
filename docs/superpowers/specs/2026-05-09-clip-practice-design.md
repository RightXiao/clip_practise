# CLIP 实战项目设计文档

**日期**: 2026-05-09
**目标**: 从零实现 CLIP 模型，掌握对比学习原理，并构建可交互的图文检索与零样本分类 Demo

---

## 方案概述

三层递进式方案：

| 阶段 | 内容 | 目的 |
|------|------|------|
| Phase 1 | 从零实现 CLIP（ViT-B/32 + 自建 Transformer），Flickr8k 训练 | 吃透原理 |
| Phase 2 | 用 open_clip 库 + 工业级训练技巧，COCO 子集训练 | 掌握工业级训练 |
| Phase 3 | Gradio Demo：文字搜图 / 以图搜图 / 零样本分类 | 可展示的完整闭环 |
| Phase 4 | 技术报告 | 面试用，覆盖模型详解 / 训练过程 / 评测分析 |

---

## 环境

- **GPU**: AutoDL 云端，单卡 RTX 4090D 24GB
- **开发**: 本地 vscode + 远程 SSH，git 同步代码
- **框架**: PyTorch 2.x, open-clip-torch, Gradio

---

## 项目结构

```
cilp_proj/
├── configs/
│   ├── phase1_scratch.yaml
│   └── phase2_finetune.yaml
├── data/                       # 数据集 (gitignore)
├── src/
│   ├── model/
│   │   ├── image_encoder.py    # ViT-B/32 图像编码器
│   │   ├── text_encoder.py     # Transformer 文本编码器
│   │   └── clip.py             # CLIP 整合 (dual encoder + projection)
│   ├── loss/
│   │   └── infonce.py          # InfoNCE 对比损失
│   ├── data/
│   │   ├── dataset.py          # 图文 pair 数据集
│   │   └── transforms.py       # 图像预处理 & 增强
│   ├── train/
│   │   ├── trainer.py          # 通用训练循环
│   │   └── utils.py            # AMP, lr schedule, checkpoint
│   ├── eval/
│   │   ├── retrieval.py        # Recall@K 检索评估
│   │   └── zeroshot.py         # 零样本分类评估
│   └── demo/
│       └── app.py              # Gradio 入口
├── scripts/
│   ├── train_phase1.py
│   ├── train_phase2.py
│   └── eval_all.py
├── requirements.txt
└── README.md
```

---

## 模型架构

### Image Encoder: ViT-B/32

| 模块 | 参数 | 说明 |
|------|------|------|
| Patch Embedding | 32×32 patch, 224×224 → 49 patches | Conv2d 投影到 768-dim |
| Position Embedding | 50 个位置 (49 + CLS) | 可学习参数 |
| Transformer Blocks | 12 层, 768-dim, 12 heads, MLP 3072 | Pre-Norm Transformer |
| Projection Head | 768 → 768 | 映射到共享嵌入空间 |

### Text Encoder

- BPE tokenizer（从训练数据学习，vocab size=10K）
- 6 层自建 Transformer, 512-dim, 8 heads
- 取 [EOS] token → 投影到 768-dim 共享空间
- 不使用预训练 BERT，保证理解完整 Transformer 流程

### CLIP 整体

双编码器输出 L2 归一化 → cosine similarity matrix → 对称 InfoNCE loss（image→text + text→image）。

---

## 评测指标

### 训练/验证共用

| 指标 | 说明 |
|------|------|
| Contrastive Loss (InfoNCE) | 训练/验证集损失曲线 |
| Image→Text Recall@K (K=1,5,10) | 给定图片，在 N 条文本中检索到正确描述的概率 |
| Text→Image Recall@K (K=1,5,10) | 给定文本，在 N 张图片中检索到正确图片的概率 |

### 零样本分类

| 指标 | 说明 |
|------|------|
| Top-1 / Top-5 Accuracy | 给定类别标签文本，预测准确率 |
| 混淆矩阵 | 可视化分类错误分布 |

---

## Phase 1: 从零实现 CLIP

### 数据集

Flickr8k: 8000 张图片，每张 5 条英文描述。

### 训练配置（4090D 优化）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| Batch size | 128 | effective 256 via grad accum ×2 |
| 精度 | bf16 + AMP | 4090 原生支持 bf16 |
| 优化器 | AdamW, lr=1e-4 | warmup 500 steps + cosine decay |
| Epochs | 30 | |
| Gradient Checkpointing | 开启 | 节省显存 |
| 图片尺寸 | 224×224 | |
| 验证频率 | 每 500 steps | |
| 预估耗时 | 2-3 小时 | |

### 成功标准

Recall@1 > 0（随机基线 = 1/N），模型学到图文关联，loss 稳定下降。

---

## Phase 2: open_clip 实战训练

### 数据集

COCO 2017 trainval 子集 ~30K 张图片（从 ~123K 中随机采样，均匀覆盖 80 个类别）。

### 训练配置（4090D 优化）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| Batch size | 128 | effective 512 via grad accum ×4 |
| 精度 | bf16 | |
| 显存优化 | gradient checkpointing + empty_cache | |
| 数据加载 | prefetch_factor=2, persistent_workers=True | |
| 数据增强 | RandomResizedCrop, RandAugment, 颜色抖动 | |
| 模型 | open_clip ViT-B/32 | 与 Phase 1 结构对齐，便于对比 |

### 输出

与 Phase 1 相同的评估指标，形成对比报告。

---

## Phase 3: Gradio Demo

### 功能

| Tab | 功能 | 说明 |
|------|------|------|
| 文字搜图 | 输入文字 → Top-K 相似图片 | 预计算特征 + FAISS 检索 |
| 以图搜图 | 上传图片 → 相似图片 | 图像特征做相似度搜索 |
| 零样本分类 | 上传图片 + 自定义类别 | 展示零样本泛化能力 |

### 技术细节

- 预计算图片库特征向量（numpy）
- FAISS 加速相似度搜索
- Gradio 3.x Blocks 界面

---

## Phase 4: 技术报告

### 目的

项目完成后，输出一份可应对面试追问的技术报告，体现对 CLIP 的深入理解和工程能力。

### 报告结构

| 章节 | 内容要点 | 预期篇幅 |
|------|----------|----------|
| **1. 摘要** | 项目动机、做了什么、核心结果 | 半页 |
| **2. CLIP 模型详解** | 对比学习原理、dual encoder 架构、InfoNCE loss 推导、为什么 ViT-B/32、共享嵌入空间设计 | 2-3 页 |
| **3. 训练过程** | 数据预处理、Patchify 细节、Position Encoding、训练配置与技巧、loss 曲线分析、Phase 1 vs Phase 2 对比 | 2-3 页 |
| **4. 评测与分析** | Recall@K 结果解读、零样本分类表现、bad case 分析、消融实验（如有）、Phase 1 手写 vs Phase 2 open_clip 对比 | 2-3 页 |
| **5. Demo 展示** | 检索效果截图、零样本分类示例、工程实现要点（FAISS、特征预计算） | 1 页 |
| **6. 总结与反思** | 关键收获、遇到的坑、改进方向 | 半页 |

### 关键面试问题覆盖

报告需确保能回答以下典型面试问题：

- CLIP 的对比学习与传统分类训练的区别是什么？
- InfoNCE loss 的公式推导和直觉理解
- 为什么用 ViT 而不是 ResNet？Patch size 的影响？
- 共享嵌入空间维度如何选择？L2 归一化的作用？
- 训练中遇到了什么问题？怎么解决的？
- Phase 1 手写和 Phase 2 open_clip 的结果差距？为什么？
- bf16 vs fp16 的区别？gradient checkpointing 的原理？
- 零样本分类的 prompt engineering 怎么做的？

### 输出格式

- Markdown 源文件：`docs/report.md`
- 支持导出 PDF（pandoc / LaTeX）
- 所有图表内嵌（loss 曲线、recall 对比、混淆矩阵、检索效果截图）

---

## 本地 → AutoDL 迁移

1. 现在本地完成设计 + 实现计划，git 管理代码
2. `git push` 到 GitHub / Gitee
3. AutoDL 上 `git clone`，安装依赖，按计划实现
4. 如果 AutoDL 上也有 Claude Code，直接喂 spec/plan 继续协作
