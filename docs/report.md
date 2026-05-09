# CLIP 实战项目技术报告

> **日期**: 2026-05
> **代码仓库**: <GitHub 链接>

## 1. 摘要

本项目从零实现了 CLIP（Contrastive Language-Image Pre-training）模型，包含手写 ViT-B/32 图像编码器 + Transformer 文本编码器，在 Flickr8k 上从零训练（Phase 1）验证对比学习原理，使用 open_clip 在 COCO 子集上工业级训练（Phase 2）进行对比，并构建 Gradio Demo 展示图文检索和零样本分类。

**核心结果**:

| 指标 | Phase 1 (手写, Flickr8k) | Phase 2 (open_clip, COCO) |
|------|--------------------------|---------------------------|
| I→T Recall@1 | [待填入] | [待填入] |
| T→I Recall@1 | [待填入] | [待填入] |
| 零样本 Top-1 | [待填入] | [待填入] |

## 2. CLIP 模型详解

### 2.1 对比学习原理

CLIP 使用对比学习训练，与传统分类训练的核心区别在于：分类训练学习固定的类别边界，而对比学习学习图文在共享空间中的相对位置关系。每个 batch 中有 N 对 (image, text) 正样本对，其余 N*(N-1) 对为负样本，模型通过 InfoNCE loss 拉近正样本对、推远负样本对。

### 2.2 Dual Encoder 架构

CLIP 采用双编码器架构：
- **Image Encoder**: 将图片编码为固定维度向量
- **Text Encoder**: 将文本编码为同维度向量
- **Projection Head**: 将两个编码器的输出投影到共享嵌入空间
- **Similarity Matrix**: 计算所有 image-text 对的余弦相似度矩阵

### 2.3 ViT-B/32 图像编码器

ViT (Vision Transformer) 将图片切分为固定大小的 patch，视为序列输入 Transformer：

- **Patch Embedding**: 224×224 图片 → 32×32 patch → 7×7=49 patches → Conv2d 投影到 768-dim
- **CLS Token + Position Embedding**: 50 个位置 (1 CLS + 49 patches)，可学习位置编码
- **12 层 Pre-Norm Transformer**: 768-dim hidden, 12 heads, MLP 3072
- **CLS → Projection Head**: 取 CLS token 输出，线性投影 768 → 768

**为什么选 ViT-B/32**: ViT 的自注意力机制能捕捉全局特征，比 CNN 更适合多模态对齐；patch_size=32 在速度和粒度之间平衡；参数量 88M，适合单卡训练。

### 2.4 文本编码器

- **BPE Tokenizer**: vocab=10K, 特殊 tokens [PAD][UNK][BOS][EOS], 从训练数据学习
- **6 层 Pre-Norm Transformer**: 512-dim, 8 heads, MLP 2048
- **EOS Token → Projection Head**: 取序列最后一个非 PAD token → 线性投影 512 → 768

**为什么自建不用 BERT**: 轻量可控（25M 参数），学习 Transformer 完整流程，避免预训练模型引入外部知识偏差。

### 2.5 InfoNCE Loss 推导

给定 batch 内 N 对 (I_i, T_i) 正样本对：

$$\mathcal{L} = \frac{1}{2N} \left[ \sum_i -\log\frac{\exp(sim(I_i, T_i)/\tau)}{\sum_j \exp(sim(I_i, T_j)/\tau)} + \sum_i -\log\frac{\exp(sim(T_i, I_i)/\tau)}{\sum_j \exp(sim(T_i, I_j)/\tau)} \right]$$

其中 sim(a,b)=a·b/|a||b| 为余弦相似度，τ 为可学习温度参数（初始值 0.07）。对称损失同时优化 image→text 和 text→image 两个方向。

### 2.6 共享嵌入空间

- **维度选择 768**: 与 ViT-B/32 输出对齐，避免降维信息瓶颈
- **L2 归一化**: 将向量约束在单位球面上，使得内积等价于余弦相似度
- **温度参数 τ**: 控制 softmax 的锐度 —— τ 越小，分布越尖锐，模型对难负样本越敏感

## 3. 训练过程

### 3.1 数据预处理

- **Flickr8k**: 8000 张图, 每张 5 条描述 → 40K image-text pairs
- **BPE Tokenizer 训练**: 在全部描述文本上训练，vocab size 10K
- **图像增强**: RandomResizedCrop (0.8-1.0), RandomHorizontalFlip, ColorJitter (0.2)

### 3.2 训练配置

| 配置项 | Phase 1 | Phase 2 |
|--------|---------|---------|
| Batch Size (per GPU) | 128 (eff 256) | 128 (eff 512) |
| 精度 | bf16 + AMP | bf16 |
| 优化器 | AdamW, lr=1e-4 | AdamW, lr=5e-4 |
| 学习率调度 | Warmup 500 + Cosine | Warmup 2000 + Cosine |
| Epochs | 30 | 30 |
| 显存优化 | Gradient Checkpointing | Grad CKPT + empty_cache |
| 预估耗时 | 2-3 小时 (4090D) | 4-6 小时 (4090D) |

### 3.3 Loss 曲线

[插入 Phase 1 和 Phase 2 的 train/val loss 曲线图]

### 3.4 显存优化技术详解

- **bf16 vs fp16**: bf16 与 fp32 指数位相同（8位），动态范围大，不需要 loss scaling；但尾数位少（7位 vs 10位），精度略低。4090 原生支持 bf16，推荐使用。
- **Gradient Checkpointing**: 前向传播时不保存中间激活值，反向传播时重新计算。将显存从 O(n) 降至 O(sqrt(n))，以 20-30% 计算时间为代价。
- **Gradient Accumulation**: 将 batch 拆分为多步累积梯度再更新参数，模拟大 batch 训练效果。4090D 24GB 下 batch=128 × accum=2 = effective 256。

## 4. 评测与分析

### 4.1 图文检索

| Recall@K | Phase 1 | Phase 2 |
|----------|---------|---------|
| I→T R@1 | [待填入] | [待填入] |
| I→T R@5 | [待填入] | [待填入] |
| I→T R@10 | [待填入] | [待填入] |
| T→I R@1 | [待填入] | [待填入] |
| T→I R@5 | [待填入] | [待填入] |
| T→I R@10 | [待填入] | [待填入] |

### 4.2 零样本分类

[插入混淆矩阵 + Top-1/Top-5 准确率表格]

### 4.3 Bad Case 分析

[挑选检索/分类失败的典型例子，分析原因]

### 4.4 Phase 1 vs Phase 2 对比

| 维度 | Phase 1 (手写) | Phase 2 (open_clip) |
|------|---------------|---------------------|
| 代码量 | ~500 lines | ~50 lines (配置) |
| 训练速度 | 较慢（naive 实现） | 较快（优化实现） |
| 灵活度 | 完全可控 | 受限于库接口 |
| 学习深度 | 深入理解每行代码 | 表层使用 |
| 效果 | [待填入] | [待填入] |

## 5. Demo 展示

### 5.1 文字搜图示例

[截图: 输入文字描述，返回 Top-K 相似图片]

### 5.2 以图搜图示例

[截图: 上传图片，返回最相似的图片]

### 5.3 零样本分类示例

[截图: 上传图片 + 自定义类别标签，预测结果]

### 5.4 工程实现要点

- **FAISS IndexFlatIP**: L2 归一化后内积搜索 = 余弦相似度，百万级图片毫秒返回
- **特征预计算**: 图片库特征离线提取并缓存，避免每次检索重复编码
- **Gradio Blocks**: 三 Tab 交互界面，支持文字搜图 / 以图搜图 / 零样本分类

## 6. 总结与反思

### 关键收获

1. 对比学习的核心是构造高质量正负样本对，通过对比损失学习语义对齐
2. ViT 的 Patchify + Self-Attention 机制能有效捕捉图片全局特征
3. L2 归一化 + 可学习温度参数 τ 是控制对比学习训练稳定性的关键技巧
4. 工业级库（open_clip）在显存管理、数据增强等方面的优化远超 naive 实现

### 踩过的坑

1. [待填入: 训练不稳定? 学习率/温度参数调整]
2. [待填入: ViT 显存爆炸? grad checkpointing 位置选择]
3. [待填入: BPE tokenizer 特殊 token 处理]

### 改进方向

1. 更大数据集（Conceptual Captions 3M / LAION-400M）
2. 更大模型（ViT-L/14, 更大的文本编码器）
3. 更先进的对齐方法（SigLIP, CoCa）
4. 多卡分布式训练（DDP / FSDP）
