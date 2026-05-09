# CLIP 实战项目

从零实现 CLIP 模型 (ViT-B/32 + Transformer)，在 Flickr8k / COCO 上训练，
Gradio 图文检索 Demo，输出面试级技术报告。

## 快速开始

```bash
pip install -r requirements.txt

# Phase 1: 从零训练 CLIP (Flickr8k)
python scripts/train_phase1.py

# Phase 2: open_clip 训练 (COCO)
python scripts/train_phase2.py

# 评估
python scripts/eval_all.py --checkpoint checkpoints/phase1/best.pt --task all

# Demo
python src/demo/app.py --checkpoint checkpoints/phase2/best.pt --image-dir data/demo_images
```

## 项目结构

```
cilp_proj/
├── configs/          # 训练配置 (yaml)
├── src/
│   ├── model/        # ViT-B/32, Text Encoder, CLIP
│   ├── loss/         # InfoNCE 对比损失
│   ├── data/         # 数据加载 & 预处理
│   ├── train/        # 训练循环 & 工具
│   ├── eval/         # 检索 & 零样本评估
│   └── demo/         # Gradio Demo + FAISS
├── scripts/          # 训练/评测入口
├── docs/             # 设计文档 & 实现计划 & 技术报告
└── README.md
```

## 文档

- [设计文档](docs/superpowers/specs/2026-05-09-clip-practice-design.md)
- [实现计划](docs/superpowers/plans/2026-05-09-clip-practice-plan.md)
- [技术报告](docs/report.md)
