# CLIP 实战项目实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零实现 CLIP 模型（ViT-B/32 + 自建 Transformer），在 Flickr8k 和 COCO 上训练，构建 Gradio 图文检索 Demo，输出面试级技术报告。

**Architecture:** Dual-encoder 架构 — ViT-B/32 编码图像（Patch Embedding → 12 层 Pre-Norm Transformer → CLS → Projection），自建 6 层 Transformer 编码文本（BPE → Transformer → EOS → Projection），双输出 L2 归一化后计算对称 InfoNCE loss。分 4 个 Phase 递进：从零手写 → open_clip 工业训练 → Demo → 报告。

**Tech Stack:** PyTorch 2.x, open-clip-torch, Gradio 3.x, FAISS, HuggingFace tokenizers, OmegaConf

---

## Phase 1: 从零实现 CLIP

### Task 1: 项目初始化

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `configs/phase1_scratch.yaml`

- [ ] **Step 1: 创建 requirements.txt**

```text
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
open-clip-torch>=2.20.0
gradio>=3.40.0
faiss-cpu
omegaconf
tqdm
matplotlib
pillow
numpy
tokenizers
```

- [ ] **Step 2: 创建 .gitignore**

```
data/
checkpoints/
__pycache__/
*.pyc
.DS_Store
.ipynb_checkpoints/
wandb/
```

- [ ] **Step 3: 创建 configs/phase1_scratch.yaml**

```yaml
# Phase 1: From-scratch CLIP training config
data:
  flickr8k_root: "data/flickr8k"
  flickr8k_captions: "data/flickr8k/captions.txt"
  val_split: 0.1
  img_size: 224
  max_text_len: 77

model:
  image:
    name: "vit_b32"
    patch_size: 32
    hidden_dim: 768
    num_layers: 12
    num_heads: 12
    mlp_dim: 3072
  text:
    vocab_size: 10000
    hidden_dim: 512
    num_layers: 6
    num_heads: 8
    mlp_dim: 2048
    max_len: 77
  proj_dim: 768
  temperature_init: 0.07

train:
  batch_size: 128
  grad_accum_steps: 2       # effective batch = 256
  epochs: 30
  lr: 1.0e-4
  weight_decay: 0.2
  warmup_steps: 500
  val_every: 500
  precision: "bf16"
  gradient_checkpointing: true
  num_workers: 4

output:
  checkpoint_dir: "checkpoints/phase1"
  log_dir: "logs/phase1"
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .gitignore configs/phase1_scratch.yaml
git commit -m "feat: initialize project structure with configs"
```

---

### Task 2: ViT-B/32 Image Encoder

**Files:**
- Create: `src/__init__.py`
- Create: `src/model/__init__.py`
- Create: `src/model/image_encoder.py`

- [ ] **Step 1: 实现 ViT-B/32 Image Encoder**

```python
"""ViT-B/32 Image Encoder for CLIP."""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class PatchEmbedding(nn.Module):
    """Split image into patches and project to hidden_dim."""

    def __init__(self, patch_size: int = 32, in_channels: int = 3, hidden_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) -> (B, 768, 7, 7) -> (B, 49, 768)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block with MLP."""

    def __init__(self, hidden_dim: int = 768, num_heads: int = 12, mlp_dim: int = 3072):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTB32(nn.Module):
    """ViT-B/32: 12-layer, 768-dim, 12-head transformer for 224x224 images."""

    def __init__(
        self,
        patch_size: int = 32,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        proj_dim: int = 768,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_channels=3, hidden_dim=hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, hidden_dim))  # 1 CLS + 49 patches
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.use_grad_ckpt = use_gradient_checkpointing
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for block in self.blocks:
            for p in block.parameters():
                if p.dim() > 1:
                    nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        x = self.patch_embed(x)                          # (B, 49, 768)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)             # (B, 50, 768)
        x = x + self.pos_embed

        for block in self.blocks:
            if self.use_grad_ckpt and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]                                # (B, 768)
        return self.proj(cls_out)                        # (B, proj_dim)
```

- [ ] **Step 2: Commit**

```bash
git add src/__init__.py src/model/__init__.py src/model/image_encoder.py
git commit -m "feat: add ViT-B/32 image encoder"
```

---

### Task 3: Text Encoder

**Files:**
- Create: `src/model/text_encoder.py`

- [ ] **Step 1: 实现 Text Encoder**

```python
"""Text Encoder for CLIP: BPE tokenizer + 6-layer Transformer."""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class TextTransformerBlock(nn.Module):
    """Pre-Norm Transformer block."""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, mlp_dim: int = 2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    """6-layer transformer text encoder with learned position embeddings."""

    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        max_len: int = 77,
        proj_dim: int = 768,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        self.blocks = nn.ModuleList([
            TextTransformerBlock(hidden_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.use_grad_ckpt = use_gradient_checkpointing
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for block in self.blocks:
            for p in block.parameters():
                if p.dim() > 1:
                    nn.init.trunc_normal_(p, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L) integer token indices
        x = self.token_embed(token_ids)                  # (B, L, 512)
        x = x + self.pos_embed[:, :x.size(1), :]

        for block in self.blocks:
            if self.use_grad_ckpt and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)
        # take the last non-padding token position (simplified: argmax of token_ids)
        eos_pos = (token_ids > 0).sum(dim=-1) - 1  # last non-pad position
        eos_pos = eos_pos.clamp(min=0)
        eos_out = x[torch.arange(x.size(0)), eos_pos]    # (B, 512)
        return self.proj(eos_out)                        # (B, proj_dim)
```

- [ ] **Step 2: Commit**

```bash
git add src/model/text_encoder.py
git commit -m "feat: add text encoder with 6-layer transformer"
```

---

### Task 4: CLIP Model + InfoNCE Loss

**Files:**
- Create: `src/loss/__init__.py`
- Create: `src/loss/infonce.py`
- Create: `src/model/clip.py`

- [ ] **Step 1: 实现 InfoNCE Loss**

```python
"""InfoNCE / symmetric contrastive loss for CLIP."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipLoss(nn.Module):
    """Symmetric InfoNCE loss on cosine similarity matrix."""

    def __init__(self, temperature_init: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / temperature_init))
        )

    def forward(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_embeds @ text_embeds.t()  # (B, B)

        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2.0

        return loss, logit_scale.detach()
```

- [ ] **Step 2: 实现 CLIP wrapper**

```python
"""CLIP model: dual encoder + contrastive loss."""
import torch
import torch.nn as nn

from .image_encoder import ViTB32
from .text_encoder import TextEncoder
from ..loss.infonce import ClipLoss


class CLIP(nn.Module):
    """CLIP model combining ViT image encoder and transformer text encoder."""

    def __init__(
        self,
        image_config: dict,
        text_config: dict,
        proj_dim: int = 768,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        self.image_encoder = ViTB32(
            patch_size=image_config.get("patch_size", 32),
            hidden_dim=image_config.get("hidden_dim", 768),
            num_layers=image_config.get("num_layers", 12),
            num_heads=image_config.get("num_heads", 12),
            mlp_dim=image_config.get("mlp_dim", 3072),
            proj_dim=proj_dim,
            use_gradient_checkpointing=image_config.get("gradient_checkpointing", True),
        )
        self.text_encoder = TextEncoder(
            vocab_size=text_config.get("vocab_size", 10000),
            hidden_dim=text_config.get("hidden_dim", 512),
            num_layers=text_config.get("num_layers", 6),
            num_heads=text_config.get("num_heads", 8),
            mlp_dim=text_config.get("mlp_dim", 2048),
            max_len=text_config.get("max_len", 77),
            proj_dim=proj_dim,
            use_gradient_checkpointing=text_config.get("gradient_checkpointing", True),
        )
        self.loss_fn = ClipLoss(temperature_init)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(token_ids)

    def forward(
        self, images: torch.Tensor, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(token_ids)
        loss, logit_scale = self.loss_fn(image_embeds, text_embeds)
        return loss, logit_scale
```

- [ ] **Step 3: Commit**

```bash
git add src/loss/__init__.py src/loss/infonce.py src/model/clip.py
git commit -m "feat: add CLIP model wrapper and InfoNCE loss"
```

---

### Task 5: Data Pipeline

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/transforms.py`
- Create: `src/data/dataset.py`

- [ ] **Step 1: 实现图像预处理 transforms**

```python
"""Image transforms for CLIP training."""
import torchvision.transforms as T


def get_train_transforms(img_size: int = 224):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                     std=(0.26862954, 0.26130258, 0.27577711)),
    ])


def get_val_transforms(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                     std=(0.26862954, 0.26130258, 0.27577711)),
    ])
```

- [ ] **Step 2: 实现 Flickr8k Dataset**

```python
"""Flickr8k dataset for CLIP training."""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def _train_bpe_tokenizer(captions: list[str], vocab_size: int = 10000) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for c in captions:
            f.write(c.lower() + "\n")
        tmp_path = f.name
    tokenizer.train([tmp_path], trainer)
    os.unlink(tmp_path)
    return tokenizer


class Flickr8kDataset(Dataset):
    """Flickr8k image-caption pairs.

    Expects captions.txt in format: image_name<TAB>caption
    """

    def __init__(self, root_dir: str, captions_file: str, transform, max_len: int = 77):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_len = max_len

        self.pairs = []
        captions_by_image = {}
        with open(captions_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_ref, caption = parts[0], parts[1]
                img_name = img_ref.split("#")[0]
                captions_by_image.setdefault(img_name, []).append(caption)

        all_captions = []
        for img_name, captions in captions_by_image.items():
            for caption in captions:
                self.pairs.append((img_name, caption.lower()))
                all_captions.append(caption.lower())

        self.tokenizer = _train_bpe_tokenizer(all_captions, vocab_size=10000)

    def __len__(self) -> int:
        return len(self.pairs)

    def _tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer.encode(f"[BOS] {text} [EOS]")
        ids = encoded.ids[:self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))  # PAD=0
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> tuple:
        img_name, caption = self.pairs[idx]
        img_path = self.root_dir / "Images" / img_name
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        token_ids = self._tokenize(caption)
        return image_tensor, token_ids


class ValFlickr8kDataset(Dataset):
    """Validation wrapper with different transform."""

    def __init__(self, base_dataset: Flickr8kDataset, indices: list[int], transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_name, caption = self.base.pairs[self.indices[idx]]
        img_path = self.base.root_dir / "Images" / img_name
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), self.base._tokenize(caption)


def create_dataloaders(
    root_dir: str,
    captions_file: str,
    batch_size: int = 128,
    val_split: float = 0.1,
    img_size: int = 224,
    max_len: int = 77,
    num_workers: int = 4,
):
    """Create train/val dataloaders from Flickr8k."""
    full_dataset = Flickr8kDataset(
        root_dir=root_dir,
        captions_file=captions_file,
        transform=get_train_transforms(img_size),
        max_len=max_len,
    )

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds_subset = random_split(full_dataset, [train_size, val_size])

    val_ds = ValFlickr8kDataset(full_dataset, val_ds_subset.indices,
                                 get_val_transforms(img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, full_dataset.tokenizer
```

- [ ] **Step 3: Commit**

```bash
git add src/data/__init__.py src/data/transforms.py src/data/dataset.py
git commit -m "feat: add Flickr8k dataset, transforms, and BPE tokenizer"
```

---

### Task 6: Training Utilities

**Files:**
- Create: `src/train/__init__.py`
- Create: `src/train/utils.py`

- [ ] **Step 1: 实现训练工具函数**

```python
"""Training utilities: optimizer, lr scheduler, checkpoint, AverageMeter."""
import math
import os
import torch
import torch.nn as nn


def create_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name or "norm" in name or "pos_embed" in name or "token_embed" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.999))


def cosine_schedule(optimizer, warmup_steps: int, total_steps: int, base_lr: float):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path: str):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path: str, device: torch.device = None):
    if device is None:
        device = torch.device("cpu")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
```

- [ ] **Step 2: Commit**

```bash
git add src/train/__init__.py src/train/utils.py
git commit -m "feat: add training utilities (optimizer, scheduler, checkpoint)"
```

---

### Task 7: Trainer

**Files:**
- Create: `src/train/trainer.py`

- [ ] **Step 1: 实现训练循环**

```python
"""Training loop for CLIP."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import AverageMeter, save_checkpoint, create_optimizer, cosine_schedule


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
    use_bf16: bool = True,
) -> float:
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    dtype = torch.bfloat16 if use_bf16 else torch.float16
    pbar = tqdm(loader, desc="Training")

    for batch_idx, (images, token_ids) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=dtype):
            loss, logit_scale = model(images, token_ids)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * grad_accum_steps, images.size(0))
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            "scale": f"{logit_scale:.2f}",
        })

    return loss_meter.avg


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_meter = AverageMeter()

    for images, token_ids in tqdm(loader, desc="Validating"):
        images = images.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, _ = model(images, token_ids)

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
):
    train_cfg = config["train"]
    output_cfg = config["output"]

    total_steps = len(train_loader) * train_cfg["epochs"]
    optimizer = create_optimizer(model, train_cfg["lr"], train_cfg["weight_decay"])
    scheduler = cosine_schedule(optimizer, train_cfg["warmup_steps"], total_steps, train_cfg["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg["precision"] == "fp16"))

    use_bf16 = train_cfg["precision"] == "bf16"
    best_val_loss = float("inf")
    os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, grad_accum_steps=train_cfg["grad_accum_steps"], use_bf16=use_bf16,
        )
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                epoch * len(train_loader), val_loss,
                os.path.join(output_cfg["checkpoint_dir"], "best.pt"),
            )

    return best_val_loss
```

- [ ] **Step 2: Commit**

```bash
git add src/train/trainer.py
git commit -m "feat: add CLIP training loop with bf16, grad accum, checkpoint"
```

---

### Task 8: Evaluation — Retrieval

**Files:**
- Create: `src/eval/__init__.py`
- Create: `src/eval/retrieval.py`

- [ ] **Step 1: 实现 Recall@K 检索评估**

```python
"""Image-Text retrieval evaluation with Recall@K."""
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def evaluate_retrieval(model, loader, device: torch.device) -> dict:
    """Compute image→text and text→image Recall@K.

    Returns dict with: i2t_r1, i2t_r5, i2t_r10, t2i_r1, t2i_r5, t2i_r10
    """
    model.eval()
    all_image_embeds = []
    all_text_embeds = []

    for images, token_ids in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        token_ids = token_ids.to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(token_ids)

        all_image_embeds.append(F.normalize(img_emb, dim=-1).cpu())
        all_text_embeds.append(F.normalize(txt_emb, dim=-1).cpu())

    image_embeds = torch.cat(all_image_embeds, dim=0)  # (N, D)
    text_embeds = torch.cat(all_text_embeds, dim=0)    # (N, D)

    sim = image_embeds @ text_embeds.t()  # (N, N)
    i2t = _recall_at_k(sim, ks=[1, 5, 10])
    t2i = _recall_at_k(sim.t(), ks=[1, 5, 10])

    return {
        "i2t_r1": i2t[0], "i2t_r5": i2t[1], "i2t_r10": i2t[2],
        "t2i_r1": t2i[0], "t2i_r5": t2i[1], "t2i_r10": t2i[2],
    }


def _recall_at_k(sim_matrix: torch.Tensor, ks: list[int]) -> list[float]:
    """sim_matrix: (num_queries, num_items). True match is on diagonal."""
    n = sim_matrix.size(0)
    labels = torch.arange(n)
    _, indices = sim_matrix.topk(max(ks), dim=1)
    correct = indices == labels.unsqueeze(1)
    return [correct[:, :k].any(dim=1).float().mean().item() for k in ks]
```

- [ ] **Step 2: Commit**

```bash
git add src/eval/__init__.py src/eval/retrieval.py
git commit -m "feat: add retrieval evaluation with Recall@K"
```

---

### Task 9: Evaluation — Zero-shot Classification

**Files:**
- Create: `src/eval/zeroshot.py`

- [ ] **Step 1: 实现零样本分类评估**

```python
"""Zero-shot classification evaluation helpers."""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


@torch.no_grad()
def zeroshot_classify(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
    """Classify images by cosine similarity to text prompts.

    image_embeds: (N, D) L2-normalized
    text_embeds: (C, D) L2-normalized  (C = num_classes)
    Returns: (N,) predicted class index
    """
    sim = image_embeds @ text_embeds.t()  # (N, C)
    return sim.argmax(dim=1)


def zeroshot_metrics(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    """Compute top-1 accuracy and confusion matrix."""
    correct_top1 = (predictions == labels).float().mean().item()

    labels_np = labels.cpu().numpy()
    preds_np = predictions.cpu().numpy()
    cm = sk_confusion_matrix(labels_np, preds_np, labels=range(num_classes))

    return {"top1_acc": correct_top1, "confusion_matrix": cm.tolist()}
```

- [ ] **Step 2: Commit**

```bash
git add src/eval/zeroshot.py
git commit -m "feat: add zero-shot classification evaluation helpers"
```

---

### Task 10: Phase 1 Training Script

**Files:**
- Create: `scripts/train_phase1.py`

- [ ] **Step 1: 实现 Phase 1 训练入口**

```python
"""Phase 1: Train from-scratch CLIP on Flickr8k."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf

from src.model.clip import CLIP
from src.data.dataset import create_dataloaders
from src.train.trainer import train


def main():
    config = OmegaConf.load("configs/phase1_scratch.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, tokenizer = create_dataloaders(
        root_dir=config.data.flickr8k_root,
        captions_file=config.data.flickr8k_captions,
        batch_size=config.train.batch_size,
        val_split=config.data.val_split,
        img_size=config.data.img_size,
        max_len=config.data.max_text_len,
        num_workers=config.train.get("num_workers", 4),
    )

    model = CLIP(
        image_config=OmegaConf.to_container(config.model.image),
        text_config=OmegaConf.to_container(config.model.text),
        proj_dim=config.model.proj_dim,
        temperature_init=config.model.temperature_init,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.1f}M")

    os.makedirs(config.output.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output.log_dir, exist_ok=True)

    best_loss = train(model, train_loader, val_loader,
                       OmegaConf.to_container(config), device)
    print(f"Training complete. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_phase1.py
git commit -m "feat: add Phase 1 training script"
```

---

### Task 11: Evaluation Script

**Files:**
- Create: `scripts/eval_all.py`

- [ ] **Step 1: 实现统一评估脚本**

```python
"""Unified evaluation: retrieval + zero-shot classification."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
from omegaconf import OmegaConf

from src.model.clip import CLIP
from src.eval.retrieval import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/phase1_scratch.yaml")
    parser.add_argument("--task", type=str, default="all",
                        choices=["retrieval", "zeroshot", "all"])
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIP(
        image_config=OmegaConf.to_container(config.model.image),
        text_config=OmegaConf.to_container(config.model.text),
        proj_dim=config.model.proj_dim,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from src.data.dataset import create_dataloaders
    _, val_loader, tokenizer = create_dataloaders(
        root_dir=config.data.flickr8k_root,
        captions_file=config.data.flickr8k_captions,
        batch_size=config.train.batch_size,
        val_split=config.data.val_split,
        img_size=config.data.img_size,
        max_len=config.data.max_text_len,
        num_workers=config.train.get("num_workers", 4),
    )

    if args.task in ("retrieval", "all"):
        metrics = evaluate_retrieval(model, val_loader, device)
        print("Retrieval Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    if args.task in ("zeroshot", "all"):
        print("Zero-shot evaluation requires a class-labeled dataset.")
        print("Use --dataset <path> with class labels.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/eval_all.py
git commit -m "feat: add unified evaluation script"
```

---

## Phase 2: open_clip Training

### Task 12: Phase 2 Config + Script

**Files:**
- Create: `configs/phase2_finetune.yaml`
- Create: `scripts/train_phase2.py`

- [ ] **Step 1: 创建 Phase 2 配置**

```yaml
# Phase 2: open_clip training on COCO subset
data:
  coco_root: "data/coco_subset"
  train_json: "data/coco_subset/train_subset.json"
  val_json: "data/coco_subset/val_subset.json"
  img_size: 224
  max_text_len: 77

model:
  arch: "ViT-B-32"
  pretrained: null

train:
  batch_size: 128
  grad_accum_steps: 4        # effective batch = 512
  epochs: 30
  lr: 5.0e-4
  weight_decay: 0.2
  warmup_epochs: 2
  warmup_steps: 2000
  val_every: 500
  precision: "bf16"
  gradient_checkpointing: true
  num_workers: 4

output:
  checkpoint_dir: "checkpoints/phase2"
  log_dir: "logs/phase2"
```

- [ ] **Step 2: 实现 Phase 2 训练脚本**

```python
"""Phase 2: Train CLIP using open_clip on COCO subset."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf


class COCOSubset(Dataset):
    def __init__(self, root: str, annotation_file: str, transform):
        with open(annotation_file) as f:
            self.data = json.load(f)  # list of {image, caption}
        self.root = root
        self.transform = transform
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(os.path.join(self.root, item["image"])).convert("RGB")
        img_tensor = self.transform(img)
        token_ids = self.tokenizer(item["caption"])
        return img_tensor, token_ids.squeeze(0)


class OpenClipWrapper(torch.nn.Module):
    """Wrap open_clip model to match our trainer's forward interface."""

    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, images, token_ids):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(token_ids)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i = torch.nn.functional.cross_entropy(logits, labels)
        loss_t = torch.nn.functional.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2
        return loss, logit_scale.detach()

    def encode_image(self, images):
        return torch.nn.functional.normalize(self.model.encode_image(images), dim=-1)

    def encode_text(self, token_ids):
        return torch.nn.functional.normalize(self.model.encode_text(token_ids), dim=-1)


def main():
    config = OmegaConf.load("configs/phase2_finetune.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.model.arch,
        pretrained=config.model.pretrained,
    )
    model.to(device)

    train_ds = COCOSubset(config.data.coco_root, config.data.train_json, preprocess)
    val_ds = COCOSubset(config.data.coco_root, config.data.val_json, preprocess)

    train_loader = DataLoader(
        train_ds, batch_size=config.train.batch_size, shuffle=True,
        num_workers=config.train.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.train.batch_size, shuffle=False,
        num_workers=config.train.num_workers, pin_memory=True,
    )

    wrapped = OpenClipWrapper(model)

    from src.train.trainer import train
    os.makedirs(config.output.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output.log_dir, exist_ok=True)

    best_loss = train(wrapped, train_loader, val_loader,
                       OmegaConf.to_container(config), device)
    print(f"Phase 2 training complete. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add configs/phase2_finetune.yaml scripts/train_phase2.py
git commit -m "feat: add Phase 2 config and open_clip training script"
```

---

## Phase 3: Gradio Demo

### Task 13: FAISS Index Builder + Gradio App

**Files:**
- Create: `src/demo/__init__.py`
- Create: `src/demo/app.py`

- [ ] **Step 1: 实现 Gradio Demo（含 FAISS 索引和三个 Tab）**

```python
"""Gradio demo: text-to-image, image-to-image, zero-shot classification."""
import os
import gradio as gr
import torch
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
import open_clip


class ClipDemo:
    def __init__(self, checkpoint_path: str, image_dir: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=checkpoint_path
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Build or load FAISS index
        index_path = os.path.join(os.path.dirname(image_dir), "demo_index.faiss")
        meta_path = os.path.join(os.path.dirname(image_dir), "demo_meta.npy")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            self.image_files = np.load(meta_path).tolist()
        else:
            self.index = self._build_index()
            faiss.write_index(self.index, index_path)
            np.save(meta_path, np.array(self.image_files))

    @torch.no_grad()
    def _extract_features(self, images):
        features = []
        for img_tensor in tqdm(images, desc="Extracting features"):
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                feat = torch.nn.functional.normalize(self.model.encode_image(img_tensor), dim=-1)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0).astype(np.float32)

    def _build_index(self):
        all_imgs = []
        for fname in self.image_files:
            img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
            all_imgs.append(self.preprocess(img))
        features = self._extract_features(all_imgs)
        index = faiss.IndexFlatIP(features.shape[1])
        index.add(features)
        return index

    @torch.no_grad()
    def text_to_image(self, query: str, top_k: int = 5):
        token_ids = self.tokenizer(query).to(self.device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            text_feat = torch.nn.functional.normalize(
                self.model.encode_text(token_ids), dim=-1
            )
        distances, indices = self.index.search(text_feat.cpu().numpy(), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            results.append((img_path, f"Similarity: {dist:.3f}"))
        return results

    @torch.no_grad()
    def image_to_image(self, image: np.ndarray, top_k: int = 5):
        pil_img = Image.fromarray(image).convert("RGB")
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            img_feat = torch.nn.functional.normalize(
                self.model.encode_image(img_tensor), dim=-1
            )
        distances, indices = self.index.search(img_feat.cpu().numpy(), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            results.append((img_path, f"Similarity: {dist:.3f}"))
        return results

    @torch.no_grad()
    def zeroshot_classify(self, image: np.ndarray, class_names_str: str) -> dict:
        class_names = [c.strip() for c in class_names_str.split(",") if c.strip()]
        if not class_names:
            return {"label": "Error", "scores": {"No classes provided": 1.0}}

        pil_img = Image.fromarray(image).convert("RGB")
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        prompts = [f"a photo of a {c}." for c in class_names]
        token_ids = torch.cat([self.tokenizer(p) for p in prompts]).to(self.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            img_feat = torch.nn.functional.normalize(
                self.model.encode_image(img_tensor), dim=-1
            )
            text_feat = torch.nn.functional.normalize(
                self.model.encode_text(token_ids), dim=-1
            )
            sim = (img_feat @ text_feat.t()).squeeze(0)
            probs = torch.softmax(sim * self.model.logit_scale.exp(), dim=0)

        scores = {name: probs[i].item() for i, name in enumerate(class_names)}
        return {"label": max(scores, key=scores.get), "scores": scores}


def create_demo(checkpoint_path: str, image_dir: str):
    demo_engine = ClipDemo(checkpoint_path, image_dir)

    with gr.Blocks(title="CLIP Demo") as app:
        gr.Markdown("# CLIP 图文检索 & 零样本分类 Demo")

        with gr.Tab("文字搜图"):
            text_input = gr.Textbox(label="输入描述文字",
                                     placeholder="a cat sitting on a chair")
            top_k_slider = gr.Slider(1, 20, value=5, step=1, label="返回数量")
            text_btn = gr.Button("搜索")
            text_gallery = gr.Gallery(label="检索结果")
            text_btn.click(
                demo_engine.text_to_image,
                inputs=[text_input, top_k_slider],
                outputs=text_gallery,
            )

        with gr.Tab("以图搜图"):
            image_input = gr.Image(label="上传图片", type="numpy")
            img_k_slider = gr.Slider(1, 20, value=5, step=1, label="返回数量")
            img_btn = gr.Button("搜索")
            img_gallery = gr.Gallery(label="相似图片")
            img_btn.click(
                demo_engine.image_to_image,
                inputs=[image_input, img_k_slider],
                outputs=img_gallery,
            )

        with gr.Tab("零样本分类"):
            classify_img = gr.Image(label="上传图片", type="numpy")
            class_input = gr.Textbox(
                label="类别 (逗号分隔)", value="cat, dog, bird, car, tree"
            )
            classify_btn = gr.Button("分类")
            classify_output = gr.Label(label="分类结果")
            classify_btn.click(
                demo_engine.zeroshot_classify,
                inputs=[classify_img, class_input],
                outputs=classify_output,
            )

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = create_demo(args.checkpoint, args.image_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port)
```

- [ ] **Step 2: Commit**

```bash
git add src/demo/__init__.py src/demo/app.py
git commit -m "feat: add Gradio demo with FAISS retrieval and zero-shot classification"
```

---

## Phase 4: Technical Report

### Task 14: Report Template

**Files:**
- Create: `docs/report.md`

- [ ] **Step 1: 创建技术报告模板**

需要创建 `docs/report.md`，包含以下 6 个章节的完整模板，其中 `[待填入]` 占位符在训练完成后替换为实际数据。

**章节结构:**

```markdown
# CLIP 实战项目技术报告

> **日期**: 2026-05
> **代码仓库**: <GitHub/Gitee 链接>

## 1. 摘要
项目动机：从零实现 CLIP，掌握对比学习原理，构建可交互 Demo。
做了什么：手写 ViT-B/32 + Transformer → Flickr8k 训练 → open_clip COCO 训练 → Gradio Demo。
核心结果表格（训练后填入 Recall@K、Top-1 等数据）。

## 2. CLIP 模型详解 (2-3 页)
2.1 对比学习原理 — 与传统分类的区别、正负样本对构建
2.2 Dual Encoder 架构 — 架构图说明 Image Encoder / Text Encoder / Projection / Similarity Matrix
2.3 ViT-B/32 图像编码器 — Patch Embedding、CLS Token、Pre-Norm Transformer、为什么选 ViT
2.4 文本编码器 — BPE Tokenizer、6 层 Transformer、为什么自建不用 BERT
2.5 InfoNCE Loss 推导 — 完整公式 + 直觉理解
2.6 共享嵌入空间 — 维度选择 768、L2 归一化、温度参数 τ

## 3. 训练过程 (2-3 页)
3.1 数据预处理 — Flickr8k/COCO、BPE 训练、图像增强
3.2 训练配置 — Phase 1 vs Phase 2 配置对比表
3.3 Loss 曲线 — 训练/验证 loss 图
3.4 显存优化技术 — bf16 vs fp16、Gradient Checkpointing 原理、Gradient Accumulation

## 4. 评测与分析 (2-3 页)
4.1 图文检索 — Recall@1/5/10 表格，Phase 1 vs Phase 2 对比
4.2 零样本分类 — 混淆矩阵 + Top-1/Top-5
4.3 Bad Case 分析 — 检索/分类失败例子
4.4 Phase 1 vs Phase 2 全面对比 — 代码量、速度、灵活度、效果

## 5. Demo 展示 (1 页)
文字搜图、以图搜图、零样本分类截图 + FAISS 工程要点

## 6. 总结与反思 (半页)
关键收获、踩坑记录、改进方向
```

报告需覆盖以下面试追问：
- CLIP 的对比学习与传统分类训练的区别
- InfoNCE loss 的公式推导和直觉理解
- 为什么用 ViT 而不是 ResNet？Patch size 的影响？
- 共享嵌入空间维度如何选择？L2 归一化的作用？
- bf16 vs fp16 的区别？gradient checkpointing 的原理？
- Phase 1 手写和 Phase 2 open_clip 的结果差距？为什么？
- 零样本分类的 prompt engineering 怎么做的？

- [ ] **Step 2: Commit**

```bash
git add docs/report.md
git commit -m "feat: add technical report template with interview Q&A coverage"
```

---

### Task 15: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: 创建 README**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

## 实现顺序依赖

```
Task 1 (init) ──→ Task 2 (ViT) ──→ Task 4 (CLIP+Loss) ──→ Task 6 (utils) ──→ Task 7 (trainer)
                Task 3 (Text Enc) ──┘  ↑                      ↑
                Task 5 (data) ────────┘                      │
                                                              │
Task 7 (trainer) ──→ Task 10 (train script) ──→ 训练验证
                                                      │
Task 8 (retrieval) ──→ Task 11 (eval script) ←───────┘
Task 9 (zeroshot)  ──┘

                      Task 12 (Phase 2 config+script)

                      Task 13 (Phase 3 demo)

                      Task 14 (Phase 4 report)
                      Task 15 (README)
```

Tasks 2, 3, 5 可并行开发（无互相依赖）。
Tasks 8, 9 可并行。
Phase 2, 3, 4 依赖 Phase 1 训练完成后的 checkpoint。
