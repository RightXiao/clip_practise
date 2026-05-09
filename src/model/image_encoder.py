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
