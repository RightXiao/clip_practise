"""Text Encoder for CLIP: 6-layer Transformer with learned position embeddings."""
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
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
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
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for block in self.blocks:
            for p in block.parameters():
                if p.dim() > 1:
                    nn.init.trunc_normal_(p, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(token_ids)                  # (B, L, 512)
        x = x + self.pos_embed[:, :x.size(1), :]

        for block in self.blocks:
            if self.use_grad_ckpt and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)
        eos_pos = (token_ids > 0).sum(dim=-1) - 1
        eos_pos = eos_pos.clamp(min=0)
        eos_out = x[torch.arange(x.size(0)), eos_pos]    # (B, 512)
        return self.proj(eos_out)                        # (B, proj_dim)
