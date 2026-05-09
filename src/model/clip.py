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
