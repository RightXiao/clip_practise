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
