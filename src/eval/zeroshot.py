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
