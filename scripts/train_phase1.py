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
