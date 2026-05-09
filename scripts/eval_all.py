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
