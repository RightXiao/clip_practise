"""Unified evaluation: retrieval + zero-shot classification."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.model.clip import CLIP
from src.data.transforms import get_val_transforms
from src.eval.retrieval import evaluate_retrieval
from src.eval.zeroshot import zeroshot_classify, zeroshot_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/phase1_scratch.yaml")
    parser.add_argument("--task", type=str, default="all",
                        choices=["retrieval", "zeroshot", "all"])
    parser.add_argument("--zeroshot-data", type=str, default=None,
                        help="Path to class-labeled dataset for zero-shot eval "
                             "(subdirs named by class)")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    seed = config.train.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        if args.zeroshot_data is None:
            print("Zero-shot evaluation requires a class-labeled dataset.")
            print("Use --zeroshot-data <path> with subdirectories named by class.")
        else:
            dataset = ImageFolder(args.zeroshot_data,
                                  transform=get_val_transforms(config.data.img_size))
            loader = DataLoader(dataset, batch_size=config.train.batch_size,
                                shuffle=False, num_workers=config.train.get("num_workers", 4),
                                pin_memory=True)
            class_names = dataset.classes

            max_len = config.data.max_text_len
            pad_id = tokenizer.token_to_id("[PAD]")
            text_prompts = [f"a photo of a {name}" for name in class_names]
            text_list = []
            for p in text_prompts:
                ids = tokenizer.encode(f"[BOS] {p} [EOS]").ids[:max_len]
                if len(ids) < max_len:
                    ids = ids + [pad_id] * (max_len - len(ids))
                text_list.append(ids)
            text_ids = torch.tensor(text_list, dtype=torch.long).to(device)

            all_preds, all_labels = [], []
            for images, labels in loader:
                images = images.to(device)
                img_embeds = model.encode_image(images)
                txt_embeds = model.encode_text(text_ids)
                preds = zeroshot_classify(img_embeds, txt_embeds)
                all_preds.append(preds.cpu())
                all_labels.append(labels)

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            metrics = zeroshot_metrics(all_preds, all_labels, len(class_names))
            print("Zero-shot Results:")
            print(f"  Top-1 Accuracy: {metrics['top1_acc']:.4f}")
            print(f"  Classes: {class_names}")


if __name__ == "__main__":
    main()
