"""Phase 2: Train CLIP using open_clip on COCO subset."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import random
import numpy as np
import torch
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    seed = config.train.get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

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
