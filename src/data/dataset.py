"""Flickr8k dataset for CLIP training."""
import os
import tempfile
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def _train_bpe_tokenizer(captions: list[str], vocab_size: int = 10000) -> Tokenizer:
    """Train a BPE tokenizer on the captions."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
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
            pad_id = self.tokenizer.token_to_id("[PAD]")
            ids = ids + [pad_id] * (self.max_len - len(ids))
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
    from .transforms import get_train_transforms, get_val_transforms

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
