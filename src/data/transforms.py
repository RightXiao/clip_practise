"""Image transforms for CLIP training."""
import torchvision.transforms as T


def get_train_transforms(img_size: int = 224):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                     std=(0.26862954, 0.26130258, 0.27577711)),
    ])


def get_val_transforms(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                     std=(0.26862954, 0.26130258, 0.27577711)),
    ])
