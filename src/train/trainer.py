"""Training loop for CLIP."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import AverageMeter, save_checkpoint, create_optimizer, cosine_schedule


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
    use_bf16: bool = True,
) -> float:
    """Train one epoch. Returns average loss."""
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    dtype = torch.bfloat16 if use_bf16 else torch.float16
    pbar = tqdm(loader, desc="Training")

    for batch_idx, (images, token_ids) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=dtype):
            loss, logit_scale = model(images, token_ids)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * grad_accum_steps, images.size(0))
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            "scale": f"{logit_scale:.2f}",
        })

    return loss_meter.avg


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Run validation. Returns average loss."""
    model.eval()
    loss_meter = AverageMeter()

    for images, token_ids in tqdm(loader, desc="Validating"):
        images = images.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, _ = model(images, token_ids)

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> float:
    """Full training loop with checkpointing. Returns best val loss."""
    train_cfg = config["train"]
    output_cfg = config["output"]

    total_steps = len(train_loader) * train_cfg["epochs"]
    optimizer = create_optimizer(model, train_cfg["lr"], train_cfg["weight_decay"])
    scheduler = cosine_schedule(optimizer, train_cfg["warmup_steps"], total_steps, train_cfg["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg["precision"] == "fp16"))

    use_bf16 = train_cfg["precision"] == "bf16"
    best_val_loss = float("inf")
    os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, grad_accum_steps=train_cfg["grad_accum_steps"], use_bf16=use_bf16,
        )
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                epoch * len(train_loader), val_loss,
                os.path.join(output_cfg["checkpoint_dir"], "best.pt"),
            )

    return best_val_loss
