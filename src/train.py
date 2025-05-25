import argparse
import os

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import (
    TensorBoardLogger,
)
from torch.utils.data import DataLoader

from dataset import RainSnowDataset
from model import PromptIR
from schedulers import LinearWarmupCosineAnnealingLR


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=False)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        degraded_img, clean_img = batch
        restored = self.net(degraded_img)
        loss = self.loss_fn(restored, clean_img)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        degraded_img, clean_img = batch
        restored = self.net(degraded_img)
        loss = self.loss_fn(restored, clean_img)
        self.log("val_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=10, max_epochs=300
        )
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", type=int, default=0)

    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    parser.add_argument(
        "--root_dir", type=str, default="../data", help="root directory"
    )

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Dataset paths
    all_dataset = RainSnowDataset(root_dir=args.root_dir, is_train=True, patch_size=128)

    # Split validation set from training set
    train_size = int(0.9 * len(all_dataset))
    val_size = len(all_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        all_dataset, [train_size, val_size]
    )

    # Create dataloaders with smaller batch size and fewer workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    torch.cuda.set_device(args.cuda)

    # Initialize model and trainer
    model = PromptIRModel()

    # Setup logging
    logger = TensorBoardLogger(save_dir="logs/", name="PromptIR")

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="promptir-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Initialize trainer with memory optimizations and resume support
    trainer = pl.Trainer(
        max_epochs=300,
        accelerator="gpu",
        devices=[args.cuda],
        logger=logger,
        callbacks=[checkpoint_callback],
        val_check_interval=0.5,  # Validate every 0.5 epochs
        gradient_clip_val=1.0,  # Add gradient clipping
        accumulate_grad_batches=4,  # Accumulate gradients to simulate larger batch size
        precision=16,  # Use mixed precision training
    )

    # Train the model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_ckpt,
    )


if __name__ == "__main__":
    main()
