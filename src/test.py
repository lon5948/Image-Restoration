import argparse
import os

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RainSnowDataset
from model import PromptIR


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)


def save_visualization(restored_np, output_path, idx):
    """Save visualization image"""
    # Convert from (3, H, W) to (H, W, 3) for PIL
    img = np.transpose(restored_np, (1, 2, 0))
    img = Image.fromarray(img)
    img.save(os.path.join(output_path, f"visualization_{idx:04d}.png"))


def test_model(net, dataset, output_path):
    """Test the model on the given dataset and save results"""
    os.makedirs(output_path, exist_ok=True)

    testloader = DataLoader(
        dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2
    )

    # Dictionary to store predictions
    predictions = {}
    num_visualizations = 5  # Number of images to save as visualization

    with torch.no_grad():
        for i, degraded_img in enumerate(tqdm(testloader)):
            degraded_img = degraded_img.cuda()

            # Process image
            restored = net(degraded_img)

            # Convert to numpy array and ensure correct format
            restored_np = restored.squeeze(0).cpu().numpy()  # Remove batch dimension
            restored_np = np.clip(restored_np, 0, 1)  # Clip to [0, 1]
            restored_np = (restored_np * 255).astype(
                np.uint8
            )  # Scale to 0-255 and convert to uint8

            # Store in predictions dictionary
            predictions[f"{i}.png"] = restored_np

            # Save visualization for first few images
            if i < num_visualizations:
                save_visualization(restored_np, output_path, i)

    # Save predictions to .npz file
    np.savez("pred.npz", **predictions)
    print(f"Saved predictions for {len(predictions)} images to pred.npz")
    print(f"Saved {num_visualizations} visualization images to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--root_dir", type=str, default="../data", help="root directory"
    )
    parser.add_argument(
        "--output_path", type=str, default="output/", help="output save path"
    )
    parser.add_argument("--seed", type=int, default=47, help="random seed")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="checkpoint path",
    )
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.cuda)

    # Create test dataset
    test_dataset = RainSnowDataset(root_dir=args.root_dir, is_train=False)

    # Load model
    print(f"Loading checkpoint: {args.ckpt_path}")
    model = PromptIRModel.load_from_checkpoint(args.ckpt_path)
    model = model.cuda()
    model.eval()

    # Run testing
    print("Starting testing...")
    test_model(model, test_dataset, args.output_path)
