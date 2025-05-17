import os
import random

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class RainSnowDataset(Dataset):
    def __init__(self, root_dir, is_train=True, patch_size=128):
        self.root_dir = root_dir
        self.is_train = is_train
        self.patch_size = patch_size
        self.image_pairs = []

        if is_train:
            self.degraded_dir = os.path.join(root_dir, "train", "degraded")
            self.clean_dir = os.path.join(root_dir, "train", "clean")
            self.degradation_types = ["rain", "snow"]

            # Collect all image pairs
            for deg_type in self.degradation_types:
                for i in range(1, 1601):  # 1600 images per degradation type
                    degraded_path = os.path.join(
                        self.degraded_dir, f"{deg_type}-{i}.png"
                    )
                    clean_path = os.path.join(
                        self.clean_dir, f"{deg_type}_clean-{i}.png"
                    )
                    if os.path.exists(degraded_path) and os.path.exists(clean_path):
                        self.image_pairs.append((degraded_path, clean_path, deg_type))

            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            self.degraded_dir = os.path.join(root_dir, "test", "degraded")
            for i in range(100):  # 100 test images
                degraded_path = os.path.join(self.degraded_dir, f"{i}.png")
                if os.path.exists(degraded_path):
                    self.image_pairs.append((degraded_path, None, None))

            # Validation / test: deterministic transform
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        degraded_path, clean_path, deg_type = self.image_pairs[idx]

        # Load degraded image
        degraded_img = Image.open(degraded_path).convert("RGB")

        if self.is_train:
            # Load clean image for training
            clean_img = Image.open(clean_path).convert("RGB")

            # Random crop for training
            w, h = degraded_img.size
            if w < self.patch_size or h < self.patch_size:
                degraded_img = degraded_img.resize((self.patch_size, self.patch_size))
                clean_img = clean_img.resize((self.patch_size, self.patch_size))
            else:
                i = random.randint(0, h - self.patch_size)
                j = random.randint(0, w - self.patch_size)
                degraded_img = degraded_img.crop(
                    (j, i, j + self.patch_size, i + self.patch_size)
                )
                clean_img = clean_img.crop(
                    (j, i, j + self.patch_size, i + self.patch_size)
                )

            # Apply transforms
            degraded_img = self.transform(degraded_img)
            clean_img = self.transform(clean_img)

            return degraded_img, clean_img
        else:
            # For test set, just return the degraded image
            degraded_img = self.transform(degraded_img)
            return degraded_img
