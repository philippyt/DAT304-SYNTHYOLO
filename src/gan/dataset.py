import os
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class PairedFloorPlanDataset(Dataset):
    def __init__(self, empty_dir, symbols_dir, image_size=(640, 640), augment=True):
        self.image_size = image_size
        self.augment = augment

        self.empty_images = sorted([os.path.join(empty_dir, f) for f in os.listdir(empty_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.symbols_images = sorted([os.path.join(symbols_dir, f) for f in os.listdir(symbols_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        common_files = set(map(os.path.basename, self.empty_images)) & set(map(os.path.basename, self.symbols_images))
        self.empty_images = [p for p in self.empty_images if os.path.basename(p) in common_files]
        self.symbols_images = [p for p in self.symbols_images if os.path.basename(p) in common_files]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.empty_images)

    def apply_augmentation(self, empty_img, symbols_img):
        if random.random() > 0.5:
            empty_img = cv2.flip(empty_img, 1)
            symbols_img = cv2.flip(symbols_img, 1)
        if random.random() > 0.5:
            empty_img = cv2.flip(empty_img, 0)
            symbols_img = cv2.flip(symbols_img, 0)
        k_rot = random.randint(0, 3)
        for _ in range(k_rot):
            empty_img = cv2.transpose(empty_img)
            empty_img = cv2.flip(empty_img, 1)
            symbols_img = cv2.transpose(symbols_img)
            symbols_img = cv2.flip(symbols_img, 1)
        return empty_img, symbols_img

    def __getitem__(self, idx):
        empty_img = cv2.imread(self.empty_images[idx], cv2.IMREAD_GRAYSCALE)
        symbols_img = cv2.imread(self.symbols_images[idx], cv2.IMREAD_GRAYSCALE)
        empty_img = cv2.resize(empty_img, self.image_size)
        symbols_img = cv2.resize(symbols_img, self.image_size)
        if self.augment:
            empty_img, symbols_img = self.apply_augmentation(empty_img, symbols_img)
        empty_tensor = self.transform(empty_img)
        symbols_tensor = self.transform(symbols_img)
        return empty_tensor, symbols_tensor