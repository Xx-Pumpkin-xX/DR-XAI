"""
pipeline.py
===========
The parts of the training protocol that must be IDENTICAL across every
backbone, collected in one place so that a comparison table cannot silently
compare two different pipelines.

Shared by `train_baseline.py`, `evaluate_model.py` and `compare_models.py`:
  * CLAHE + black-border-crop preprocessing (from preprocessing.py)
  * augmentation policy (train) and resize-only policy (valid)
  * the 3-view TTA set: original + horizontal flip + vertical flip
  * SmoothL1 ordinal-regression train / validate loops with AMP
  * OptimizedRounder threshold fitting

Anything that legitimately differs per architecture (encoder, input size,
batch size, learning rate) is an argument, not a hard-coded constant.
"""

import os
import contextlib

import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from preprocessing import crop_image_from_gray, apply_clahe

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==========================================================================
# LOGGING
# ==========================================================================
def print_step(step_name):
    print(f"\n{'-' * 78}\n STEP: {step_name}\n{'-' * 78}")

def print_success(msg):
    print(f"   [SUCCESS] {msg}")

def print_warning(msg):
    print(f"   [WARNING] {msg}")

def print_error_and_exit(msg, error=""):
    import sys
    print(f"   [FAILED] {msg}\n\n SYSTEM ERROR DETAILS:\n{error}")
    sys.exit(1)


# ==========================================================================
# AMP (torch>=2.4 API with a fallback for older builds)
# ==========================================================================
def autocast(device):
    if device != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def make_scaler(device):
    if device != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


# ==========================================================================
# TRANSFORMS
# ==========================================================================
def get_train_transforms(img_size):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_valid_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_tta_transforms(img_size):
    """The 3-view TTA set used everywhere: original, H-flip, V-flip."""
    base = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return [
        A.Compose([A.Resize(height=img_size, width=img_size)] + base),
        A.Compose([A.Resize(height=img_size, width=img_size), A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.Resize(height=img_size, width=img_size), A.VerticalFlip(p=1.0)] + base),
    ]


# ==========================================================================
# DATASETS
# ==========================================================================
class FundusDataset(Dataset):
    """
    One dataset class for APTOS, EyePACS and MESSIDOR.

    `df` must expose an image column, a label column, and either an image
    directory + extension, or a fully resolved `image_path` column.
    """

    def __init__(self, df, img_dir=None, transforms=None,
                 id_col="id_code", label_col="diagnosis", ext=".png",
                 path_col=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.id_col = id_col
        self.label_col = label_col
        self.ext = ext
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, row):
        if self.path_col and self.path_col in row and isinstance(row[self.path_col], str):
            return row[self.path_col]
        return os.path.join(self.img_dir, f"{row[self.id_col]}{self.ext}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row)
        label = float(row[self.label_col])

        image = cv2.imread(img_path)
        if image is None:
            # A corrupt file must not kill a multi-hour run, but it must be loud.
            print_warning(f"Unreadable image, substituting black frame: {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = crop_image_from_gray(image)
            image = apply_clahe(image)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.float32)


DATASET_SPECS = {
    "aptos":    dict(id_col="id_code", label_col="diagnosis", ext=".png"),
    "eyepacs":  dict(id_col="image",   label_col="level",     ext=".jpeg"),
    "messidor": dict(id_col="image_id", label_col="label",    ext=".png", path_col="image_path"),
}


def make_dataset(df, dataset_name, img_dir, transforms):
    spec = DATASET_SPECS[dataset_name]
    return FundusDataset(df, img_dir=img_dir, transforms=transforms, **spec)


# ==========================================================================
# TRAIN / VALIDATE
# ==========================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device,
                    accum_steps=1, max_grad_norm=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc="[Train]", leave=False)
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with autocast(device):
            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            if max_grad_norm:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        real_loss = loss.item() * accum_steps
        total_loss += real_loss
        pbar.set_postfix(loss=f"{real_loss:.4f}")

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def predict_with_tta(model, df, dataset_name, img_dir, img_size, device,
                     batch_size=16, num_workers=4, use_tta=True, desc="Valid"):
    """
    Mean prediction over the 3 TTA views (or a single view when use_tta=False).

    Returns (mean_preds, targets, per_view_preds).
    """
    model.eval()

    transforms_list = get_tta_transforms(img_size) if use_tta else [get_valid_transforms(img_size)]

    per_view = []
    targets = []
    targets_collected = False

    for view_idx, transform in enumerate(transforms_list):
        dataset = make_dataset(df, dataset_name, img_dir, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device == "cuda"))

        preds = []
        pbar = tqdm(loader, desc=f"[{desc} TTA {view_idx + 1}/{len(transforms_list)}]", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            with autocast(device):
                outputs = model(images)
            preds.extend(outputs.float().cpu().numpy())
            if not targets_collected:
                targets.extend(labels.numpy())

        per_view.append(np.array(preds).flatten())
        targets_collected = True

    mean_preds = np.mean(per_view, axis=0)
    return mean_preds, np.array(targets).flatten(), per_view


# ==========================================================================
# CHECKPOINT SIDE-CAR
# ==========================================================================
def save_run_config(save_dir, config: dict):
    """Persist the exact settings next to the weights; compare_models reads it."""
    import json
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
