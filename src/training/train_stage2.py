import sys, os
core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "0_Core")
if core_path not in sys.path: sys.path.append(core_path)

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import APTOSModel, calculate_qwk
from preprocessing import crop_image_from_gray, apply_clahe
from rounder import OptimizedRounder

EFF_SIZES = {'B0': 224, 'B1': 240, 'B2': 260, 'B3': 300, 'B4': 380, 'B5': 456, 'B6': 528, 'B7': 600}

# ==========================================
# SYSTEM LOGGER
# ==========================================
def print_step(step_name):
    print(f"\n{'-'*60}\n STEP: {step_name}\n{'-'*60}")

def print_success(msg):
    print(f"   [SUCCESS] {msg}")

def print_error_and_exit(msg, error):
    print(f"   [FAILED] {msg}\n\n SYSTEM ERROR DETAILS:\n{error}")
    sys.exit(1)

# ==========================================
# DATASET & AUGMENTATION (REGRESSION MODE)
# ==========================================
class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_name}.png")
        label = float(self.df.iloc[idx]['diagnosis']) # Float for regression

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = apply_clahe(image)

        if self.transforms:
            image = self.transforms(image=image)['image']
        return image, torch.tensor(label, dtype=torch.float32)

def get_train_transforms(img_size):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# Base transform for original image
def get_valid_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# TTA transforms (Horizontal and Vertical Flips)
def get_tta_transforms(img_size):
    return [
        A.Compose([A.Resize(height=img_size, width=img_size), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(height=img_size, width=img_size), A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
    ]

# ==========================================
# TRAINING & TTA VALIDATION PIPELINE
# ==========================================
def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="[Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.cuda(), labels.cuda().unsqueeze(1)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    scheduler.step()
    return total_loss / len(dataloader)

@torch.no_grad()
def validate_with_tta(model, df_valid, img_dir, img_size):
    """Validation process with Test-Time Augmentation (Original + H-Flip + V-Flip)"""
    model.eval()
    
    transforms_list = [get_valid_transforms(img_size)] + get_tta_transforms(img_size)
    all_tta_preds = []
    all_targets = []
    targets_collected = False

    # Run inference for each TTA variation
    for tta_idx, transform in enumerate(transforms_list):
        dataset = APTOSDataset(df_valid, img_dir, transforms=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
        
        preds = []
        
        pbar = tqdm(dataloader, desc=f"[Valid TTA {tta_idx+1}/3]", leave=False)
        for images, labels in pbar:
            images = images.cuda()
            with torch.cuda.amp.autocast():
                outputs = model(images)
            preds.extend(outputs.float().cpu().numpy())
            
            if not targets_collected:
                all_targets.extend(labels.numpy())
                
        all_tta_preds.append(np.array(preds).flatten())
        targets_collected = True # Only collect targets once

    # Average predictions across all TTA variations
    mean_preds = np.mean(all_tta_preds, axis=0)
    all_targets = np.array(all_targets).flatten()
    
    return mean_preds, all_targets

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Stage 2: Fine-Tuning on APTOS 2019")
    parser.add_argument('--model', type=str, default='B7')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20) 
    parser.add_argument('--use_cbam', action='store_true')
    parser.add_argument('--pretrain_path', type=str, required=True, help="Path to stage1_best.pth")
    args = parser.parse_args()

    args.model = args.model.upper()
    img_size = EFF_SIZES[args.model]
    
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(THIS_DIR, "../data/train_folds.csv")
    IMG_DIR = os.path.join(THIS_DIR, "../../aptos2019-blindness-detection/train_images/")
    FOLD_TO_TRAIN = 0

    # Save directory
    exp_name = f"Stage2_Finetune_{args.model}"
    if args.use_cbam: exp_name += "_CBAM"
    SAVE_DIR = os.path.join(THIS_DIR, "../experiments", exp_name)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("="*70)
    print(f" STAGE 2: FINE-TUNING ON APTOS (REGRESSION + TTA) ")
    print(f" Experiment: {exp_name} | Target Fold: {FOLD_TO_TRAIN} ")
    print("="*70)

    # 1. Prepare Data
    df = pd.read_csv(CSV_PATH)
    df_train = df[df['fold'] != FOLD_TO_TRAIN]
    df_valid = df[df['fold'] == FOLD_TO_TRAIN]
    print(f"[INFO] Train: {len(df_train)} | Valid: {len(df_valid)}")

    train_dataset = APTOSDataset(df_train, IMG_DIR, transforms=get_train_transforms(img_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 2. Initialize Model and Load Stage 1 Weights
    model_name_timm = f"tf_efficientnet_{args.model.lower()}_ns"
    model = APTOSModel(model_name=model_name_timm, num_classes=1, use_cbam=args.use_cbam).cuda()
    
    print(f"[INFO] Loading Stage 1 pre-trained weights from: {args.pretrain_path}")
    if os.path.exists(args.pretrain_path):
        model.load_state_dict(torch.load(args.pretrain_path))
        print("   [SUCCESS] Weights loaded successfully!")
    else:
        print(f"   [ERROR] Cannot find pre-trained weights. Exiting.")
        return

    # 3. Fine-tuning Setup (Lower Learning Rate)
    criterion = nn.SmoothL1Loss() 
    # Notice the lower LR: 2e-5 instead of 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler()
    optR = OptimizedRounder()
    best_qwk = -1.0

    # 4. Training Loop
    for epoch in range(args.epochs):
        print(f"\n[EPOCH {epoch+1}/{args.epochs}]")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, scaler)
        
        # Validation with TTA
        print("   [INFO] Validating with Test-Time Augmentation (3 versions)...")
        mean_preds, all_targets = validate_with_tta(model, df_valid, IMG_DIR, img_size)

        # Optimize Thresholds
        optR.fit(mean_preds, all_targets)
        rounded_preds = optR.predict(mean_preds, optR.coef_['x'])
        val_qwk = calculate_qwk(all_targets, rounded_preds)

        print(f"   [RESULT] Train Loss: {train_loss:.4f} | Val QWK: {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "stage2_best_model.pth"))
            np.save(os.path.join(SAVE_DIR, "best_thresholds.npy"), optR.coef_['x'])
            print(f"   [SUCCESS] New best model saved! (QWK: {best_qwk:.4f})")

    print(f"\n[INFO] Stage 2 Completed. Final Best QWK: {best_qwk:.4f}")

if __name__ == "__main__":
    main()