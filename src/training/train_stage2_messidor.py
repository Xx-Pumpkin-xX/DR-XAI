import sys, os
core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "0_Core")
if core_path not in sys.path: sys.path.append(core_path)

import os
import sys
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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
    print(f"\n{'-'*80}\n STEP: {step_name}\n{'-'*80}")

def print_success(msg):
    print(f"   [SUCCESS] {msg}")

def print_error_and_exit(msg, error):
    print(f"   [FAILED] {msg}\n\n SYSTEM ERROR DETAILS:\n{error}")
    sys.exit(1)

# ==========================================
# DATA PARSING & PREPARATION
# ==========================================
def map_messidor_label(dr_str):
    dr_str = str(dr_str).strip().upper()
    for i in range(5):
        if str(i) in dr_str: return float(i)
    return 0.0

def get_exact_image_path(base_name, search_dir):
    for root, dirs, files in os.walk(search_dir):
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            if base_name + ext in files:
                return os.path.join(root, base_name + ext)
    return None

def build_messidor_dataframe(maples_dir, img_dir):
    csv_files = glob.glob(os.path.join(maples_dir, "**", "diagnosis.csv"), recursive=True)
    data = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            base_name = str(row['name'])
            grade = map_messidor_label(row['DR'])
            img_path = get_exact_image_path(base_name, img_dir)
            if img_path:
                data.append({'image_id': base_name, 'image_path': img_path, 'label': grade})
    return pd.DataFrame(data)

# ==========================================
# DATASET & AUGMENTATION
# ==========================================
class MessidorDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label']

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

def get_valid_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_tta_transforms(img_size):
    return [
        A.Compose([A.Resize(height=img_size, width=img_size), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(height=img_size, width=img_size), A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
    ]

# ==========================================
# TRAINING & TTA VALIDATION
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
def validate_with_tta(model, df_valid, img_size):
    model.eval()
    transforms_list = [get_valid_transforms(img_size)] + get_tta_transforms(img_size)
    all_tta_preds = []
    all_targets = []
    targets_collected = False 

    for tta_idx, transform in enumerate(transforms_list):
        dataset = MessidorDataset(df_valid, transforms=transform)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
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
        targets_collected = True 

    mean_preds = np.mean(all_tta_preds, axis=0)
    all_targets = np.array(all_targets).flatten()
    return mean_preds, all_targets

# ==========================================
# MAIN COMMAND LINE PARSER
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Fine-Tuning on MESSIDOR")
    parser.add_argument('--model', type=str, default='B7')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30) 
    parser.add_argument('--use_cbam', action='store_true')
    parser.add_argument('--pretrain_path', type=str, required=True, help="Path to stage1_best.pth")
    return parser.parse_args()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    args = parse_args()
    args.model = args.model.upper()
    img_size = EFF_SIZES[args.model]
    
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    MAPLES_DIR = os.path.join(THIS_DIR, "../../MAPLES-DR")
    MESSIDOR_IMG_DIR = os.path.join(THIS_DIR, "../../datasets1")

    # [NEW] DYNAMIC FOLDER NAMING TO PREVENT OVERWRITING APTOS RESULTS
    exp_name = f"Stage2_Finetune_MESSIDOR_{args.model}_Batch{args.batch_size}"
    if args.use_cbam: exp_name += "_CBAM"
    SAVE_DIR = os.path.join(THIS_DIR, "../experiments", exp_name)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print_step(f"STAGE 2: FINE-TUNING ON MESSIDOR (REGRESSION + TTA)")
    print(f"[INFO] Experiment Output: {exp_name}")

    # 1. Prepare Data
    df_all = build_messidor_dataframe(MAPLES_DIR, MESSIDOR_IMG_DIR)
    if len(df_all) == 0:
        print_error_and_exit("No images matched between CSV and datasets1!", "")

    # 80/20 Split: Train on 158, Validate on 40 to avoid data leakage for XAI
    df_train, df_valid = train_test_split(df_all, test_size=0.2, stratify=df_all['label'], random_state=42)
    print_success(f"Data parsed. Train: {len(df_train)} | Valid (Unseen): {len(df_valid)}")
    
    # Save the split lists so XAI benchmark knows which 40 images to test later
    df_train.to_csv(os.path.join(SAVE_DIR, "messidor_train_split.csv"), index=False)
    df_valid.to_csv(os.path.join(SAVE_DIR, "messidor_valid_split.csv"), index=False)

    train_dataset = MessidorDataset(df_train, transforms=get_train_transforms(img_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 2. Initialize Model
    model_name_timm = f"tf_efficientnet_{args.model.lower()}_ns"
    model = APTOSModel(model_name=model_name_timm, num_classes=1, use_cbam=args.use_cbam).cuda()
    
    if os.path.exists(args.pretrain_path):
        model.load_state_dict(torch.load(args.pretrain_path))
        print_success(f"Loaded Stage 1 weights from: {args.pretrain_path}")
    else:
        print_error_and_exit("Cannot find pre-trained weights.", args.pretrain_path)

    # 3. Fine-tuning Setup
    criterion = nn.SmoothL1Loss() 
    # Use small learning rate to preserve Stage 1 knowledge
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler()
    optR = OptimizedRounder()
    best_qwk = -1.0

    # 4. Training Loop
    for epoch in range(args.epochs):
        print(f"\n[EPOCH {epoch+1}/{args.epochs}]")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, scaler)
        
        mean_preds, all_targets = validate_with_tta(model, df_valid, img_size)

        optR.fit(mean_preds, all_targets)
        rounded_preds = optR.predict(mean_preds, optR.coef_['x'])
        val_qwk = calculate_qwk(all_targets, rounded_preds)

        print(f"   [RESULT] Train Loss: {train_loss:.4f} | Val QWK: {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "stage2_best_model.pth"))
            np.save(os.path.join(SAVE_DIR, "best_thresholds.npy"), optR.coef_['x'])
            print(f"   [SUCCESS] New best model saved! (QWK: {best_qwk:.4f})")

    print(f"\n[INFO] Stage 2 Completed. Final Best QWK on Unseen Valid: {best_qwk:.4f}")

if __name__ == "__main__":
    main()