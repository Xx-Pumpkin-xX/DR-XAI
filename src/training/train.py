import sys, os
core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "0_Core")
if core_path not in sys.path: sys.path.append(core_path)

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# --- SYSTEM LOGGER ---
def print_step(step_name):
    print(f"\n{'-'*60}\n STEP: {step_name}\n{'-'*60}")

def print_success(msg):
    print(f"   [SUCCESS] {msg}")

def print_error_and_exit(msg, error):
    print(f"   [FAILED] {msg}\n\n SYSTEM ERROR DETAILS:\n{error}")
    sys.exit(1)

# --- EFFICIENTNET STANDARD SIZES ---
EFF_SIZES = {
    'B0': 224, 'B1': 240, 'B2': 260, 'B3': 300, 
    'B4': 380, 'B5': 456, 'B6': 528, 'B7': 600
}

# ==========================================
# COMMAND LINE ARGUMENTS PARSER
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="APTOS Blindness Detection Training Pipeline")
    parser.add_argument('--model', type=str, default='B0', help='Model version (e.g., B0, B4, B6)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--use_cbam', action='store_true', help='Enable CBAM Attention module')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps (Default 1: Normal Training)')
    
    args = parser.parse_args()
    
    args.model = args.model.upper()
    if args.model not in EFF_SIZES:
        raise ValueError(f"Model {args.model} is not supported!")
        
    return args

# ==========================================
# IMPORT AND SETUP CONFIGURATIONS
# ==========================================
try:
    from augmentation import get_train_transforms, get_valid_transforms
    from model import APTOSModel, FocalLoss, get_class_weights, create_balanced_sampler, calculate_qwk
    from evaluation import evaluate_baseline
    from preprocessing import crop_image_from_gray, apply_clahe
    
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(THIS_DIR, "../data/train_folds.csv")
    IMG_DIR = os.path.join(THIS_DIR, "../../aptos2019-blindness-detection/train_images/")
    FOLD_TO_TRAIN = 0
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    print(f"Initialization Error: {e}")
    sys.exit(1)

# ==========================================
# DATASET & TRAIN/VAL FUNCTIONS
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
        label = self.df.iloc[idx]['diagnosis']

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = apply_clahe(image)

        if self.transforms:
            image = self.transforms(image=image)['image']
        return image, torch.tensor(label, dtype=torch.long)

def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, accum_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad() # Xóa rác gradient ở đầu mỗi epoch
    
    pbar = tqdm(dataloader, desc="[Train]", leave=False)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels) / accum_steps
        loss.backward()
        
        # Cập nhật khi đủ số bước, hoặc khi kết thúc Dataloader
        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        real_loss = loss.item() * accum_steps
        total_loss += real_loss
        pbar.set_postfix(loss=f"{real_loss:.4f}")
        
    scheduler.step()
    return total_loss / len(dataloader)

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_probs = [], [], []
    
    pbar = tqdm(dataloader, desc="[Valid]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_targets.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
    val_loss = total_loss / len(dataloader)
    val_qwk = calculate_qwk(all_targets, all_preds)
    return val_loss, val_qwk, all_targets, all_preds, all_probs

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    args = parse_args()
    img_size = EFF_SIZES[args.model]
    model_name_timm = f"tf_efficientnet_{args.model.lower()}_ns"

    # Đặt tên folder tự động
    folder_name = f"Eff_{args.model}_Batch{args.batch_size}"
    if args.use_cbam: folder_name += "_CBAM"
    if args.accum_steps > 1: folder_name += f"_Accum{args.accum_steps}"
    
    SAVE_DIR = os.path.join(THIS_DIR, "../experiments", folder_name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"\n========================================================")
    print(f" STARTING EXPERIMENT: {folder_name}")
    print(f"========================================================")

    df = pd.read_csv(CSV_PATH)
    df_train = df[df['fold'] != FOLD_TO_TRAIN]
    df_valid = df[df['fold'] == FOLD_TO_TRAIN]
    
    train_dataset = APTOSDataset(df_train, IMG_DIR, transforms=get_train_transforms(img_size))
    valid_dataset = APTOSDataset(df_valid, IMG_DIR, transforms=get_valid_transforms(img_size))
    
    sampler = create_balanced_sampler(df_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = APTOSModel(model_name=model_name_timm, use_cbam=args.use_cbam).to(DEVICE)
    weights = get_class_weights(df_train).to(DEVICE)
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_qwk = -1.0
    best_val_data = None
    
    for epoch in range(args.epochs):
        print(f"\n EPOCH {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, args.accum_steps)
        val_loss, val_qwk, targets, preds, probs = validate_one_epoch(model, valid_loader, criterion)
        
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f}")
        
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_val_data = (targets, preds, probs)
            save_path = os.path.join(SAVE_DIR, f"best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   [!] Best model saved (QWK: {best_qwk:.4f})")

    print("\n GENERATING REPORT & CHARTS...")
    evaluate_baseline(best_val_data[0], best_val_data[1], best_val_data[2], save_dir=SAVE_DIR)
    print(f" EXPERIMENT COMPLETED! All results saved in: {SAVE_DIR}/")

if __name__ == "__main__":
    main()