import sys, os
core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "0_Core")
if core_path not in sys.path: sys.path.append(core_path)

import os
import sys
import glob
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import APTOSModel, calculate_qwk
from preprocessing import crop_image_from_gray, apply_clahe
from rounder import OptimizedRounder

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
# CONFIGURATIONS
# ==========================================
EFF_SIZES = {'B0': 224, 'B1': 240, 'B2': 260, 'B3': 300, 'B4': 380, 'B5': 456, 'B6': 528, 'B7': 600}
LESION_TYPES = ["Microaneurysms", "Hemorrhages", "Exudates", "CottonWoolSpots"]

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

def build_maples_train_dataframe(maples_dir, img_dir):
    """STRICTLY parse only images that belong to the MAPLES-DR TRAIN split"""
    data = []
    # Only look for masks in the train directory
    for lesion in LESION_TYPES:
        train_mask_files = glob.glob(os.path.join(maples_dir, "train", lesion, "*.png"))
        for mask_path in train_mask_files:
            base_name = os.path.basename(mask_path).replace('.png', '')
            
            # Find its grade from diagnosis.csv
            csv_path = os.path.join(maples_dir, "train", "diagnosis.csv")
            df = pd.read_csv(csv_path)
            grade_row = df[df['name'] == base_name]
            if len(grade_row) == 0: continue
            grade = map_messidor_label(grade_row.iloc[0]['DR'])
            
            img_path = get_exact_image_path(base_name, img_dir)
            if img_path:
                data.append({'image_id': base_name, 'image_path': img_path, 'label': grade})
    
    # Remove duplicates since an image might have multiple lesions
    df_unique = pd.DataFrame(data).drop_duplicates(subset=['image_id'])
    return df_unique

# ==========================================
# DATASET WITH MASTER MASK
# ==========================================
class AttentionGuidedDataset(Dataset):
    def __init__(self, df, img_dir, maples_dir, img_size, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.maples_dir = maples_dir
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        base_name = self.df.iloc[idx]['image_id']
        label = float(self.df.iloc[idx]['label'])
        img_path = self.df.iloc[idx]['image_path']

        original_img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        master_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        # ONLY look in the train folder
        for lesion in LESION_TYPES:
            l_path = os.path.join(self.maples_dir, "train", lesion, f"{base_name}.png")
            if os.path.exists(l_path):
                lesion_mask = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)
                if lesion_mask.shape[:2] != master_mask.shape[:2]:
                    lesion_mask = cv2.resize(lesion_mask, (master_mask.shape[1], master_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                master_mask = cv2.bitwise_or(master_mask, lesion_mask)

        gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        mask_bool = gray_img > 7
        coords = np.argwhere(mask_bool)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            clean_img = img_rgb[y0:y1, x0:x1]
            clean_mask = master_mask[y0:y1, x0:x1]
        else:
            clean_img = img_rgb
            clean_mask = master_mask

        clean_img = apply_clahe(clean_img)

        if self.transforms:
            augmented = self.transforms(image=clean_img, mask=clean_mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask'].to(torch.float32) / 255.0 
            mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, torch.tensor(label, dtype=torch.float32), mask_tensor

def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
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

# ==========================================
# TRAINING PIPELINE WITH DUAL LOSS
# ==========================================
def train_attention_epoch(model, dataloader, optimizer, cls_criterion, attn_criterion, scheduler, scaler, alpha, device):
    model.train()
    total_cls_loss, total_attn_loss = 0, 0
    
    attention_maps = []
    def cbam_hook_fn(module, input, output):
        attention_maps.append(output)

    hook = model.cbam.sa.register_forward_hook(cbam_hook_fn)

    pbar = tqdm(dataloader, desc="[Train]", leave=False)
    for images, labels, masks in pbar:
        images, labels, masks = images.to(device), labels.to(device).unsqueeze(1), masks.to(device)
        
        attention_maps.clear()
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            preds = model(images)
            loss_cls = cls_criterion(preds, labels)
            
            cbam_map = attention_maps[0] 
            cbam_map_upsampled = F.interpolate(cbam_map, size=(masks.shape[1], masks.shape[2]), mode='bilinear', align_corners=False).squeeze(1)
            
            # Attention loss forces CBAM to mimic the expert mask
            loss_attn = attn_criterion(cbam_map_upsampled, masks)
            loss_total = loss_cls + alpha * loss_attn
            
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_cls_loss += loss_cls.item()
        total_attn_loss += loss_attn.item()
        pbar.set_postfix(Cls=f"{loss_cls.item():.4f}", Attn=f"{loss_attn.item():.4f}")
        
    scheduler.step()
    hook.remove()
    return total_cls_loss / len(dataloader), total_attn_loss / len(dataloader)

@torch.no_grad()
def validate_attention_epoch(model, dataloader, cls_criterion, attn_criterion, device):
    model.eval()
    total_cls_loss, total_attn_loss = 0, 0
    all_preds, all_targets = [], []
    
    attention_maps = []
    def cbam_hook_fn(module, input, output):
        attention_maps.append(output)
    hook = model.cbam.sa.register_forward_hook(cbam_hook_fn)

    pbar = tqdm(dataloader, desc="[Valid]", leave=False)
    for images, labels, masks in pbar:
        images, labels, masks = images.to(device), labels.to(device).unsqueeze(1), masks.to(device)
        attention_maps.clear()
        
        with torch.cuda.amp.autocast():
            preds = model(images)
            loss_cls = cls_criterion(preds, labels)
            
            cbam_map = attention_maps[0]
            cbam_map_upsampled = F.interpolate(cbam_map, size=(masks.shape[1], masks.shape[2]), mode='bilinear', align_corners=False).squeeze(1)
            loss_attn = attn_criterion(cbam_map_upsampled, masks)
            
        total_cls_loss += loss_cls.item()
        total_attn_loss += loss_attn.item()
        
        all_preds.extend(preds.float().cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
    hook.remove()
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    return total_cls_loss / len(dataloader), total_attn_loss / len(dataloader), all_preds, all_targets

# ==========================================
# COMMAND LINE PARSER
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Attention-Guided Fine-Tuning on MAPLES-DR Train Split")
    parser.add_argument('--model', type=str, default='B7')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15) 
    parser.add_argument('--pretrain_path', type=str, required=True, help="Path to APTOS 0.92 best weights")
    parser.add_argument('--alpha', type=float, default=5.0, help="Weight multiplier for Attention Loss")
    return parser.parse_args()

def main():
    args = parse_args()
    img_size = EFF_SIZES[args.model.upper()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    MAPLES_DIR = os.path.join(THIS_DIR, "../../MAPLES-DR")
    MESSIDOR_IMG_DIR = os.path.join(THIS_DIR, "../../datasets1")

    exp_name = f"Stage3_AttnGuided_{args.model.upper()}_CBAM_Batch{args.batch_size}"
    SAVE_DIR = os.path.join(THIS_DIR, "../experiments", exp_name)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print_step("STAGE 3: ATTENTION-GUIDED FINE-TUNING (TRAIN SPLIT ONLY)")
    print(f"[INFO] Initializing experiment: {exp_name}")
    print(f"[INFO] Using Pretrained Weights: {args.pretrain_path}")

    # 1. Parse only the TRAIN split from MAPLES-DR
    df_train_all = build_maples_train_dataframe(MAPLES_DIR, MESSIDOR_IMG_DIR)
    if len(df_train_all) == 0:
        print_error_and_exit("No train images found. Check MAPLES-DR directory structure.", "")

    # Split 80/20 internally just to monitor validation loss, but ALL of these come from the official MAPLES-DR Train split.
    df_train, df_valid = train_test_split(df_train_all, test_size=0.2, stratify=df_train_all['label'], random_state=42)
    print_success(f"Parsed MAPLES-DR Train Set. Internal split -> Train: {len(df_train)} | Valid: {len(df_valid)}")

    train_dataset = AttentionGuidedDataset(df_train, MESSIDOR_IMG_DIR, MAPLES_DIR, img_size, transforms=get_train_transforms(img_size))
    valid_dataset = AttentionGuidedDataset(df_valid, MESSIDOR_IMG_DIR, MAPLES_DIR, img_size, transforms=get_valid_transforms(img_size))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Load the 0.92 APTOS Model
    model_name_timm = f"tf_efficientnet_{args.model.lower()}_ns"
    model = APTOSModel(model_name=model_name_timm, num_classes=1, use_cbam=True).to(device)
    
    if os.path.exists(args.pretrain_path):
        model.load_state_dict(torch.load(args.pretrain_path, map_location=device))
        print_success("Successfully loaded APTOS baseline weights.")
    else:
        print_error_and_exit("Cannot find pre-trained weights.", args.pretrain_path)

    # 3. Setup Loss and Optimizer
    cls_criterion = nn.SmoothL1Loss() 
    attn_criterion = nn.MSELoss() 
    
    # Ultra-low learning rate to maintain the 0.92 QWK capability
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    
    scaler = torch.cuda.amp.GradScaler()
    optR = OptimizedRounder()
    best_qwk = -1.0
    best_attn_loss = 999.0

    # 4. Training Loop
    print_step("STARTING DUAL-OBJECTIVE TRAINING")
    for epoch in range(args.epochs):
        print(f"\n[EPOCH {epoch+1}/{args.epochs}]")
        
        train_cls_loss, train_attn_loss = train_attention_epoch(model, train_loader, optimizer, cls_criterion, attn_criterion, scheduler, scaler, args.alpha, device)
        val_cls_loss, val_attn_loss, mean_preds, all_targets = validate_attention_epoch(model, valid_loader, cls_criterion, attn_criterion, device)

        optR.fit(mean_preds, all_targets)
        rounded_preds = optR.predict(mean_preds, optR.coef_['x'])
        val_qwk = calculate_qwk(all_targets, rounded_preds)

        print(f"   [RESULT] Train Cls: {train_cls_loss:.4f} | Train Attn: {train_attn_loss:.4f}")
        print(f"   [RESULT] Valid Cls: {val_cls_loss:.4f} | Valid Attn: {val_attn_loss:.4f} | Valid QWK: {val_qwk:.4f}")

        # Save model if QWK improves OR if QWK is stable but Attention Loss decreases
        if val_qwk > best_qwk or (val_qwk == best_qwk and val_attn_loss < best_attn_loss):
            best_qwk = val_qwk
            best_attn_loss = val_attn_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "stage3_attention_best.pth"))
            np.save(os.path.join(SAVE_DIR, "best_thresholds.npy"), optR.coef_['x'])
            print(f"   [!] New Best Model Saved! (QWK: {best_qwk:.4f} | Attn Loss: {best_attn_loss:.4f})")

    print(f"\n[INFO] Stage 3 Completed. New weights and thresholds are saved in {SAVE_DIR}.")

if __name__ == "__main__":
    main()