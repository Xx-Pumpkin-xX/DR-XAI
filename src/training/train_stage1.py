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
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Import nội bộ
from model import APTOSModel, calculate_qwk
from augmentation import get_train_transforms, get_valid_transforms
from preprocessing import crop_image_from_gray, apply_clahe
from rounder import OptimizedRounder

EFF_SIZES = {'B0': 224, 'B1': 240, 'B2': 260, 'B3': 300, 'B4': 380, 'B5': 456, 'B6': 528, 'B7': 600}

# ==========================================
# DATASET DÀNH CHO EYEPACS 2015
# ==========================================
class EyePacsDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        # File của EyePACS dùng cột 'image' và đuôi .jpeg
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, f"{img_name}.jpeg")
        label = float(self.df.iloc[idx]['level']) # Regression cần nhãn float

        image = cv2.imread(img_path)
        if image is None:
            # Nếu lỡ ảnh lỗi, lấy đại ảnh đen để không sập model
            image = np.zeros((224, 224, 3), dtype=np.uint8) 
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = crop_image_from_gray(image)
            image = apply_clahe(image)

        if self.transforms:
            image = self.transforms(image=image)['image']
            
        return image, torch.tensor(label, dtype=torch.float32)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='B6')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10) # 35k ảnh chỉ cần 10 epoch là đủ
    parser.add_argument('--use_cbam', action='store_true')
    args = parser.parse_args()

    args.model = args.model.upper()
    img_size = EFF_SIZES[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    # Trỏ vào thư mục EyePACS bạn đang tải
    CSV_PATH = os.path.join(THIS_DIR, "../../EyePACS_2015/trainLabels.csv")
    IMG_DIR = os.path.join(THIS_DIR, "../../EyePACS_2015/train_images/") # Hoặc resized_train tùy file giải nén

    SAVE_DIR = os.path.join(THIS_DIR, "../experiments", f"Stage1_Pretrain_{args.model}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n" + "="*60)
    print(" 🚀 STAGE 1: PRE-TRAINING TRÊN BỘ DỮ LIỆU EYEPACS 2015 (REGRESSION)")
    print("="*60)

    # 1. Đọc và chia dữ liệu (90% Train - 10% Valid)
    df = pd.read_csv(CSV_PATH)
    df_train, df_valid = train_test_split(df, test_size=0.1, stratify=df['level'], random_state=42)
    print(f"[*] Tổng số ảnh: Train ({len(df_train)}) | Valid ({len(df_valid)})")

    train_dataset = EyePacsDataset(df_train, IMG_DIR, transforms=get_train_transforms(img_size))
    valid_dataset = EyePacsDataset(df_valid, IMG_DIR, transforms=get_valid_transforms(img_size))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Khởi tạo Model (1 Node) và Smooth L1 Loss
    model_name_timm = f"tf_efficientnet_{args.model.lower()}_ns"
    model = APTOSModel(model_name=model_name_timm, num_classes=1, use_cbam=args.use_cbam).to(device)
    
    # HÀM LOSS CHO REGRESSION
    criterion = nn.SmoothL1Loss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_qwk = -1.0
    optR = OptimizedRounder()

    # 3. VÒNG LẶP TRAIN
    for epoch in range(args.epochs):
        print(f"\n👉 EPOCH {epoch+1}/{args.epochs}")
        
        # --- TRAIN ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc="[Train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1) # Chuyển label thành cột
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="[Valid]", leave=False):
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # Chuyển về dạng mảng 1 chiều
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()

        # TÌM NGƯỠNG LÀM TRÒN TỐI ƯU TRÊN TẬP VALID
        print("   [*] Đang chạy thuật toán Optimized Rounder...")
        optR.fit(all_preds, all_targets)
        
        # Ép số thập phân về số nguyên để tính QWK
        rounded_preds = optR.predict(all_preds, optR.coef_['x'])
        val_qwk = calculate_qwk(all_targets, rounded_preds)

        train_loss /= len(train_loader)
        val_loss /= len(valid_loader)
        
        print(f"   📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "stage1_best.pth"))
            
            # Lưu lại cả cái ngưỡng làm tròn để lát nữa dùng cho Stage 2
            np.save(os.path.join(SAVE_DIR, "best_thresholds.npy"), optR.coef_['x'])
            print(f"   🏆 Đã lưu Model Stage 1 (QWK: {best_qwk:.4f})")

if __name__ == "__main__":
    main()