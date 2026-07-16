import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    cohen_kappa_score, 
    accuracy_score, 
    f1_score, 
    precision_score,
    recall_score,
    confusion_matrix, 
    classification_report
)

# Import internal project modules
from model import APTOSModel
from preprocessing import crop_image_from_gray, apply_clahe

# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================
# Dynamically get the directory where evaluate.py is located
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to get the Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))

# Dynamically construct all other paths based on the project structure
IMG_DIR = os.path.join(PROJECT_ROOT, "datasets", "aptos2019-blindness-detection", "train_images")
CSV_PATH = os.path.join(THIS_DIR, "../data", "train_folds.csv")

# Experiment paths
EXP_DIR = os.path.join(THIS_DIR, "../experiments", "Stage2_Finetune_B7_CBAM")
MODEL_PATH = os.path.join(EXP_DIR, "stage2_best_model.pth")
NPY_PATH = os.path.join(EXP_DIR, "best_thresholds.npy")

# Evaluation Report Save Path
REPORT_PATH = os.path.join(EXP_DIR, "evaluation_report.txt")

# Model execution parameters
FOLD_TO_EVAL = 0       
MODEL_NAME = "B7"
IMG_SIZE = 600         
USE_CBAM = True
BATCH_SIZE = 16

# ==========================================
# 2. DATASET AND AUGMENTATION
# ==========================================
class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_name}.png")
        label = float(self.df.iloc[idx]['diagnosis'])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Consistent preprocessing with the training pipeline
        image = crop_image_from_gray(image)
        image = apply_clahe(image)

        if self.transforms:
            image = self.transforms(image=image)['image']
        return image, torch.tensor(label, dtype=torch.float32)

def get_valid_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_tta_transforms(img_size):
    """Test-Time Augmentation transforms (Horizontal and Vertical Flips)"""
    return [
        A.Compose([A.Resize(height=img_size, width=img_size), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(height=img_size, width=img_size), A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
    ]

# ==========================================
# 3. THRESHOLD APPLICATION
# ==========================================
def apply_thresholds(predictions, thresholds):
    """
    Converts continuous regression predictions into discrete integer classes (0 to 4) 
    using the optimized thresholds generated during training.
    """
    preds_rounded = np.copy(predictions)
    for i, pred in enumerate(preds_rounded):
        if pred < thresholds[0]:   preds_rounded[i] = 0
        elif pred < thresholds[1]: preds_rounded[i] = 1
        elif pred < thresholds[2]: preds_rounded[i] = 2
        elif pred < thresholds[3]: preds_rounded[i] = 3
        else:                      preds_rounded[i] = 4
    return preds_rounded.astype(int)

# ==========================================
# 4. EVALUATION PIPELINE
# ==========================================
@torch.no_grad()
def evaluate():
    # Verify the existence of essential files
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model weights not found at: {MODEL_PATH}")
        return
    if not os.path.exists(NPY_PATH):
        print(f"[ERROR] Threshold configuration file not found at: {NPY_PATH}")
        return

    # Load dataset fold
    df = pd.read_csv(CSV_PATH)
    df_valid = df[df['fold'] == FOLD_TO_EVAL]
    print(f"[INFO] Validation subset initiated (Fold {FOLD_TO_EVAL}) with {len(df_valid)} samples.")

    # Initialize model architecture
    model_name_timm = f"tf_efficientnet_{MODEL_NAME.lower()}_ns"
    model = APTOSModel(model_name=model_name_timm, num_classes=1, use_cbam=USE_CBAM).cuda()
    
    # Load optimal weights and evaluation states
    print(f"[INFO] Loading model weights from: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    thresholds = np.load(NPY_PATH)
    print(f"[INFO] Applied regression thresholds: {thresholds}")

    # Prepare TTA (Test-Time Augmentation)
    transforms_list = [get_valid_transforms(IMG_SIZE)] + get_tta_transforms(IMG_SIZE)
    all_tta_preds = []
    all_targets = []
    targets_collected = False

    # Execute inference iterations
    for tta_idx, transform in enumerate(transforms_list):
        dataset = APTOSDataset(df_valid, IMG_DIR, transforms=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        preds = []
        pbar = tqdm(dataloader, desc=f"[INFO] Inference execution (TTA {tta_idx+1}/3)", leave=False)
        for images, labels in pbar:
            images = images.cuda()
            with torch.cuda.amp.autocast():
                outputs = model(images)
            preds.extend(outputs.float().cpu().numpy())
            
            if not targets_collected:
                all_targets.extend(labels.numpy())
                
        all_tta_preds.append(np.array(preds).flatten())
        targets_collected = True 

    # ==========================================
    # 5. METRICS CALCULATION & EXPORT
    # ==========================================
    # Aggregate TTA predictions via mean pooling
    mean_preds = np.mean(all_tta_preds, axis=0)
    all_targets = np.array(all_targets).flatten().astype(int)
    
    # Discretize continuous predictions
    final_preds = apply_thresholds(mean_preds, thresholds)

    # Compute key performance indicators
    qwk = cohen_kappa_score(all_targets, final_preds, weights='quadratic')
    acc = accuracy_score(all_targets, final_preds)
    
    macro_precision = precision_score(all_targets, final_preds, average='macro', zero_division=0)
    macro_recall    = recall_score(all_targets, final_preds, average='macro', zero_division=0)
    macro_f1        = f1_score(all_targets, final_preds, average='macro', zero_division=0)
    weighted_f1     = f1_score(all_targets, final_preds, average='weighted', zero_division=0)
    
    cls_report = classification_report(all_targets, final_preds, digits=4, zero_division=0)
    cm = confusion_matrix(all_targets, final_preds)

    # Construct the final report string
    report_content = f"""
=================================================================
 EVALUATION METRICS SUMMARY (Model: {MODEL_NAME} | Fold: {FOLD_TO_EVAL})
=================================================================
 Quadratic Weighted Kappa (QWK):    {qwk:.5f}
 Accuracy:                          {acc:.5f}
 Macro Precision:                   {macro_precision:.5f}
 Macro Recall (Sensitivity):        {macro_recall:.5f}
 Macro F1-Score:                    {macro_f1:.5f}
 Weighted F1-Score:                 {weighted_f1:.5f}
=================================================================

 CLASSIFICATION REPORT:
{cls_report}

 CONFUSION MATRIX:
{cm}
=================================================================
"""
    
    # Print to console
    print(report_content)

    # Save to text file
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"[SUCCESS] Evaluation report saved successfully to: \n -> {REPORT_PATH}")

if __name__ == "__main__":
    evaluate()