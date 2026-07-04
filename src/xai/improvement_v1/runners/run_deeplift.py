import os
import sys
import cv2
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Path routing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XAI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))     # src/xai
SRC_DIR = os.path.abspath(os.path.join(XAI_DIR, ".."))            # src
TRAINING_DIR = os.path.join(SRC_DIR, "training")
for _p in (XAI_DIR, TRAINING_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import APTOSModel
from augmentation import get_valid_transforms
from preprocessing import apply_clahe, crop_image_from_gray

# Imports from Improvement V1 Core
from improvement_v1.core.deeplift_core import DeepLIFTCore

def print_step(step_name):
    print(f"\n{'-'*80}\n STEP: {step_name}\n{'-'*80}")

def print_success(msg):
    print(f"   [SUCCESS] {msg}")

def print_error_and_exit(msg, error=""):
    print(f"   [FAILED] {msg}\n SYSTEM ERROR DETAILS: {error}")
    sys.exit(1)

def print_warning(msg):
    print(f"   [WARNING] {msg}")

def parse_args():
    parser = argparse.ArgumentParser(description="Runner for DeepLIFT (Reference-based XAI)")
    parser.add_argument('--model', type=str, default='B7', help="Model version")
    parser.add_argument('--weight_path', type=str, required=True, help="Path to the .pth model weights")
    parser.add_argument('--use_cbam', action='store_true', help="Flag for CBAM version")
    parser.add_argument('--max_samples', type=int, default=999, help="Max images to process")
    parser.add_argument('--img_dir', type=str, required=True, help="Explicit path to the directory containing images")
    parser.add_argument('--csv_path', type=str, required=True, help="Explicit path to the CSV file containing image IDs")
    parser.add_argument('--baseline_mode', type=str, default='blurred', choices=['zeros', 'blurred'], help="Type of baseline image for DeepLIFT")
    return parser.parse_args()

def get_exact_image_path(base_name, search_dir):
    for root, dirs, files in os.walk(search_dir):
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            if base_name + ext in files:
                return os.path.join(root, base_name + ext)
    return None

def main():
    args = parse_args()
    
    IMG_SIZE = 600
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Directory Setup
    EXP_NAME = f"Eff_{args.model.upper()}"
    EXP_NAME += "_CBAM" if args.use_cbam else "_Base"
    EXP_NAME += "_AttnGuided" 
    
    OUTPUT_DIR = os.path.join(XAI_DIR, "improvement_v1", "xai_outputs", EXP_NAME, "DeepLIFT")
    os.makedirs(os.path.join(OUTPUT_DIR, "npy"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "visuals"), exist_ok=True)

    print_step("DEEPLIFT EXECUTION INITIALIZATION")
    print(f"[INFO] Weights Path : {args.weight_path}")
    print(f"[INFO] Image Target : {args.img_dir}")
    print(f"[INFO] CSV Target   : {args.csv_path}")
    print(f"[INFO] Baseline Mode: {args.baseline_mode}")

    # 1. Load Model
    model_name_timm = f"tf_efficientnet_{args.model.lower()}_ns"
    model = APTOSModel(model_name=model_name_timm, num_classes=1, use_cbam=args.use_cbam, pretrained=False)
    
    weight_path_full = os.path.join(SRC_DIR, args.weight_path)
    if not os.path.exists(weight_path_full):
        print_error_and_exit("Weights not found.", weight_path_full)
        
    model.load_state_dict(torch.load(weight_path_full, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    print_success("Model and weights loaded successfully.")

    # 2. Setup Core Explainer
    explainer = DeepLIFTCore(model, device=DEVICE)
    transform = get_valid_transforms(IMG_SIZE)

    # 3. Parse CSV to get target images
    if not os.path.exists(args.csv_path):
        print_error_and_exit("CSV file not found.", args.csv_path)
        
    df_valid = pd.read_csv(args.csv_path)
    
    id_col = None
    for col in ['image_filename', 'image_id', 'name', 'id_code']:
        if col in df_valid.columns:
            id_col = col
            break
            
    if id_col is None:
        print_error_and_exit("Could not find an Image ID column in the provided CSV.", f"Columns: {list(df_valid.columns)}")
        
    image_list = df_valid[id_col].astype(str).tolist()
    print_success(f"Extracted {len(image_list)} targets from CSV.")

    processed_count = 0

    # 4. Execution Loop
    print_step(f"GENERATING DEEPLIFT HEATMAPS (Max Samples: {args.max_samples})")
    
    for item in tqdm(image_list, desc="Processing"):
        if processed_count >= args.max_samples: break
        
        base_name = os.path.splitext(item)[0]
        
        img_path = get_exact_image_path(base_name, args.img_dir)
        if not img_path: 
            print_warning(f"Image file not found for ID: {base_name}")
            continue

        original_img = cv2.imread(img_path)
        if original_img is None:
            print_warning(f"Failed to read image at: {img_path}")
            continue

        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        clean_img = apply_clahe(crop_image_from_gray(img_rgb))
        clean_img_resized = cv2.resize(clean_img, (IMG_SIZE, IMG_SIZE))
        vis_img_float = np.float32(clean_img_resized) / 255.0
        
        input_tensor = transform(image=clean_img)['image'].unsqueeze(0).to(DEVICE)

        # Generate Heatmap (Notice it is extremely fast)
        # Using a try-catch block because Captum DeepLift can be sensitive to certain layers
        try:
            heatmap = explainer.generate_heatmap(input_tensor, baseline_mode=args.baseline_mode)
        except Exception as e:
            print_warning(f"DeepLIFT failed on image {base_name}. Error: {e}. Skipping.")
            continue

        # Save NPY
        np.save(os.path.join(OUTPUT_DIR, "npy", f"{base_name}_DeepLIFT.npy"), heatmap)

        # Save Overlay Visual
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = np.clip(0.7 * vis_img_float + 0.3 * np.float32(heatmap_color)/255.0, 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(vis_img_float); axes[0].set_title(f"Original: {base_name}"); axes[0].axis('off')
        axes[1].imshow(overlay); axes[1].set_title(f"DeepLIFT (Baseline: {args.baseline_mode})"); axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "visuals", f"{base_name}_DeepLIFT.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        processed_count += 1

    print_step("EXECUTION SUMMARY")
    print_success(f"Successfully processed {processed_count} images.")
    print_success(f"Results securely saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()