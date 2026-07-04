import os
import sys
import glob
import cv2
import argparse
import re
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

from augmentation import get_valid_transforms
from preprocessing import apply_clahe, crop_image_from_gray
from improvement_v1.core.vws_core import VWSCore

# ==========================================
# SYSTEM LOGGER
# ==========================================
def print_step(step_name):
    print(f"\n{'-'*80}\n STEP: {step_name}\n{'-'*80}")

def print_success(msg):
    print(f"   [SUCCESS] {msg}")

def print_error_and_exit(msg, error=""):
    print(f"   [FAILED] {msg}\n SYSTEM ERROR DETAILS: {error}")
    sys.exit(1)

def print_warning(msg):
    print(f"   [WARNING] {msg}")

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def get_exact_image_path(base_name, search_dir):
    for root, dirs, files in os.walk(search_dir):
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            if base_name + ext in files:
                return os.path.join(root, base_name + ext)
    return None

def find_method_npy(npy_dir, img_id, method_keywords):
    """Recursively scan all subdirectories for matching ID and method."""
    search_pattern = os.path.join(npy_dir, "**", f"*{img_id}*.npy")
    possible_files = glob.glob(search_pattern, recursive=True)
    
    for f in possible_files:
        f_lower = f.lower() 
        if any(kw.lower() in f_lower for kw in method_keywords):
            return f
    return None

def create_overlay(image_rgb, heatmap, alpha=0.30, min_visible=1e-6):
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    else:
        image_rgb = image_rgb.astype(np.uint8)

    h, w = image_rgb.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    support = heatmap > min_visible
    overlay = image_rgb.astype(np.float32).copy()
    overlay[support] = (1.0 - alpha) * image_rgb.astype(np.float32)[support] + alpha * heatmap_color.astype(np.float32)[support]
    return np.clip(overlay, 0, 255).astype(np.uint8)

# ==========================================
# COMMAND LINE PARSER
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Runner for Variance-Weighted Saliency (VWS)")
    parser.add_argument('--xai_base_dir', type=str, required=True, help="Path to XAI run directory")
    parser.add_argument('--img_dir', type=str, required=True, help="Explicit path to original images")
    parser.add_argument('--csv_path', type=str, required=True, help="Explicit path to CSV containing image IDs")
    parser.add_argument('--img_size', type=int, default=600, help="Image resolution")
    return parser.parse_args()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    args = parse_args()
    
    INPUT_NPY_DIR = os.path.join(args.xai_base_dir, "npy")
    OUTPUT_NPY_DIR = os.path.join(args.xai_base_dir, "npy")
    OUTPUT_VIS_DIR = os.path.join(args.xai_base_dir, "visuals", "VWS_Fusion")
    
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    print_step("VWS FUSION INITIALIZATION")
    print(f"[INFO] Reading .npy files recursively from: {INPUT_NPY_DIR}")

    if not os.path.exists(args.csv_path):
        print_error_and_exit("CSV file not found.", args.csv_path)
        
    df_valid = pd.read_csv(args.csv_path)
    id_col = next((col for col in ['image_filename', 'image_id', 'name', 'id_code'] if col in df_valid.columns), None)
            
    if id_col is None:
        print_error_and_exit("Could not find an Image ID column.", "")
        
    image_list = df_valid[id_col].astype(str).tolist()

    fusion_engine = VWSCore()
    processed_count = 0

    print_step("EXECUTING VARIANCE-WEIGHTED SALIENCY FUSION")
    
    for item in tqdm(image_list, desc="Fusing Maps"):
        base_name = os.path.splitext(item)[0]
        
        # [CRITICAL FIX] Clean the prefix "GradeX_" if it exists so we match the true NPY filename
        clean_id = re.sub(r'^Grade\d+_', '', base_name)
        
        # Flexible matching: Accepts either ScoreCAM or GradCAM as the first map
        map1_path = find_method_npy(INPUT_NPY_DIR, clean_id, ["scorecam", "score-cam", "gradcam", "gcam"])
        adas_path = find_method_npy(INPUT_NPY_DIR, clean_id, ["adasise", "ada-sise"])
        igsg_path = find_method_npy(INPUT_NPY_DIR, clean_id, ["smoothig", "smooth-ig", "igsg"])

        if not (map1_path and adas_path and igsg_path):
            print_warning(f"Missing one or more XAI maps for {clean_id}. Skipping.")
            continue

        map1 = np.load(map1_path)
        map_a = np.load(adas_path)
        map_i = np.load(igsg_path)

        # Generate VWS
        vws_map = fusion_engine.generate(map1, map_a, map_i)
        np.save(os.path.join(OUTPUT_NPY_DIR, f"{clean_id}_VWS.npy"), vws_map)

        # Process Visuals
        img_path = get_exact_image_path(clean_id, args.img_dir)
        if not img_path: continue
        
        original_img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        clean_img = apply_clahe(crop_image_from_gray(img_rgb))
        clean_img_resized = cv2.resize(clean_img, (args.img_size, args.img_size))
        vis_img_float = np.float32(clean_img_resized) / 255.0

        ov_1 = create_overlay(vis_img_float, map1)
        ov_a = create_overlay(vis_img_float, map_a)
        ov_i = create_overlay(vis_img_float, map_i)
        ov_vws = create_overlay(vis_img_float, vws_map)

        # Determine title for the first map
        map1_title = "Score-CAM" if "score" in map1_path.lower() else "Grad-CAM++"

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(vis_img_float); axes[0].set_title(f"Original: {clean_id}"); axes[0].axis('off')
        axes[1].imshow(ov_1); axes[1].set_title(map1_title); axes[1].axis('off')
        axes[2].imshow(ov_a); axes[2].set_title("Ada-SISE"); axes[2].axis('off')
        axes[3].imshow(ov_i); axes[3].set_title("Smooth-IG"); axes[3].axis('off')
        axes[4].imshow(ov_vws); axes[4].set_title("FINAL VWS MAP", fontweight='bold', color='red'); axes[4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_VIS_DIR, f"{clean_id}_VWS_Comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        processed_count += 1

    print_step("EXECUTION SUMMARY")
    print_success(f"Successfully fused {processed_count} images.")
    print_success(f"VWS Numpy files saved to : {OUTPUT_NPY_DIR}/")
    print_success(f"VWS Visuals saved to     : {OUTPUT_VIS_DIR}/")

if __name__ == "__main__":
    main()