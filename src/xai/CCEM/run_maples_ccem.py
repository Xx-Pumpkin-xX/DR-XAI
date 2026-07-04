import os
import sys
import glob
import cv2
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Path routing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XAI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))        # src/xai
SRC_DIR = os.path.abspath(os.path.join(XAI_DIR, ".."))            # src
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))       # repo root
TRAINING_DIR = os.path.join(SRC_DIR, "training")
for _p in (XAI_DIR, TRAINING_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import APTOSModel
from augmentation import get_valid_transforms
from preprocessing import apply_clahe
from pytorch_grad_cam.utils.image import show_cam_on_image

# IMPORT FROM CORE
from CCEM.ccem_core import (
    min_max_normalize, 
    calculate_spearman, 
    generate_ccem, 
    calculate_advanced_metrics
)

# ==========================================
# SYSTEM LOGGER & FILE UTILS
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

def map_messidor_label(dr_str):
    dr_str = str(dr_str).strip().upper()
    for i in range(5):
        if str(i) in dr_str: return i
    return 0

def crop_image_and_mask(img, mask, tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask_bool = gray_img > tol
    coords = np.argwhere(mask_bool)
    if coords.size == 0: return img, mask
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]

def get_exact_image_path(base_name, search_dir):
    for root, dirs, files in os.walk(search_dir):
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            if base_name + ext in files:
                return os.path.join(root, base_name + ext)
    return None

def find_method_npy(base_dir, img_id, method_keywords):
    search_pattern = os.path.join(base_dir, "**", f"*{img_id}*.npy")
    possible_files = glob.glob(search_pattern, recursive=True)
    for f in possible_files:
        f_lower = os.path.basename(f).lower()
        if any(kw.lower() in f_lower for kw in method_keywords):
            return f
    return None

# ==========================================
# COMMAND LINE PARSER
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate CCEM and ODExAI Metrics from NPY files")
    parser.add_argument('--xai_dir', type=str, required=True, help="Path to directory containing NPY files")
    parser.add_argument('--weight_path', type=str, required=True, help="Path to .pth model weights")
    parser.add_argument('--model', type=str, default='B7', help="Model version")
    parser.add_argument('--use_cbam', action='store_true', help="Flag for CBAM version")
    return parser.parse_args()

# ==========================================
# MAIN RUNNER
# ==========================================
def main():
    args = parse_args()
    
    IMG_SIZE = 600
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LESION_TYPES = ["Microaneurysms", "Hemorrhages", "Exudates", "CottonWoolSpots"]
    
    MESSIDOR_IMG_DIR = os.path.join(PROJECT_ROOT, "datasets1")
    MAPLES_DIR = os.path.join(PROJECT_ROOT, "MAPLES-DR")
    
    OUTPUT_DIR = os.path.join(args.xai_dir, "CCEM_Evaluation_Results")
    os.makedirs(os.path.join(OUTPUT_DIR, "visuals"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "npy"), exist_ok=True)

    print_step("CCEM & METRICS EVALUATION INITIALIZATION")
    
    # Load Thresholds
    weight_dir = os.path.dirname(args.weight_path)
    threshold_path = os.path.join(weight_dir, "best_thresholds.npy")
    if not os.path.exists(threshold_path):
        print_error_and_exit("Thresholds file not found!", threshold_path)
    coef = np.load(threshold_path)
    print_success("Regression thresholds loaded.")

    # Load Model (Required for Faithfulness Metrics)
    model_name_timm = f"tf_efficientnet_{args.model.lower()}_ns"
    model = APTOSModel(model_name=model_name_timm, num_classes=1, use_cbam=args.use_cbam, pretrained=False)
    if not os.path.exists(args.weight_path):
        print_error_and_exit("Weights file not found!", args.weight_path)
    model.load_state_dict(torch.load(args.weight_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    print_success("Model weights loaded.")

    transform = get_valid_transforms(IMG_SIZE)

    # Trace True Labels
    true_labels = {}
    csv_files = glob.glob(os.path.join(MAPLES_DIR, "**", "diagnosis.csv"), recursive=True)
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            true_labels[str(row['name'])] = map_messidor_label(row['DR'])

    # Find unique Image IDs from the NPY directory
    csv_files = glob.glob(os.path.join(args.xai_dir, "*_results.csv"))
    if not csv_files:
        print_error_and_exit("No CSV files found in XAI directory to extract exact IDs.", args.xai_dir)
    
    df_ref = pd.read_csv(csv_files[0])
    image_ids = set(df_ref['Image_ID'].astype(str))

    print(f"[INFO] Found {len(image_ids)} unique image IDs in {args.xai_dir}")

    # Metrics Tracking
    xai_metrics = {m: {'EBPG': [], 'PG': [], 'Sparsity': [], 'Del': [], 'Ins': [], 'OA': []} 
                   for m in ["GradCAM++", "Ada-SISE", "IG_Smooth", "CCEM"]}
    xas_results = {'Corr_G_Ada': [], 'Corr_G_IG': [], 'Corr_Ada_IG': [], 'XAS': []}
    rows = []
    processed_count = 0

    print_step("FUSING MAPS AND CALCULATING METRICS")

    for img_id in tqdm(image_ids, desc="Evaluating"):
        if img_id not in true_labels: continue

        # Load NPY files
        gcam_path = find_method_npy(args.xai_dir, img_id, ["gradcam", "gcam"])
        adas_path = find_method_npy(args.xai_dir, img_id, ["adasise", "ada-sise"])
        igsg_path = find_method_npy(args.xai_dir, img_id, ["smoothig", "smooth-ig", "igsg"])

        if not (gcam_path and adas_path and igsg_path):
            continue

        G = np.load(gcam_path)
        AdaS = np.load(adas_path)
        IG_map = np.load(igsg_path)

        # Generate CCEM (CALLING CORE)
        C = generate_ccem(G, AdaS, IG_map, tau=0.85)
        np.save(os.path.join(OUTPUT_DIR, "npy", f"{img_id}_CCEM.npy"), C)

        # Image Processing for Model Input & Visualization
        img_path = get_exact_image_path(img_id, MESSIDOR_IMG_DIR)
        if not img_path: continue

        original_img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        master_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        for lesion in LESION_TYPES:
            l_path_train = os.path.join(MAPLES_DIR, "train", lesion, f"{img_id}.png")
            l_path_test = os.path.join(MAPLES_DIR, "test", lesion, f"{img_id}.png")
            l_path = l_path_train if os.path.exists(l_path_train) else l_path_test
            
            if os.path.exists(l_path):
                lesion_mask = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)
                if lesion_mask.shape[:2] != master_mask.shape[:2]:
                    lesion_mask = cv2.resize(lesion_mask, (master_mask.shape[1], master_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                master_mask = cv2.bitwise_or(master_mask, lesion_mask)

        clean_img, clean_mask = crop_image_and_mask(img_rgb, master_mask)
        clean_img = apply_clahe(clean_img)
        clean_img_resized = cv2.resize(clean_img, (IMG_SIZE, IMG_SIZE))
        vis_img_float = np.float32(clean_img_resized) / 255.0
        mask_binary = (cv2.resize(clean_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)

        input_tensor = transform(image=clean_img)['image'].unsqueeze(0).to(DEVICE)

        # Get Prediction
        with torch.no_grad():
            pred_value = model(input_tensor).item()
            
        if pred_value < coef[0]: pred_class = 0
        elif pred_value < coef[1]: pred_class = 1
        elif pred_value < coef[2]: pred_class = 2
        elif pred_value < coef[3]: pred_class = 3
        else: pred_class = 4
            
        true_class = true_labels[img_id]

        # Metric Calculations (CALLING CORE)
        if np.sum(mask_binary) > 0: 
            # XAS Calculation
            corr_g_ada = calculate_spearman(G, AdaS)
            corr_g_ig = calculate_spearman(G, IG_map)
            corr_ada_ig = calculate_spearman(AdaS, IG_map)
            xas_score = np.mean([corr_g_ada, corr_g_ig, corr_ada_ig])
            
            xas_results['Corr_G_Ada'].append(corr_g_ada)
            xas_results['Corr_G_IG'].append(corr_g_ig)
            xas_results['Corr_Ada_IG'].append(corr_ada_ig)
            xas_results['XAS'].append(xas_score)

            heatmaps = {"GradCAM++": G, "Ada-SISE": AdaS, "IG_Smooth": IG_map, "CCEM": C}
            row_data = {"Image_ID": img_id, "True_Grade": true_class, "Pred_Grade": pred_class, "XAS": xas_score}
            
            for method, hm in heatmaps.items():
                ebpg, pg, sparsity, del_auc, ins_auc, oa = calculate_advanced_metrics(model, input_tensor, hm, mask_binary)
                xai_metrics[method]['EBPG'].append(ebpg)
                xai_metrics[method]['PG'].append(pg)
                xai_metrics[method]['Sparsity'].append(sparsity)
                xai_metrics[method]['Del'].append(del_auc)
                xai_metrics[method]['Ins'].append(ins_auc)
                xai_metrics[method]['OA'].append(oa)
                
                row_data[f"{method}_EBPG"] = ebpg
                row_data[f"{method}_OA"] = oa 

            rows.append(row_data)

            # Visualization
            fig, axes = plt.subplots(1, 6, figsize=(30, 5))
            axes[0].imshow(vis_img_float); axes[0].set_title(f"Orig | True: {true_class}, Pred: {pred_class}"); axes[0].axis('off')
            
            mask_overlay = np.zeros_like(vis_img_float)
            mask_overlay[:, :, 1] = mask_binary
            gt_vis = cv2.addWeighted(vis_img_float, 0.7, mask_overlay, 0.5, 0)
            axes[1].imshow(gt_vis); axes[1].set_title("Expert Master Mask"); axes[1].axis('off')
            
            axes[2].imshow(show_cam_on_image(vis_img_float, G, use_rgb=True, colormap=cv2.COLORMAP_JET)); axes[2].set_title("Grad-CAM++"); axes[2].axis('off')
            axes[3].imshow(show_cam_on_image(vis_img_float, AdaS, use_rgb=True, colormap=cv2.COLORMAP_JET)); axes[3].set_title("Ada-SISE"); axes[3].axis('off')
            axes[4].imshow(show_cam_on_image(vis_img_float, IG_map, use_rgb=True, colormap=cv2.COLORMAP_JET)); axes[4].set_title("IG + SmoothGrad"); axes[4].axis('off')
            axes[5].imshow(show_cam_on_image(vis_img_float, C, use_rgb=True, colormap=cv2.COLORMAP_JET)); axes[5].set_title(f"FINAL CCEM (XAS: {xas_score:.3f})", color='red', fontweight='bold'); axes[5].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "visuals", f"{img_id}_fusion.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        processed_count += 1

    print_step("FINAL EVALUATION REPORT")
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "ccem_metrics_details.csv"), index=False)

    report_content = f"""
================================================================================
                    FINAL XAI FUSION REPORT (FROM NPY FILES)                  
================================================================================
Total Images Fused : {processed_count}

[1] XAI AGREEMENT SCORE (XAS):
    GLOBAL MEAN XAS      : {np.mean(xas_results['XAS']):.4f}

[2] ODExAI EVALUATION METRICS:
--------------------------------------------------------------------------------
Method          | EBPG (Energy) | Pointing Game | Sparsity | Deletion (↓) | Insertion (↑) | Over-All
--------------------------------------------------------------------------------
"""
    for method in ["GradCAM++", "Ada-SISE", "IG_Smooth", "CCEM"]:
        if len(xai_metrics[method]['EBPG']) > 0:
            m_ebpg = np.mean(xai_metrics[method]['EBPG']) * 100
            m_pg = np.mean(xai_metrics[method]['PG']) * 100
            m_spar = np.mean(xai_metrics[method]['Sparsity'])
            m_del = np.mean(xai_metrics[method]['Del'])
            m_ins = np.mean(xai_metrics[method]['Ins'])
            m_oa = np.mean(xai_metrics[method]['OA'])
            report_content += f"{method:<15} | {m_ebpg:>12.2f}% | {m_pg:>12.2f}% | {m_spar:>8.4f} | {m_del:>12.4f} | {m_ins:>13.4f} | {m_oa:>8.4f}\n"
        else:
            report_content += f"{method:<15} | {'N/A':>13} | {'N/A':>13} | {'N/A':>8} | {'N/A':>12} | {'N/A':>13} | {'N/A':>8}\n"
    
    report_content += "================================================================================\n"
    
    print(report_content)
    with open(os.path.join(OUTPUT_DIR, "final_ccem_report.txt"), "w") as f:
        f.write(report_content)
    print_success(f"Execution completed. All results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()