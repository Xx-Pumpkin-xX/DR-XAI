import os
import sys
import glob
import shutil
import cv2
import numpy as np
import pandas as pd

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

# ==========================================
# CONFIGURATIONS
# ==========================================
# Excluded anatomical structures: Macula, OpticCup, OpticDisc, Vessels
LESION_TYPES = [
    "Microaneurysms", 
    "Hemorrhages", 
    "Exudates", 
    "CottonWoolSpots",
    "Drusens",
    "Neovascularization",
    "BrightUncertains",
    "RedUncertains"
]

# ==========================================
# DATA PARSING FUNCTIONS
# ==========================================
def map_messidor_label(dr_str):
    dr_str = str(dr_str).strip().upper()
    for i in range(5):
        if str(i) in dr_str: return int(i)
    return 0

def get_true_labels_dict(maples_dir):
    true_labels = {}
    csv_files = glob.glob(os.path.join(maples_dir, "**", "diagnosis.csv"), recursive=True)
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            true_labels[str(row['name'])] = map_messidor_label(row['DR'])
    return true_labels

def get_exact_image_path(base_name, search_dir):
    for root, dirs, files in os.walk(search_dir):
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            if base_name + ext in files:
                return os.path.join(root, base_name + ext)
    return None

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print_step("STRATIFIED SAMPLING: 10 IMAGES AND MASTER MASKS FROM MAPLES-DR")

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    MAPLES_DIR = os.path.join(THIS_DIR, "../../MAPLES-DR")
    MESSIDOR_IMG_DIR = os.path.join(THIS_DIR, "../../datasets1")
    
    OUTPUT_DIR = os.path.join(THIS_DIR, "Sampled_10_MAPLES")
    IMG_OUT_DIR = os.path.join(OUTPUT_DIR, "images")
    MASK_OUT_DIR = os.path.join(OUTPUT_DIR, "masks")
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(IMG_OUT_DIR, exist_ok=True)
    os.makedirs(MASK_OUT_DIR, exist_ok=True)

    print(f"[INFO] MAPLES-DR Path : {MAPLES_DIR}")
    print(f"[INFO] MESSIDOR Path  : {MESSIDOR_IMG_DIR}")
    print(f"[INFO] Included Lesions: {', '.join(LESION_TYPES)}")

    # 1. Get all labels
    true_labels_dict = get_true_labels_dict(MAPLES_DIR)
    if not true_labels_dict:
        print_error_and_exit("No labels found. Check MAPLES-DR directory.")
        
    print(f"[INFO] Extracted {len(true_labels_dict)} unique labels from MAPLES-DR diagnosis.csv.")

    # 2. Match with physical images in datasets1
    valid_data = []
    for base_name, grade in true_labels_dict.items():
        img_path = get_exact_image_path(base_name, MESSIDOR_IMG_DIR)
        if img_path:
            valid_data.append({
                'image_id': base_name, 
                'grade': grade, 
                'img_path': img_path
            })
            
    df_valid = pd.DataFrame(valid_data)
    print(f"[INFO] Successfully matched {len(df_valid)} images with physical files.")

    # 3. Perform Stratified Sampling (2 per grade)
    print("\n[INFO] Performing stratified sampling (2 images per grade)...")
    try:
        sampled_df = df_valid.groupby('grade').sample(n=2, random_state=42).reset_index(drop=True)
    except ValueError as e:
        print_error_and_exit("Cannot sample 2 images per grade. Check class distribution.", str(e))

    # 4. Process and Copy Files
    print_step("COPYING IMAGES AND GENERATING MASTER MASKS")
    
    final_csv_data = []

    for _, row in sampled_df.iterrows():
        img_id = row['image_id']
        grade = row['grade']
        src_img_path = row['img_path']
        original_filename = os.path.basename(src_img_path)
        
        # 4.1 Copy Original Image
        dest_img_path = os.path.join(IMG_OUT_DIR, original_filename)
        shutil.copy2(src_img_path, dest_img_path)
        
        # 4.2 Build Master Mask (Combining all pathological types)
        original_img = cv2.imread(src_img_path)
        h, w = original_img.shape[:2]
        master_mask = np.zeros((h, w), dtype=np.uint8)
        
        for lesion in LESION_TYPES:
            l_path_train = os.path.join(MAPLES_DIR, "train", lesion, f"{img_id}.png")
            l_path_test = os.path.join(MAPLES_DIR, "test", lesion, f"{img_id}.png")
            l_path = l_path_train if os.path.exists(l_path_train) else l_path_test
            
            if os.path.exists(l_path):
                lesion_mask = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)
                if lesion_mask.shape[:2] != master_mask.shape[:2]:
                    lesion_mask = cv2.resize(lesion_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                master_mask = cv2.bitwise_or(master_mask, lesion_mask)

        # Save Master Mask
        mask_filename = f"{img_id}.png"
        dest_mask_path = os.path.join(MASK_OUT_DIR, mask_filename)
        cv2.imwrite(dest_mask_path, master_mask)
        
        # Append to CSV tracking
        final_csv_data.append({
            'image_filename': original_filename,
            'mask_filename': mask_filename,
            'grade': grade
        })
        
        print(f"   [PROCESSED] {original_filename} (Grade {grade})")

    # 5. Save Reference CSV
    csv_out_path = os.path.join(OUTPUT_DIR, "sampled_reference.csv")
    df_final = pd.DataFrame(final_csv_data)
    df_final = df_final.sort_values(by='grade').reset_index(drop=True)
    df_final.to_csv(csv_out_path, index=False)
    
    print_step("COMPLETED")
    print_success(f"Output directory    : {OUTPUT_DIR}/")
    print_success(f"Images saved to     : {IMG_OUT_DIR}/")
    print_success(f"Masks saved to      : {MASK_OUT_DIR}/")
    print_success(f"Reference CSV saved : {csv_out_path}")

if __name__ == "__main__":
    main()