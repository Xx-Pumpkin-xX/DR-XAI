import sys, os
core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "0_Core")
if core_path not in sys.path: sys.path.append(core_path)

import os
import glob
import shutil
import zipfile
import pandas as pd
from tqdm import tqdm

# ==========================================
# CONFIGURATIONS
# ==========================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MAPLES_DIR = os.path.join(THIS_DIR, "../../MAPLES-DR")
MESSIDOR_IMG_DIR = os.path.join(THIS_DIR, "../../datasets1")

EXPORT_DIR = os.path.join(THIS_DIR, "Blind_Grading_198")
ZIP_FILENAME = os.path.join(THIS_DIR, "Blind_Grading_Package.zip")

# ==========================================
# HELPER FUNCTION
# ==========================================
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
    print("="*70)
    print(" PREPARING BLIND GRADING PACKAGE FOR CLINICAL TEAM ")
    print("="*70)

    # 1. Setup export directory
    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    img_export_dir = os.path.join(EXPORT_DIR, "images")
    os.makedirs(img_export_dir, exist_ok=True)

    # 2. Get the exact 198 IDs from MAPLES-DR masks
    print("[INFO] Scanning for 198 MAPLES-DR IDs...")
    mask_files = glob.glob(os.path.join(MAPLES_DIR, "train", "Microaneurysms", "*.png")) + \
                 glob.glob(os.path.join(MAPLES_DIR, "test", "Microaneurysms", "*.png"))
                 
    base_names = set([os.path.basename(p).replace('.png', '') for p in mask_files])
    
    if len(base_names) == 0:
        print("[FAILED] No IDs found. Check MAPLES-DR path.")
        return

    # 3. Copy images and prepare data for grading template
    grading_data = []
    copied_count = 0

    print("[INFO] Extracting raw images...")
    for base_name in tqdm(base_names, desc="Copying"):
        img_path = get_exact_image_path(base_name, MESSIDOR_IMG_DIR)
        
        if img_path:
            # Copy image to export folder
            ext = os.path.splitext(img_path)[1]
            new_img_name = f"{base_name}{ext}"
            new_img_path = os.path.join(img_export_dir, new_img_name)
            shutil.copy2(img_path, new_img_path)
            
            # Prepare row for grading template based on clinical criteria
            grading_data.append({
                "Image_ID": new_img_name,
                "Microaneurysm (>=1 or Many)": "",
                "Hemorrhage": "",
                "Exudate (Hard/Soft)": "",
                "IRMA": "",
                "Neovascularization": "",
                "FINAL_GRADE (0-4)": "",
                "Clinical_Notes_and_Blindness_Risk": ""
            })
            copied_count += 1
        else:
            print(f"[WARNING] Image not found for ID: {base_name}")

    # 4. Generate the Clinical Grading Template (CSV & Excel)
    print("\n[INFO] Generating Clinical Grading Template...")
    df_grading = pd.DataFrame(grading_data)
    
    # Sort alphabetically by Image ID for convenience
    df_grading = df_grading.sort_values(by="Image_ID").reset_index(drop=True)
    
    csv_path = os.path.join(EXPORT_DIR, "Clinical_Grading_Template.csv")
    df_grading.to_csv(csv_path, index=False)

    # 5. Zip everything together for easy download
    print("[INFO] Zipping package for download...")
    with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(EXPORT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, EXPORT_DIR)
                zipf.write(file_path, arcname)

    print("\n" + "="*70)
    print(f" [SUCCESS] Extracted {copied_count} images.")
    print(f" [SUCCESS] Package ready for download: {ZIP_FILENAME}")
    print("="*70)

if __name__ == "__main__":
    main()