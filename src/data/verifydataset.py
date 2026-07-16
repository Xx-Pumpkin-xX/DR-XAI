import sys, os
core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "0_Core")
if core_path not in sys.path: sys.path.append(core_path)

import os
import glob
import cv2
from tqdm import tqdm

def verify_messidor_maples_final():


    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(THIS_DIR, "../../datasets1/MAPLES_DR_198_Pack/images")
    MASK_DIR = os.path.abspath(os.path.join(THIS_DIR, "../../MAPLES-DR"))

    if not os.path.exists(DATA_DIR) or not os.path.exists(MASK_DIR):
        print("Wrong org path!")
        return

    mask_paths = glob.glob(os.path.join(MASK_DIR, "**", "*.png"), recursive=True)
    mask_stems = set(os.path.splitext(os.path.basename(p))[0] for p in mask_paths)
    print(f"[*] Total mask in MAPLES-DR : {len(mask_stems)} masks.")

    img_paths = []
    for ext in ('*.tif', '*.tiff', '*.png', '*.jpg'):
        img_paths.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))
    
    img_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}
    print(f"[*] Total scanned fundus: {len(img_dict)} imgs.\n")

    matched_stems = mask_stems.intersection(img_dict.keys())
    missing_masks_from_maples = mask_stems - img_dict.keys()
    images_without_mask = set(img_dict.keys()) - mask_stems


    print(f" Successful match : {len(matched_stems)}")
    print(f" No mask  : {len(images_without_mask)} imgs (Normal)")
    
    if len(missing_masks_from_maples) > 0:
        print(f"{len(missing_masks_from_maples)} mask without match imgs!")
        print(f"    Ví dụ: {list(missing_masks_from_maples)[:3]}")

    corrupted = 0
    for stem in tqdm(matched_stems, desc="integrity check"):
        if cv2.imread(img_dict[stem]) is None:
            corrupted += 1

    print("\n" + "="*75)
    if len(matched_stems) == len(mask_stems) and corrupted == 0:
        print(f"100% match ({len(matched_stems)}/{len(mask_stems)}) DOCTOR MASKS!")
    else:
        print("Not matching!")
    print("="*75 + "\n")

if __name__ == "__main__":
    verify_messidor_maples_final()