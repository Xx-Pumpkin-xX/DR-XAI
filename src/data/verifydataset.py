import sys, os
core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "0_Core")
if core_path not in sys.path: sys.path.append(core_path)

import os
import glob
import cv2
from tqdm import tqdm

def verify_messidor_maples_final():
    print("\n" + "="*75)
    print(" 🕵️ HỆ THỐNG ĐỐI CHIẾU CHUẨN: MESSIDOR-1 vs MAPLES-DR MASKS ")
    print("="*75)

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(THIS_DIR, "../../datasets1/MAPLES_DR_198_Pack/images")
    MASK_DIR = os.path.abspath(os.path.join(THIS_DIR, "../../MAPLES-DR"))

    if not os.path.exists(DATA_DIR) or not os.path.exists(MASK_DIR):
        print("❌ Lỗi sai đường dẫn gốc. Vui lòng kiểm tra lại!")
        return

    # 1. Quét toàn bộ Mask và lấy đúng Tên Gốc (bỏ đuôi .png)
    mask_paths = glob.glob(os.path.join(MASK_DIR, "**", "*.png"), recursive=True)
    mask_stems = set(os.path.splitext(os.path.basename(p))[0] for p in mask_paths)
    print(f"[*] Tổng số Mask bác sĩ có trong MAPLES-DR : {len(mask_stems)} masks.")

    # 2. Quét toàn bộ ảnh Messidor thực tế đang có
    img_paths = []
    for ext in ('*.tif', '*.tiff', '*.png', '*.jpg'):
        img_paths.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))
    
    img_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}
    print(f"[*] Tổng số ảnh võng mạc quét được        : {len(img_dict)} ảnh.\n")

    # 3. Đối chiếu giao tập hợp (Intersection)
    matched_stems = mask_stems.intersection(img_dict.keys())
    missing_masks_from_maples = mask_stems - img_dict.keys()
    images_without_mask = set(img_dict.keys()) - mask_stems

    print("-" * 75)
    print(" 📊 KẾT QUẢ SO KHỚP THỰC TẾ")
    print("-" * 75)
    print(f" ✅ Số cặp Ảnh <-> Mask ghép cặp thành công : {len(matched_stems)} cặp")
    print(f" ℹ️ Số ảnh võng mạc không có Mask đi kèm  : {len(images_without_mask)} ảnh (Bình thường)")
    
    if len(missing_masks_from_maples) > 0:
        print(f" ⚠️ Có {len(missing_masks_from_maples)} mask không tìm thấy ảnh võng mạc tương ứng!")
        print(f"    Ví dụ: {list(missing_masks_from_maples)[:3]}")

    # 4. Kiểm tra nhanh độ hỏng của các ảnh đã ghép cặp
    corrupted = 0
    for stem in tqdm(matched_stems, desc="Kiểm tra độ toàn vẹn cặp dữ liệu"):
        if cv2.imread(img_dict[stem]) is None:
            corrupted += 1

    print("\n" + "="*75)
    if len(matched_stems) == len(mask_stems) and corrupted == 0:
        print(f" 🏆 TUYỆT VỜI! BẠN ĐÃ GHÉP KHỚP THÀNH CÔNG 100% ({len(matched_stems)}/{len(mask_stems)}) DOCTOR MASKS! 🏆 ")
        print(" Bạn đã sẵn sàng để viết DataLoader cho mô hình phân đoạn (Segmentation)!")
    else:
        print(" ❗ Dữ liệu khớp chưa hoàn toàn, hãy kiểm tra lại các thông số trên. ❗ ")
    print("="*75 + "\n")

if __name__ == "__main__":
    verify_messidor_maples_final()