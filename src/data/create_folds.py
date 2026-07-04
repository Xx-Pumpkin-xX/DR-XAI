import sys, os
core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "0_Core")
if core_path not in sys.path: sys.path.append(core_path)

import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def create_folds(csv_path, output_path, n_splits=5):
    # 1. Đọc dữ liệu
    df = pd.read_csv(csv_path)
    
    # 2. Khởi tạo cột fold với giá trị -1
    df["fold"] = -1
    
    # 3. Cấu hình StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # XÁC ĐỊNH CỘT GROUP:
    # Nếu bạn có cột 'patient_id', hãy thay 'id_code' bằng 'patient_id'
    group_col = 'id_code' 
    target_col = 'diagnosis'
    
    # 4. Thực hiện chia fold
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X=df, y=df[target_col], groups=df[group_col])):
        df.loc[val_idx, "fold"] = fold
        
    # 5. Lưu lại file CSV đã chia fold
    df.to_csv(output_path, index=False)
    
    print(f"[*] Đã chia thành {n_splits} folds và lưu tại: {output_path}")
    
    # Kiểm tra phân bố ở Fold 0 để đảm bảo chia đúng
    print("\n--- Phân bố nhãn ở Fold 0 (Tập Validation) ---")
    print(df[df["fold"] == 0][target_col].value_counts().sort_index())

if __name__ == "__main__":
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(THIS_DIR, "../../aptos2019-blindness-detection/train.csv")
    #input_csv = os.path.join(THIS_DIR, "sampled_120_stratified.csv")
    output_csv = os.path.join(THIS_DIR, "train_folds.csv")
    
    create_folds(input_csv, output_csv)