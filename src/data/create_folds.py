import sys, os
core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "0_Core")
if core_path not in sys.path: sys.path.append(core_path)

import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def create_folds(csv_path, output_path, n_splits=5):
    df = pd.read_csv(csv_path)
    
    df["fold"] = -1
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    group_col = 'id_code' 
    target_col = 'diagnosis'
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X=df, y=df[target_col], groups=df[group_col])):
        df.loc[val_idx, "fold"] = fold
        
    df.to_csv(output_path, index=False)
    
    print(f"[*] Splited to {n_splits} folds and saved at: {output_path}")
    
    print("\n--- Fold 0 grade distribution (Validation) ---")
    print(df[df["fold"] == 0][target_col].value_counts().sort_index())

if __name__ == "__main__":
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(THIS_DIR, "../../aptos2019-blindness-detection/train.csv")
    #input_csv = os.path.join(THIS_DIR, "sampled_120_stratified.csv")
    output_csv = os.path.join(THIS_DIR, "train_folds.csv")
    
    create_folds(input_csv, output_csv)