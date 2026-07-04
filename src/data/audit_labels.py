import sys, os
core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "0_Core")
if core_path not in sys.path: sys.path.append(core_path)

import os
import glob
import pandas as pd

# ==========================================
# CONFIGURATIONS
# ==========================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MAPLES_DIR = os.path.join(THIS_DIR, "../../MAPLES-DR")
OUTPUT_CSV = os.path.join(THIS_DIR, "audited_labels_198.csv")

# ==========================================
# EXPLICIT AND SAFE MAPPING FUNCTION
# ==========================================
def safe_map_messidor_label(dr_str):
    """
    Explicitly maps MESSIDOR strings to APTOS integers.
    This guarantees 100% accuracy without unintended side effects.
    """
    dr_str = str(dr_str).strip().upper()
    mapping = {
        'R0': 0,
        'R1': 1,
        'R2': 2,
        'R3': 3,
        'R4': 4,
        'R4A': 4,
        'R4B': 4
    }
    # Return -1 if a weird label is encountered
    return mapping.get(dr_str, -1) 

# ==========================================
# MAIN AUDIT EXECUTION
# ==========================================
def main():
    print("="*60)
    print(" LABEL MAPPING AUDIT: MESSIDOR -> APTOS ")
    print("="*60)

    csv_files = glob.glob(os.path.join(MAPLES_DIR, "**", "diagnosis.csv"), recursive=True)
    
    if not csv_files:
        print("[FAILED] No diagnosis.csv files found in MAPLES-DR.")
        return

    all_data = []

    for csv_path in csv_files:
        subset = "Train" if "train" in csv_path.lower() else "Test"
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            img_id = str(row['name']).strip()
            raw_dr = str(row['DR']).strip().upper()
            raw_me = str(row.get('ME', 'Unknown')).strip().upper()
            
            mapped_aptos = safe_map_messidor_label(raw_dr)
            
            all_data.append({
                'Image_ID': img_id,
                'Subset': subset,
                'Raw_MESSIDOR_DR': raw_dr,
                'Raw_MESSIDOR_ME': raw_me,
                'Mapped_APTOS_Grade': mapped_aptos
            })

    audit_df = pd.DataFrame(all_data)
    total_images = len(audit_df)

    print(f"[INFO] Scanned {len(csv_files)} CSV files.")
    print(f"[INFO] Total images found: {total_images}")

    # Check for mapping errors
    errors = audit_df[audit_df['Mapped_APTOS_Grade'] == -1]
    if len(errors) > 0:
        print("\n[CRITICAL WARNING] Found unrecognized labels!")
        print(errors)
    else:
        print("[SUCCESS] No mapping errors detected.")

    print("\n" + "-"*60)
    print(" CROSS-TABULATION (RAW vs MAPPED) ")
    print("-" * 60)
    
    # Generate a cross-tabulation table to prove the mapping is correct
    cross_tab = pd.crosstab(audit_df['Raw_MESSIDOR_DR'], audit_df['Mapped_APTOS_Grade'], margins=True)
    print(cross_tab)

    print("-" * 60)
    
    # Save the detailed audit list to CSV
    audit_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Full detailed list saved to: {OUTPUT_CSV}")
    print("="*60)

if __name__ == "__main__":
    main()