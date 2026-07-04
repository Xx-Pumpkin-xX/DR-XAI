import cv2
import torch
import numpy as np
from scipy.stats import rankdata, spearmanr

# ==========================================
# MATHEMATICAL UTILITIES
# ==========================================
def min_max_normalize(heatmap):
    hm = heatmap - np.min(heatmap)
    return hm / (np.max(hm) + 1e-8)

def calculate_spearman(map1, map2):
    corr, _ = spearmanr(map1.flatten(), map2.flatten())
    return 0.0 if np.isnan(corr) else corr

# ==========================================
# CCEM FUSION ALGORITHM
# ==========================================
def generate_ccem(G, AdaS, IG, tau=0.85):
    """
    Fuses 3 XAI heatmaps into a Consensus-Calibrated Explanation Map.
    Applies rank transformation, majority gating, and disagreement penalty.
    """
    def robust_norm(M):
        p1, p99 = np.percentile(M, 1), np.percentile(M, 99)
        return (np.clip(M, p1, p99) - p1) / (p99 - p1 + 1e-8)
        
    def rank_map(M):
        ranks = rankdata(M.flatten(), method='average')
        return (ranks / len(ranks)).reshape(M.shape)
    
    Q_G, Q_AdaS, Q_IG = rank_map(robust_norm(G)), rank_map(robust_norm(AdaS)), rank_map(robust_norm(IG))
    
    # Custom Weighting: Grad-CAM++ (0.4), Ada-SISE (0.3), IG (0.3)
    S = 0.4 * Q_G + 0.3 * Q_AdaS + 0.3 * Q_IG
    
    # Majority Gate
    A = (((Q_G >= tau).astype(np.float32) + (Q_AdaS >= tau).astype(np.float32) + (Q_IG >= tau).astype(np.float32)) >= 2).astype(np.float32)
    
    # Disagreement Penalty
    D = np.std(np.stack([Q_G, Q_AdaS, Q_IG], axis=0), axis=0)
    D_hat = D / (np.max(D) + 1e-8)
    
    C_final = cv2.GaussianBlur(S * A * (1.0 - D_hat), (5, 5), 0)
    return min_max_normalize(C_final)

# ==========================================
# ODExAI EVALUATION METRICS
# ==========================================
def calculate_advanced_metrics(model, image_tensor, heatmap, mask_binary):
    """
    Calculates Localization, Faithfulness, and Complexity metrics 
    based on the ODExAI framework (arXiv:2504.19249).
    """
    # 1. LOCALIZATION: EBPG (Energy-Based Pointing Game)
    ebpg = np.sum(heatmap * mask_binary) / (np.sum(heatmap) + 1e-8)
    
    # 2. LOCALIZATION: PG (Pointing Game)
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    pg = 1.0 if mask_binary[max_idx] > 0 else 0.0
    
    # 3. COMPLEXITY: Sparsity (Hoyer measure)
    N_pixels = heatmap.size
    l1_norm = np.sum(heatmap)
    l2_norm = np.sqrt(np.sum(heatmap**2))
    sparsity = (np.sqrt(N_pixels) - l1_norm / (l2_norm + 1e-8)) / (np.sqrt(N_pixels) - 1 + 1e-8)
    
    # 4. FAITHFULNESS: Deletion & Insertion
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1] 
    
    steps = np.linspace(0, 1, 11)
    del_scores = []
    ins_scores = []
    baseline_tensor = torch.zeros_like(image_tensor)
    
    with torch.no_grad():
        for step in steps:
            num_pixels = int(step * N_pixels)
            mask_1d = np.zeros(N_pixels, dtype=np.float32)
            mask_1d[sorted_indices[:num_pixels]] = 1.0
            mask_2d = mask_1d.reshape(heatmap.shape)
            mask_tensor = torch.tensor(mask_2d, device=image_tensor.device).unsqueeze(0).unsqueeze(0)
            
            # Deletion
            del_img = image_tensor * (1 - mask_tensor)
            del_val = model(del_img).item()
            del_scores.append(np.clip(del_val / 4.0, 0.0, 1.0))
            
            # Insertion
            ins_img = baseline_tensor * (1 - mask_tensor) + image_tensor * mask_tensor
            ins_val = model(ins_img).item()
            ins_scores.append(np.clip(ins_val / 4.0, 0.0, 1.0))
            
    del_auc = np.trapezoid(del_scores, steps)
    ins_auc = np.trapezoid(ins_scores, steps)
    
    # 5. FAITHFULNESS: Over-All (OA)
    oa = ins_auc - del_auc
    
    return ebpg, pg, sparsity, del_auc, ins_auc, oa