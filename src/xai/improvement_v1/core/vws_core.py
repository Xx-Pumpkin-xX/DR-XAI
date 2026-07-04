import numpy as np
import cv2

class VWSCore:
    """
    Variance-Weighted Saliency (VWS) Fusion Algorithm.
    A soft-weighting consensus mechanism that avoids harsh gating thresholds.
    Pixels with high disagreement (variance) across methods receive lower weights,
    preserving subtle lesion details while suppressing isolated noise.
    """
    def __init__(self, eps=1e-8):
        self.eps = eps

    def _normalize(self, heatmap):
        """Standard Min-Max Normalization to [0, 1]"""
        hm = heatmap - np.min(heatmap)
        return hm / (np.max(hm) + self.eps)

    def generate(self, map_s, map_a, map_i):
        """
        Fuses three explanation maps using Variance-Weighted Saliency.
        map_s: Score-CAM heatmap
        map_a: Ada-SISE heatmap
        map_i: SmoothIG heatmap
        """
        # 1. Ensure all maps are strictly in [0, 1]
        M_S = self._normalize(map_s)
        M_A = self._normalize(map_a)
        M_I = self._normalize(map_i)

        # 2. Stack maps to compute pixel-wise statistics
        stack = np.stack([M_S, M_A, M_I], axis=0)
        
        # 3. Compute Variance across the 3 methods
        variance = np.var(stack, axis=0)
        
        # Normalize variance to [0, 1] to use as a penalty factor
        var_norm = self._normalize(variance)
        
        # 4. Calculate Weights (Inverse of variance)
        # Low variance (high agreement) -> Weight approaches 1.0
        # High variance (disagreement) -> Weight approaches 0.0
        weights = 1.0 - var_norm
        
        # 5. Weighted Average Fusion
        # Multiply each map by the consensus weight
        weighted_sum = (M_S * weights) + (M_A * weights) + (M_I * weights)
        
        # Divide by the sum of weights (3 * weights since weights are shared pixel-wise)
        vws_map = weighted_sum / (3.0 * weights + self.eps)
        
        # 6. Final Smoothing and Normalization for visualization
        vws_map = cv2.GaussianBlur(vws_map.astype(np.float32), (5, 5), 0)
        vws_map = self._normalize(vws_map)
        
        return vws_map.astype(np.float32)