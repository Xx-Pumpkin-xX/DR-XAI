import torch
import torch.nn.functional as F
import numpy as np
import cv2

class ScoreCAMCore:
    """
    Core implementation of Score-CAM for Regression models.
    Gradient-free approach: Uses the model's forward pass confidence on masked 
    inputs to weight the activation maps.
    """
    def __init__(self, model, target_layer, input_size=(600, 600), batch_size=16, device='cuda'):
        self.model = model
        self.target_layer = target_layer
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        
        self.activations = []
        self.hook_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations.append(output.detach())

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()

    def generate_heatmap(self, input_tensor):
        """
        Generates the Score-CAM heatmap for a single input tensor.
        input_tensor: Shape (1, C, H, W)
        """
        self.activations = []
        self.model.eval()
        
        # 1. Forward pass to get activations
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        if not self.activations:
            raise ValueError("No activations captured. Check target layer.")
            
        # Shape: (1, Channels, H_feat, W_feat)
        activations = self.activations[0] 
        channels = activations.shape[1]
        
        # 2. Upsample activations to match input image size
        upsampled_activations = F.interpolate(
            activations, size=self.input_size, mode='bilinear', align_corners=False
        ).squeeze(0) # Shape: (Channels, H, W)
        
        # 3. Normalize each channel to [0, 1]
        max_vals = upsampled_activations.view(channels, -1).max(dim=1)[0].view(channels, 1, 1)
        min_vals = upsampled_activations.view(channels, -1).min(dim=1)[0].view(channels, 1, 1)
        normalized_activations = (upsampled_activations - min_vals) / (max_vals - min_vals + 1e-8)
        
        # 4. Create masked inputs and get scores in batches to prevent OOM
        scores = []
        with torch.no_grad():
            for i in range(0, channels, self.batch_size):
                end_idx = min(i + self.batch_size, channels)
                batch_masks = normalized_activations[i:end_idx].unsqueeze(1) # Shape: (B, 1, H, W)
                
                # Mask the original input
                masked_inputs = input_tensor * batch_masks
                
                # Forward pass the masked inputs
                batch_scores = self.model(masked_inputs) # Regression output (B, 1)
                scores.append(batch_scores.view(-1))
                
        scores = torch.cat(scores) # Shape: (Channels,)
        
        # 5. Weighted linear combination of activations
        scores = scores.view(channels, 1, 1)
        weighted_activations = (scores * upsampled_activations).sum(dim=0)
        
        # 6. Apply ReLU to keep only features that positively contribute to the score
        heatmap = F.relu(weighted_activations)
        
        # Final Min-Max normalization for the heatmap
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap.cpu().numpy()