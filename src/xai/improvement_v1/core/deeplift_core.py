import cv2
import torch
import numpy as np
from captum.attr import DeepLift

class DeepLIFTCore:
    """
    Core implementation of DeepLIFT for Regression models.
    Computes attributions by comparing the activation of each neuron to its 
    'reference' (baseline) activation.
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        # Initialize Captum's DeepLift explainer
        self.explainer = DeepLift(self.model)

    def generate_heatmap(self, input_tensor, baseline_mode='blurred'):
        """
        Generates the DeepLIFT heatmap.
        baseline_mode: 'zeros' (black image) or 'blurred' (lesion-free retinal background).
        """
        input_tensor.requires_grad_()
        
        # Create the Reference/Baseline Image
        if baseline_mode == 'blurred':
            # Extract the numpy image from the tensor for blurring
            img_np = input_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            
            # Apply extreme Gaussian Blur to erase small lesions (Microaneurysms, Hemorrhages)
            # This creates a "healthy" version of the exact same retina as the baseline
            blurred_img = cv2.GaussianBlur(img_np, (99, 99), 0)
            
            # Convert back to tensor
            baseline_tensor = torch.tensor(blurred_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        else:
            # Standard black image baseline
            baseline_tensor = torch.zeros_like(input_tensor).to(self.device)

        # Generate Attribution using Captum
        # Target is 0 because we are using a regression model (1 output node)
        attributions = self.explainer.attribute(input_tensor, baselines=baseline_tensor, target=0)
        
        # Reduce color channels by taking the sum of absolute attribution values
        heatmap = torch.sum(torch.abs(attributions), dim=1).squeeze(0).cpu().detach().numpy()
        
        # Normalize the heatmap to [0, 1] range
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        # Apply a mild smoothing to reduce attribution noise (shattered gradients)
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        return heatmap