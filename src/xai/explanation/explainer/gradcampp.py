import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class CompactGradCAMPlusPlus:
    """
    Core wrapper for Grad-CAM++ to maintain a consistent API across all explainers.
    It automatically determines the optimal target layer (CBAM or final Conv block)
    and generates the raw heatmap.
    """
    def __init__(self, model, use_cbam=False):
        self.model = model
        self.use_cbam = use_cbam
        
        # Determine the target layer dynamically based on the architecture
        if self.use_cbam and hasattr(model, 'cbam'):
            self.target_layers = [model.cbam]
        elif hasattr(model.encoder, 'conv_head'):
            self.target_layers = [model.encoder.conv_head]
        else:
            # Fallback for other timm models
            self.target_layers = [list(model.encoder.children())[-2]]
            
        self.cam_explainer = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers)

    def generate_heatmap(self, input_tensor, target_class=0):
        """
        Generates the raw Grad-CAM++ heatmap.
        Returns:
            raw_heatmap (numpy.ndarray): The 2D heatmap array.
            target_class (int): The class index used for attribution.
        """
        targets = [ClassifierOutputTarget(target_class)]
        
        # pytorch_grad_cam processes batches, we extract the first image [0, :]
        raw_heatmap = self.cam_explainer(input_tensor=input_tensor, targets=targets)[0, :]
        
        return raw_heatmap, target_class

    def get_target_layer_name(self):
        """Returns the name of the layer being targeted for gradients."""
        return self.target_layers[0].__class__.__name__