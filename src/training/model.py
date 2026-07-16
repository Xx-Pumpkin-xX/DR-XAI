import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import WeightedRandomSampler

# ==========================================
# 1. MODULE CBAM (ATTENTION)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ==========================================
# 2. MODEL Structure (CÓ CBAM OPTION)
# ==========================================
class APTOSModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=1, pretrained=True, use_cbam=False):
        super(APTOSModel, self).__init__()
        self.model_name = model_name
        self.use_cbam = use_cbam
        
        print(f"[*] Init Model: {model_name} | Use CBAM: {self.use_cbam}")
        
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.encoder.num_features
        if self.use_cbam:
            self.cbam = CBAM(num_features)
            
        self.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.encoder.forward_features(x)
        if self.use_cbam:
            features = self.cbam(features)
            
        pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        out = self.head(pooled)
        return out

def calculate_qwk(y_true, y_pred):
    if torch.is_tensor(y_true): y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred): y_pred = y_pred.cpu().numpy()
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def get_class_weights(df, target_col='diagnosis'):
    class_counts = df[target_col].value_counts().sort_index().values
    total_samples = np.sum(class_counts)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

def create_balanced_sampler(df, target_col='diagnosis'):
    class_counts = df[target_col].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in df[target_col]]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights), 
        replacement=True
    )
    return sampler