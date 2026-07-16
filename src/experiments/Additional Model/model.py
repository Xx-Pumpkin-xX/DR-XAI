import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import WeightedRandomSampler

from backbones import build_encoder, get_backbone_config, get_img_size, resolve_arch

# ==========================================
# 1. MODULE CBAM (ATTENTION)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, max(in_planes // ratio, 1), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(max(in_planes // ratio, 1), in_planes, 1, bias=False)
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
# 2. GENERIC DR MODEL (ANY BACKBONE + CBAM OPTION)
# ==========================================
class DRModel(nn.Module):
    """
    Backbone-agnostic DR grading model.

        image -> encoder.forward_features -> [CBAM] -> pooled head -> Linear(1)

    Module names (`encoder`, `cbam`, `head`) are kept identical to the original
    APTOSModel, so every EfficientNet-B7 + CBAM checkpoint already produced by
    this project still loads with strict=True, and the XAI runners (which hook
    `model.cbam` / `model.encoder`) keep working untouched.

    Two details that matter when the backbone is no longer EfficientNet:

    1. Channel count is probed with a dummy forward instead of being read from
       `encoder.num_features`. For timm's MobileNetV3 those two numbers differ
       (960 spatial channels vs. a 1280-d head), and trusting `num_features`
       would build a CBAM and a Linear layer of the wrong width.

    2. Pooling is delegated to `encoder.forward_head(..., pre_logits=True)`, so
       each architecture contributes its own canonical pooled embedding
       (MobileNetV3 keeps its post-pool `conv_head`). For EfficientNet and
       DenseNet this is arithmetically the same global average pool the
       original code performed, which is what preserves checkpoint compatibility.
    """

    def __init__(
        self,
        arch="b7",
        num_classes=1,
        pretrained=True,
        use_cbam=False,
        chexnet_weights=None,
        img_size=None,
        cbam_ratio=16,
        verbose=True,
        require_chexnet=True,
    ):
        super(DRModel, self).__init__()

        self.arch = resolve_arch(arch)
        self.use_cbam = use_cbam

        self.encoder, self.backbone_cfg = build_encoder(
            self.arch,
            pretrained=pretrained,
            chexnet_weights=chexnet_weights,
            verbose=verbose,
            require_chexnet=require_chexnet,
        )
        self.model_name = self.backbone_cfg["timm_name"]
        self.img_size = get_img_size(self.arch, img_size)

        feat_channels, pooled_dim = self._probe_dims()
        self.feat_channels = feat_channels
        self.pooled_dim = pooled_dim

        if self.use_cbam:
            self.cbam = CBAM(feat_channels, ratio=cbam_ratio)

        self.head = nn.Linear(pooled_dim, num_classes)

        if verbose:
            print(
                f"[*] Init Model: {self.arch} ({self.model_name}) | CBAM: {self.use_cbam} | "
                f"init: {self.backbone_cfg.get('init')} | img={self.img_size} | "
                f"feat_ch={feat_channels} -> pooled={pooled_dim}"
            )

    @torch.no_grad()
    def _probe_dims(self):
        """Discover feature-map channels and pooled dimension for any backbone."""
        was_training = self.encoder.training
        self.encoder.eval()
        dummy = torch.zeros(1, 3, 224, 224)
        feats = self.encoder.forward_features(dummy)
        pooled = self._pool(feats)
        if was_training:
            self.encoder.train()
        return int(feats.shape[1]), int(pooled.shape[1])

    def _pool(self, features):
        """Architecture's own pooled embedding, with a global-average fallback."""
        try:
            pooled = self.encoder.forward_head(features, pre_logits=True)
            if pooled.ndim > 2:
                pooled = torch.flatten(pooled, 1)
            return pooled
        except (TypeError, AttributeError):
            return F.adaptive_avg_pool2d(features, 1).flatten(1)

    def forward_features(self, x):
        """Post-attention feature map. Used by the XAI hooks."""
        features = self.encoder.forward_features(x)
        if self.use_cbam:
            features = self.cbam(features)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        pooled = self._pool(features)
        return self.head(pooled)


# ==========================================
# 3. BACKWARDS-COMPATIBLE WRAPPER
# ==========================================
class APTOSModel(DRModel):
    """
    Legacy entry point. Existing calls of the form

        APTOSModel(model_name='tf_efficientnet_b7_ns', num_classes=1, use_cbam=True)

    continue to work unchanged (train_stage1/2/3, all XAI runners). New code
    should prefer DRModel(arch=...) or build_model(...).
    """

    _TIMM_TO_ARCH = {
        cfg["timm_name"]: key for key, cfg in
        [(k, get_backbone_config(k)) for k in
         ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "densenet121", "mobilenetv3", "mobilenetv3_s"]]
    }

    def __init__(self, model_name='efficientnet_b0', num_classes=1, pretrained=True,
                 use_cbam=False, **kwargs):
        arch = self._TIMM_TO_ARCH.get(model_name)

        if arch is None:
            # Unregistered timm name: honour it verbatim rather than guessing.
            nn.Module.__init__(self)
            self.arch = model_name
            self.model_name = model_name
            self.use_cbam = use_cbam
            self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.backbone_cfg = dict(timm_name=model_name, init="timm-default", key=model_name)
            self.img_size = kwargs.get("img_size") or 224
            feat_channels, pooled_dim = DRModel._probe_dims(self)
            self.feat_channels, self.pooled_dim = feat_channels, pooled_dim
            if use_cbam:
                self.cbam = CBAM(feat_channels)
            self.head = nn.Linear(pooled_dim, num_classes)
            print(f"[*] Init Model: {model_name} | Use CBAM: {use_cbam}")
            return

        super().__init__(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained,
            use_cbam=use_cbam,
            **kwargs,
        )


def build_model(arch, num_classes=1, pretrained=True, use_cbam=False,
                chexnet_weights=None, img_size=None, verbose=True,
                require_chexnet=True):
    """Single factory used by every training / evaluation script."""
    return DRModel(
        arch=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        use_cbam=use_cbam,
        chexnet_weights=chexnet_weights,
        img_size=img_size,
        verbose=verbose,
        require_chexnet=require_chexnet,
    )


def load_checkpoint(model, weight_path, device="cpu", strict=True):
    """Load a training checkpoint, tolerating DataParallel / wrapped state dicts."""
    state = torch.load(weight_path, map_location=device, weights_only=False)
    for key in ("state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(cleaned, strict=strict)
    return model


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ==========================================
# 4. LOSSES / SAMPLERS / METRICS (unchanged)
# ==========================================
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
