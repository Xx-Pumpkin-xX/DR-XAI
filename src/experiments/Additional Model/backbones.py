"""
backbones.py
============
Central registry of every classification backbone used in the DR-grading study.

The paper compares four architectures under ONE identical pipeline
(same CLAHE preprocessing, same augmentation, same SmoothL1 ordinal-regression
head, same TTA, same OptimizedRounder, same fold). Only the encoder changes.

    Key            | Encoder                     | Init
    ---------------|-----------------------------|------------------------------
    b0..b7         | EfficientNet-B0..B7 (NS)    | ImageNet / NoisyStudent
    chexnet        | DenseNet-121                | CheXNet (ChestX-ray14)
    densenet121    | DenseNet-121                | ImageNet
    mobilenetv3    | MobileNetV3-Large 1.0       | ImageNet
    mobilenetv3_s  | MobileNetV3-Small 1.0       | ImageNet

Matched-input protocol
----------------------
`default_img_size` is 600 for every backbone. DenseNet and MobileNetV3 are
fully convolutional and accept 600x600 without modification. Feeding the
baselines their native 224 px while EfficientNet-B7 receives 600 px would
confound resolution with architecture, and a reviewer would say so. Run the
whole comparison at one resolution; use `--img_size` only for a deliberate
resolution ablation.
"""

import os
import re
from collections import OrderedDict

import timm
import torch


# ==========================================================================
# 1. REGISTRY
# ==========================================================================
BACKBONES = {
    # --- EfficientNet family (the proposed model lives here) ---------------
    "b0": dict(timm_name="tf_efficientnet_b0_ns", native_img_size=224, default_img_size=600, family="efficientnet"),
    "b1": dict(timm_name="tf_efficientnet_b1_ns", native_img_size=240, default_img_size=600, family="efficientnet"),
    "b2": dict(timm_name="tf_efficientnet_b2_ns", native_img_size=260, default_img_size=600, family="efficientnet"),
    "b3": dict(timm_name="tf_efficientnet_b3_ns", native_img_size=300, default_img_size=600, family="efficientnet"),
    "b4": dict(timm_name="tf_efficientnet_b4_ns", native_img_size=380, default_img_size=600, family="efficientnet"),
    "b5": dict(timm_name="tf_efficientnet_b5_ns", native_img_size=456, default_img_size=600, family="efficientnet"),
    "b6": dict(timm_name="tf_efficientnet_b6_ns", native_img_size=528, default_img_size=600, family="efficientnet"),
    "b7": dict(timm_name="tf_efficientnet_b7_ns", native_img_size=600, default_img_size=600, family="efficientnet"),

    # --- Baselines ---------------------------------------------------------
    "chexnet": dict(
        timm_name="densenet121",
        native_img_size=224,
        default_img_size=600,
        family="densenet",
        requires_external_weights=True,
        paper_name="CheXNet (DenseNet-121, ChestX-ray14 init)",
    ),
    "densenet121": dict(
        timm_name="densenet121",
        native_img_size=224,
        default_img_size=600,
        family="densenet",
        paper_name="DenseNet-121 (ImageNet init)",
    ),
    "mobilenetv3": dict(
        timm_name="mobilenetv3_large_100",
        native_img_size=224,
        default_img_size=600,
        family="mobilenet",
        paper_name="MobileNetV3-Large",
    ),
    "mobilenetv3_s": dict(
        timm_name="mobilenetv3_small_100",
        native_img_size=224,
        default_img_size=600,
        family="mobilenet",
        paper_name="MobileNetV3-Small",
    ),
}

# Legacy alias table so `--arch B7` and `--model B7` both keep working.
ALIASES = {
    "efficientnet_b7": "b7", "effnet_b7": "b7", "efficientnetb7": "b7",
    "densenet": "densenet121", "dense121": "densenet121", "densenet-121": "densenet121",
    "chex": "chexnet", "chexnet121": "chexnet", "chex-net": "chexnet",
    "mobilenet": "mobilenetv3", "mobilenetv3_large": "mobilenetv3",
    "mobilenet_v3": "mobilenetv3", "mobilenetv3l": "mobilenetv3",
    "mobilenetv3_small": "mobilenetv3_s",
}


def resolve_arch(name):
    """Normalise a user-supplied architecture string to a registry key."""
    key = str(name).strip().lower().replace(" ", "")
    key = ALIASES.get(key, key)
    if key not in BACKBONES:
        raise ValueError(
            f"Unknown architecture '{name}'. Available: {sorted(BACKBONES.keys())}"
        )
    return key


def get_backbone_config(name):
    key = resolve_arch(name)
    cfg = dict(BACKBONES[key])
    cfg["key"] = key
    cfg.setdefault("paper_name", key)
    cfg.setdefault("requires_external_weights", False)
    return cfg


def get_img_size(name, override=None):
    """Resolution used for this backbone. `override` wins when given."""
    if override is not None:
        return int(override)
    return get_backbone_config(name)["default_img_size"]


# ==========================================================================
# 2. CHEXNET WEIGHT LOADING
# ==========================================================================
# CheXNet is architecturally DenseNet-121. What makes it CheXNet is the
# ChestX-ray14 pre-training. Without those weights, `--arch chexnet` is
# byte-for-byte identical to `--arch densenet121` and the row is meaningless
# in the paper, so this loader is strict by default.
#
# Where to get a checkpoint:
#   * arnoweng/CheXNet            -> model.pth.tar
#   * jrzech/reproduce-chexnet    -> checkpoint (PyTorch)
#
# Both were saved from torchvision's DenseNet under nn.DataParallel, so their
# keys look like:
#     module.densenet121.features.denseblock1.denselayer1.norm.1.weight
# while timm expects:
#     features.denseblock1.denselayer1.norm1.weight
# The remapper below closes that gap and then reports coverage, so a silent
# "loaded 3 of 364 tensors" can never slip into the results table.
# ==========================================================================

_TORCHVISION_DENSENET_PATTERN = re.compile(
    r"^(.*denselayer\d+\.(?:norm|relu|conv))\.(\d+)\.(.*)$"
)


def _remap_chexnet_key(key):
    """One raw CheXNet checkpoint key -> one timm densenet121 key (or None)."""
    k = key

    for prefix in ("module.", "model.", "densenet121.", "densenet.", "net."):
        while k.startswith(prefix):
            k = k[len(prefix):]

    # The 14-class chest head is discarded; we attach our own DR head.
    if k.startswith("classifier"):
        return None

    # `norm.1` / `conv.2` (old torchvision) -> `norm1` / `conv2` (timm)
    m = _TORCHVISION_DENSENET_PATTERN.match(k)
    if m:
        k = f"{m.group(1)}{m.group(2)}.{m.group(3)}"

    return k


def load_chexnet_weights(encoder, weight_path, verbose=True, min_coverage=0.80):
    """
    Load ChestX-ray14 (CheXNet) DenseNet-121 weights into a timm encoder.

    Returns a dict of coverage statistics. Raises if fewer than `min_coverage`
    of the encoder's parameter tensors were actually populated.
    """
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"CheXNet checkpoint not found: {weight_path}\n"
            "Download model.pth.tar from arnoweng/CheXNet (or the equivalent "
            "checkpoint from jrzech/reproduce-chexnet) and pass it via "
            "--chexnet_weights."
        )

    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)

    for k in ("state_dict", "model_state_dict", "model"):
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            ckpt = ckpt[k]
            break

    target = encoder.state_dict()
    remapped = OrderedDict()
    skipped_shape = []

    for raw_key, value in ckpt.items():
        new_key = _remap_chexnet_key(raw_key)
        if new_key is None or new_key not in target:
            continue
        if tuple(target[new_key].shape) != tuple(value.shape):
            skipped_shape.append(new_key)
            continue
        remapped[new_key] = value

    missing, unexpected = encoder.load_state_dict(remapped, strict=False)

    n_target = len(target)
    n_loaded = len(remapped)
    coverage = n_loaded / max(n_target, 1)

    stats = dict(
        loaded=n_loaded,
        total=n_target,
        coverage=coverage,
        missing=len(missing),
        shape_mismatch=len(skipped_shape),
    )

    if verbose:
        print(f"   [CheXNet] Source checkpoint : {weight_path}")
        print(f"   [CheXNet] Tensors populated : {n_loaded}/{n_target} ({coverage:.1%})")
        if skipped_shape:
            print(f"   [CheXNet] Shape mismatches  : {len(skipped_shape)} (e.g. {skipped_shape[:3]})")
        if missing:
            print(f"   [CheXNet] Left at ImageNet  : {len(missing)} (e.g. {list(missing)[:3]})")

    if coverage < min_coverage:
        raise RuntimeError(
            f"CheXNet weights only covered {coverage:.1%} of DenseNet-121 "
            f"(threshold {min_coverage:.0%}). The checkpoint layout was not "
            "recognised, so this run would report an ImageNet DenseNet under "
            "the CheXNet label. Inspect the checkpoint keys before continuing."
        )

    return stats


# ==========================================================================
# 3. ENCODER FACTORY
# ==========================================================================
def build_encoder(arch, pretrained=True, chexnet_weights=None, verbose=True,
                  require_chexnet=True):
    """
    Return (encoder, config). The encoder is a headless timm model
    (`num_classes=0`) exposing `forward_features` and `forward_head`.

    `require_chexnet=False` is for evaluation, where every weight comes from the
    trained checkpoint and the original ChestX-ray14 file is no longer needed.
    """
    cfg = get_backbone_config(arch)

    # For CheXNet we still start from ImageNet so that any tensor the CheXNet
    # checkpoint does not carry is at least sensibly initialised; the loader
    # below then overwrites the chest-pretrained ones and reports coverage.
    encoder = timm.create_model(cfg["timm_name"], pretrained=pretrained, num_classes=0)

    if cfg["key"] == "chexnet":
        if chexnet_weights:
            cfg["chexnet_stats"] = load_chexnet_weights(encoder, chexnet_weights, verbose=verbose)
            cfg["init"] = "ChestX-ray14 (CheXNet)"
        elif not require_chexnet:
            cfg["init"] = "from checkpoint"
        else:
            raise ValueError(
                "--arch chexnet requires --chexnet_weights.\n"
                "CheXNet and DenseNet-121 are the same architecture; only the "
                "pre-training differs. Running 'chexnet' on ImageNet weights "
                "would duplicate the densenet121 row under a different name.\n"
                "To deliberately run that ablation, use --arch densenet121."
            )
    else:
        cfg["init"] = "ImageNet/NoisyStudent" if pretrained else "random"

    return encoder, cfg
