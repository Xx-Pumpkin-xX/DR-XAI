"""
train_baseline.py
=================
One trainer, four backbones, one protocol.

This is the script that produces every row of the architecture-comparison table.
It reproduces the exact recipe already used for EfficientNet-B7 + CBAM
(SmoothL1 ordinal regression, CLAHE, 3-view TTA, Nelder-Mead OptimizedRounder,
fold 0 of train_folds.csv) and swaps only the encoder, so any difference in the
final QWK is attributable to the architecture rather than to the pipeline.

Stage 1 (optional but recommended): pre-train on EyePACS 2015.
Stage 2 (required):                 fine-tune on APTOS 2019 fold 0, or MESSIDOR.

Examples
--------
# Baseline 1 - CheXNet (DenseNet-121 initialised from ChestX-ray14)
python src/training/train_baseline.py --arch chexnet --stage 1 \
    --chexnet_weights weights/chexnet_model.pth.tar --batch_size 16 --epochs 10
python src/training/train_baseline.py --arch chexnet --stage 2 \
    --pretrain_path src/experiments/Stage1_chexnet/stage1_best.pth \
    --batch_size 16 --epochs 20

# Baseline 2 - DenseNet-121 (ImageNet)
python src/training/train_baseline.py --arch densenet121 --stage 1 --batch_size 16 --epochs 10
python src/training/train_baseline.py --arch densenet121 --stage 2 \
    --pretrain_path src/experiments/Stage1_densenet121/stage1_best.pth --epochs 20

# Baseline 3 - MobileNetV3-Large
python src/training/train_baseline.py --arch mobilenetv3 --stage 1 --batch_size 24 --epochs 10
python src/training/train_baseline.py --arch mobilenetv3 --stage 2 \
    --pretrain_path src/experiments/Stage1_mobilenetv3/stage1_best.pth --epochs 20

# Proposed model, same script (and the no-CBAM ablation)
python src/training/train_baseline.py --arch b7 --use_cbam --stage 2 \
    --pretrain_path src/experiments/Stage1_Pretrain_B7/stage1_best.pth --epochs 20
python src/training/train_baseline.py --arch b7 --stage 2 \
    --pretrain_path src/experiments/Stage1_Pretrain_B7/stage1_best.pth --epochs 20
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from backbones import get_backbone_config, get_img_size, resolve_arch
from model import build_model, calculate_qwk, count_parameters, load_checkpoint
from rounder import OptimizedRounder
from pipeline import (
    get_train_transforms, make_dataset, make_scaler, predict_with_tta,
    print_error_and_exit, print_step, print_success, print_warning,
    save_run_config, train_one_epoch,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# NEW
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))

# ==========================================================================
# DATA LOADING PER DATASET
# ==========================================================================
def load_eyepacs(project_root):
    csv_path = os.path.join(project_root, "datasets", "EyePACS_2015", "trainLabels.csv")
    img_dir = os.path.join(project_root, "datasets", "EyePACS_2015", "train_images")
    if not os.path.exists(csv_path):
        print_error_and_exit("EyePACS labels not found.", csv_path)

    df = pd.read_csv(csv_path)
    df_train, df_valid = train_test_split(
        df, test_size=0.1, stratify=df["level"], random_state=42
    )
    return df_train, df_valid, img_dir


def load_aptos(project_root, fold=0):
    csv_path = os.path.join(project_root, "src", "data", "train_folds.csv")
    img_dir = os.path.join(project_root, "datasets", "aptos2019-blindness-detection", "train_images")
    if not os.path.exists(csv_path):
        print_error_and_exit("train_folds.csv not found. Run src/data/create_folds.py first.", csv_path)

    df = pd.read_csv(csv_path)
    return df[df["fold"] != fold], df[df["fold"] == fold], img_dir


def _map_messidor_label(dr_str):
    dr_str = str(dr_str).strip().upper()
    for i in range(5):
        if str(i) in dr_str:
            return float(i)
    return 0.0


def _find_image(base_name, search_dir):
    for root, _dirs, files in os.walk(search_dir):
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            if base_name + ext in files:
                return os.path.join(root, base_name + ext)
    return None


def load_messidor(project_root):
    # 1. Path to where the actual image files are saved:
    img_dir = os.path.join(project_root, "datasets", "datasets1", "MAPLES_DR_198_Pack", "images")
    
    # 2. Path to where the CSV grading files are saved:
    maples_label_dir = os.path.join(project_root, "datasets", "MAPLES-DR")

    rows = []
    
    # Search for ALL diagnosis.csv files inside datasets/MAPLES-DR (this will catch both train and test!)
    csv_files = glob.glob(os.path.join(maples_label_dir, "**", "diagnosis.csv"), recursive=True)
    
    if not csv_files:
        print_error_and_exit(f"Could not find any diagnosis.csv in {maples_label_dir}", maples_label_dir)

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        
        # Smart detection of the Image column
        name_col = df.columns[0] 
        for c in df.columns:
            if c.lower() in ["name", "image_id", "image", "id_code", "image_name", "file_name", "filename", "img"]:
                name_col = c
                break

        # Smart detection of the DR Label column
        dr_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        for c in df.columns:
            if "dr" in c.lower() or "grade" in c.lower() or "label" in c.lower() or "diagnosis" in c.lower():
                dr_col = c
                break
                
        for _, row in df.iterrows():
            # Clean up the name
            base_name = str(row[name_col]).replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
            
            path = _find_image(base_name, img_dir)
            if path:
                rows.append(dict(image_id=base_name, image_path=path,
                                 label=_map_messidor_label(row[dr_col])))

    df_all = pd.DataFrame(rows).drop_duplicates(subset=["image_id"])
    
    if len(df_all) == 0:
        print_error_and_exit(f"Failed to match images in {img_dir} with the CSVs.", img_dir)

    print(f"[*] MAPLES-DR Loaded Successfully: {len(df_all)} images found!")
    
    # Return the FULL dataset (df_all) as validation to test all 198 images
    return df_all, df_all, img_dir


DATA_LOADERS = {
    "eyepacs": load_eyepacs,
    "aptos": load_aptos,
    "messidor": load_messidor,
}


# ==========================================================================
# MAIN
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Backbone-agnostic DR grading trainer (CheXNet / DenseNet / MobileNet / EfficientNet)"
    )
    p.add_argument("--arch", type=str, required=True,
                   help="b7 | chexnet | densenet121 | mobilenetv3 | mobilenetv3_s | b0..b6")
    p.add_argument("--stage", type=int, default=2, choices=[1, 2],
                   help="1 = EyePACS pre-training, 2 = fine-tuning")
    p.add_argument("--dataset", type=str, default=None,
                   help="Override dataset. Default: eyepacs for stage 1, aptos for stage 2.")
    p.add_argument("--fold", type=int, default=0, help="APTOS fold held out for validation")

    p.add_argument("--use_cbam", action="store_true", help="Insert CBAM after the encoder")
    p.add_argument("--chexnet_weights", type=str, default=None,
                   help="Path to the ChestX-ray14 DenseNet-121 checkpoint (required for --arch chexnet)")
    p.add_argument("--pretrain_path", type=str, default=None,
                   help="Stage-1 checkpoint to warm-start Stage 2")

    p.add_argument("--img_size", type=int, default=None,
                   help="Input resolution. Default 600 for every backbone (matched-input protocol).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=None,
                   help="Default: 1e-4 for stage 1, 2e-5 for stage 2")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_tta", action="store_true", help="Disable TTA during validation")
    p.add_argument("--exp_name", type=str, default=None)
    return p.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    arch = resolve_arch(args.arch)
    cfg = get_backbone_config(arch)
    img_size = get_img_size(arch, args.img_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = args.dataset or ("eyepacs" if args.stage == 1 else "aptos")
    lr = args.lr if args.lr is not None else (1e-4 if args.stage == 1 else 2e-5)

    exp_name = args.exp_name or (
        f"Stage{args.stage}_{arch}"
        + ("_CBAM" if args.use_cbam else "")
        + (f"_{dataset_name}" if dataset_name not in ("eyepacs", "aptos") else "")
    )
    save_dir = os.path.join(PROJECT_ROOT, "src", "experiments", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print_step(f"STAGE {args.stage}: {cfg['paper_name']} on {dataset_name.upper()}")
    print_success(f"Experiment : {exp_name}")
    print_success(f"Device     : {device} | img_size: {img_size} | batch: {args.batch_size}")

    # ---------------- data ----------------
    loader_fn = DATA_LOADERS[dataset_name]
    if dataset_name == "aptos":
        df_train, df_valid, img_dir = loader_fn(PROJECT_ROOT, fold=args.fold)
    else:
        df_train, df_valid, img_dir = loader_fn(PROJECT_ROOT)

    print_success(f"Train: {len(df_train)} | Valid: {len(df_valid)}")

    if dataset_name == "messidor":
        df_train.to_csv(os.path.join(save_dir, "messidor_train_split.csv"), index=False)
        df_valid.to_csv(os.path.join(save_dir, "messidor_valid_split.csv"), index=False)

    train_ds = make_dataset(df_train, dataset_name, img_dir, get_train_transforms(img_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"),
                              drop_last=False)

    # ---------------- model ----------------
    model = build_model(
        arch=arch,
        num_classes=5,
        pretrained=True,
        use_cbam=args.use_cbam,
        chexnet_weights=args.chexnet_weights,
        img_size=img_size,
    ).to(device)

    if args.stage == 2:
        if not args.pretrain_path:
            print_warning(
                "Stage 2 without --pretrain_path: fine-tuning straight from the "
                "initialisation weights. This skips EyePACS pre-training, which the "
                "proposed model received. Report it as such, or run Stage 1 first."
            )
        elif not os.path.exists(args.pretrain_path):
            print_error_and_exit("Stage-1 checkpoint not found.", args.pretrain_path)
        else:
            load_checkpoint(model, args.pretrain_path, device=device, strict=True)
            print_success(f"Loaded Stage-1 weights: {args.pretrain_path}")

    n_params = count_parameters(model)
    print_success(f"Parameters : {n_params / 1e6:.2f} M")

    # ---------------- optimisation ----------------
    #criterion = nn.SmoothL1Loss()
    class ClassificationLossWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_fn = nn.CrossEntropyLoss()
            
        def forward(self, preds, targets):
            # Flatten the targets from [batch_size, 1] to [batch_size]
            # and convert them to integers (long)
            targets = targets.view(-1).long()
            return self.loss_fn(preds, targets)

    criterion = ClassificationLossWrapper()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr / 20
    )
    scaler = make_scaler(device)
    optR = OptimizedRounder()

    best_qwk = -1.0
    history = []

    save_run_config(save_dir, dict(
        arch=arch, paper_name=cfg["paper_name"], timm_name=cfg["timm_name"],
        init=cfg.get("init"), stage=args.stage, dataset=dataset_name,
        fold=args.fold, use_cbam=args.use_cbam, img_size=img_size,
        batch_size=args.batch_size, epochs=args.epochs, lr=lr,
        weight_decay=args.weight_decay, tta=not args.no_tta, seed=args.seed,
        params_millions=round(n_params / 1e6, 3),
        pretrain_path=args.pretrain_path,
        chexnet_weights=args.chexnet_weights,
    ))

    # ---------------- loop ----------------
    print_step("TRAINING")
    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\n[EPOCH {epoch + 1}/{args.epochs}]  lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            accum_steps=args.accum_steps,
        )
        scheduler.step()

        mean_preds, targets, _ = predict_with_tta(
            model, df_valid, dataset_name, img_dir, img_size, device,
            batch_size=max(args.batch_size, 8), num_workers=args.num_workers,
            use_tta=not args.no_tta,
        )

        #optR.fit(mean_preds, targets)
        #coef = optR.coef_["x"]
        #val_qwk = calculate_qwk(targets, optR.predict(mean_preds, coef))
        mean_preds = mean_preds.reshape(-1, 5)
        val_preds = np.argmax(mean_preds, axis=1)
        val_qwk = calculate_qwk(targets, val_preds)

        print(f"   [RESULT] Train Loss: {train_loss:.4f} | Val QWK: {val_qwk:.4f} "
              f"| {time.time() - t0:.0f}s")

        history.append(dict(epoch=epoch + 1, train_loss=float(train_loss),
                            val_qwk=float(val_qwk)))

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            ckpt_name = "stage1_best.pth" if args.stage == 1 else "stage2_best_model.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, ckpt_name))
            #np.save(os.path.join(save_dir, "best_thresholds.npy"), coef)
            np.save(os.path.join(save_dir, "valid_preds.npy"), mean_preds)
            np.save(os.path.join(save_dir, "valid_targets.npy"), targets)
            print_success(f"New best model saved (QWK {best_qwk:.4f})")

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print_step("DONE")
    print_success(f"Best Val QWK: {best_qwk:.5f}")
    print_success(f"Artefacts   : {save_dir}")
    if args.stage == 2:
        print("\n   Next: score this checkpoint on the shared protocol —")
        print(f"   python src/training/evaluate_model.py --exp_dir {save_dir}")


if __name__ == "__main__":
    main()
