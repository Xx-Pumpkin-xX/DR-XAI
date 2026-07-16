"""
evaluate_classification.py
==========================
Scores a classification checkpoint (5 output classes).
Uses argmax instead of thresholds to assign final DR grades.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from backbones import get_backbone_config, get_img_size
from model import build_model, count_parameters, load_checkpoint
from pipeline import (
    predict_with_tta, print_error_and_exit, print_step, print_success, print_warning,
)
from train_baseline import DATA_LOADERS

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# NEW
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))

# ==========================================================================
# METRICS
# ==========================================================================
def compute_metrics(y_true, y_pred):
    return dict(
        qwk=float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )

def bootstrap_ci(y_true, y_pred, metric="qwk", n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        if metric == "qwk":
            scores.append(cohen_kappa_score(yt, yp, weights="quadratic"))
        else:
            scores.append(accuracy_score(yt, yp))
    if not scores:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)

# ==========================================================================
# REPORT
# ==========================================================================
def write_report(path, cfg, y_true, y_pred, metrics, cis):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    lines = [
        "",
        "=" * 65,
        f" CLASSIFICATION EVALUATION (Model: {cfg['paper_name']} | Fold: {cfg.get('fold', 0)})",
        "=" * 65,
        f" Architecture:                      {cfg['arch']} ({cfg['timm_name']})",
        f" Initialisation:                    {cfg.get('init', 'n/a')}",
        f" CBAM:                              {cfg.get('use_cbam', False)}",
        f" Input size:                        {cfg['img_size']}",
        f" Parameters (M):                    {cfg.get('params_millions', float('nan')):.2f}",
        f" TTA:                               {cfg.get('tta', True)}",
        f" Validation images:                 {len(y_true)}",
        "=" * 65,
        f" Quadratic Weighted Kappa (QWK):    {metrics['qwk']:.5f}   [95% CI {cis['qwk'][0]:.5f} - {cis['qwk'][1]:.5f}]",
        f" Accuracy:                          {metrics['accuracy']:.5f}   [95% CI {cis['accuracy'][0]:.5f} - {cis['accuracy'][1]:.5f}]",
        f" Macro Precision:                   {metrics['macro_precision']:.5f}",
        f" Macro Recall (Sensitivity):        {metrics['macro_recall']:.5f}",
        f" Macro F1-Score:                    {metrics['macro_f1']:.5f}",
        f" Weighted F1-Score:                 {metrics['weighted_f1']:.5f}",
        "=" * 65,
        "",
        " CLASSIFICATION REPORT:",
        report,
        "",
        " CONFUSION MATRIX:",
        str(cm),
        "=" * 65,
    ]
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(text)

# ==========================================================================
# MAIN
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Score a classification checkpoint")
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--weight_path", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = os.path.join(args.exp_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        print_error_and_exit("run_config.json not found.", cfg_path)

    with open(cfg_path) as f:
        run_cfg = json.load(f)

    arch = run_cfg["arch"]
    bcfg = get_backbone_config(arch)
    img_size = get_img_size(arch, run_cfg.get("img_size"))
    use_cbam = bool(run_cfg.get("use_cbam", False))
    dataset_name = args.dataset or run_cfg.get("dataset", "aptos")
    fold = args.fold if args.fold is not None else run_cfg.get("fold", 0)

    weight_path = args.weight_path or os.path.join(args.exp_dir, "stage2_best_model.pth")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(args.exp_dir, "stage1_best.pth")

    print_step(f"EVALUATION: {bcfg['paper_name']}{' + CBAM' if use_cbam else ''} (Classification)")

    # ---------------- data ----------------
    if dataset_name == "aptos":
        _df_train, df_valid, img_dir = DATA_LOADERS["aptos"](PROJECT_ROOT, fold=fold)
    else:
        _df_train, df_valid, img_dir = DATA_LOADERS[dataset_name](PROJECT_ROOT)

    # ---------------- model ----------------
    # CHANGED: num_classes=5 for classification
    model = build_model(
        arch=arch, num_classes=5, pretrained=False, use_cbam=use_cbam,
        img_size=img_size, verbose=True, require_chexnet=False,
    )
    load_checkpoint(model, weight_path, device=device, strict=True)
    model = model.to(device).eval()
    n_params = count_parameters(model)

    # ---------------- predict ----------------
    mean_preds, targets, per_view = predict_with_tta(
        model, df_valid, dataset_name, img_dir, img_size, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_tta=not args.no_tta, desc="Eval",
    )

    # CHANGED: Argmax to get the predicted class directly
    mean_preds = mean_preds.reshape(-1, 5)
    y_pred = np.argmax(mean_preds, axis=1)
    y_true = targets.astype(int)

    # ---------------- score ----------------
    metrics = compute_metrics(y_true, y_pred)
    cis = dict(
        qwk=bootstrap_ci(y_true, y_pred, "qwk", args.n_bootstrap),
        accuracy=bootstrap_ci(y_true, y_pred, "accuracy", args.n_bootstrap),
    )

    report_cfg = dict(
        arch=arch, paper_name=bcfg["paper_name"], timm_name=bcfg["timm_name"],
        init=run_cfg.get("init"), use_cbam=use_cbam, img_size=img_size,
        params_millions=n_params / 1e6, tta=not args.no_tta, fold=fold,
    )

    # ---------------- score & report ----------------
    # 1. Save Report with dataset name appended
    report_filename = f"evaluation_report_{dataset_name}.txt"
    write_report(os.path.join(args.exp_dir, report_filename), report_cfg, y_true, y_pred, metrics, cis)

    # ---------------- artefacts ----------------
    payload = dict(
        **report_cfg,
        dataset=dataset_name,
        n_valid=int(len(y_true)),
        metrics=metrics,
        ci=dict(qwk=cis["qwk"], accuracy=cis["accuracy"]),
        thresholds=[],
        threshold_source="classification (argmax)",
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist(),
        exp_dir=os.path.abspath(args.exp_dir),
        weight_path=os.path.abspath(weight_path),
    )
    
    # 2. Save JSON with dataset name appended
    json_filename = f"metrics_{dataset_name}.json"
    with open(os.path.join(args.exp_dir, json_filename), "w") as f:
        json.dump(payload, f, indent=2)

    # 3. Save raw arrays and CSV with dataset name appended
    np.save(os.path.join(args.exp_dir, f"eval_y_true_{dataset_name}.npy"), y_true)
    np.save(os.path.join(args.exp_dir, f"eval_y_pred_{dataset_name}.npy"), y_pred)
    np.save(os.path.join(args.exp_dir, f"eval_raw_preds_{dataset_name}.npy"), mean_preds)

    pd.DataFrame(dict(y_true=y_true, y_pred=y_pred)).to_csv(
        os.path.join(args.exp_dir, f"eval_predictions_{dataset_name}.csv"), index=False
    )

    print_success(f"Wrote {report_filename} + {json_filename} to {args.exp_dir}")

if __name__ == "__main__":
    main()