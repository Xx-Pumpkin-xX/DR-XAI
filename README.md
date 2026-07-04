# ProjectDR2026

Diabetic Retinopathy (DR) grading and Explainable AI (XAI) benchmarking pipeline.

The project trains an EfficientNet-based regression model (with an optional CBAM
attention module) to grade DR severity from fundus images, then generates and
benchmarks multiple explanation methods (Grad-CAM++, AdaSISE, SmoothIG) against
expert-annotated lesion masks from MAPLES-DR. A final method, **CCEM
(Consensus-Calibrated Explanation Map)**, fuses these explanations into a single
consensus heatmap.

## Datasets

Located under `datasets/` (not tracked in git):

| Dataset | Purpose |
| --- | --- |
| APTOS 2019 Blindness Detection | Stage 2 fine-tuning / main DR grading dataset |
| EyePACS 2015 | Stage 1 large-scale pre-training |
| MESSIDOR-2 | Stage 2 fine-tuning + XAI benchmarking |
| MAPLES-DR | Expert lesion masks used as XAI ground truth |

Helper scripts: `datasets/download_maples.py`, `datasets/check_health.py`,
`datasets/check_maples.py`.

## Project Layout

See [`src/docs/localize.md`](src/docs/localize.md) for the full directory map.
High level:

```text
src/
  data/            # Preprocessing, folds, label auditing, blind grading prep
  training/        # Model (EfficientNet + CBAM), augmentation, 3-stage training
  xai/
    explanation/   # Grad-CAM++, AdaSISE, SmoothIG explainers + runners
    CCEM/          # Consensus fusion of the explanation maps
    improvement_v1/
  Archive/         # Deprecated scripts kept for reference/ablation
test_scripts/      # Convenience wrappers around the XAI runners
results/           # Benchmark reports and visuals
test_scripts/XAI_10ex_run/      # Sample XAI outputs (10 images/grade) used for benchmarking
```

## Model Training

Three-stage training pipeline (see `src/training/`):

1. `train_stage1.py` — pre-train on EyePACS 2015 (~35k images)
2. `train_stage2.py` / `train_stage2_messidor.py` — fine-tune on APTOS 2019 / MESSIDOR
3. `train_stage3_attention.py` — attention-guided fine-tuning using MAPLES-DR masks
   to supervise the CBAM module

Example:

```bash
python src/training/train_stage2.py --model B7 --use_cbam --batch_size 16 --epochs 15
```

## Explainability (XAI)

Each explainer loads a trained checkpoint, produces a heatmap, and scores it
against MAPLES-DR lesion masks (Energy, AUC-ROC, IoU). See
[`src/xai/explanation/README.md`](src/xai/explanation/README.md) for full options.

```bash
python src/xai/explanation/run_maples_gradcampp.py \
  --model B7 --weight_path <checkpoint>.pth --use_cbam --img_size 600

python src/xai/explanation/run_maples_adasise.py \
  --model B7 --weight_path <checkpoint>.pth --use_cbam --target_layer_mode lesion

python src/xai/explanation/run_maples_smoothig.py \
  --model B7 --weight_path <checkpoint>.pth --use_cbam --nt_samples 16
```

### CCEM (Consensus Fusion)

Fuses Grad-CAM++, AdaSISE, and SmoothIG heatmaps into one consensus map. See
[`src/xai/CCEM/README.md`](src/xai/CCEM/README.md).

```bash
python src/xai/CCEM/run_maples_ccem.py \
  --xai_dir test_scripts/XAI_10ex_run --messidor_img_dir datasets1 --maples_dir MAPLES-DR --img_size 600
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Results

Benchmark reports and visuals are under `results/maples_benchmark_results/` and
`test_scripts/XAI_10ex_run/`, comparing classification performance (Accuracy, QWK) and XAI
localization performance (Energy, AUC-ROC, IoU) across methods.
