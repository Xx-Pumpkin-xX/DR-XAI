## Consensus-Calibrated Explanation Map (CCEM)

The CCEM algorithm provides a final consensus formulation for combining multiple explanation maps (Grad-CAM++, AdaSISE, and SmoothIG). It addresses the scale mismatch between different attribution mechanisms by converting them into percentile-rank maps, gating the result by multi-method agreement, and applying a pixelwise disagreement penalty.

This repository implements the algorithm by weighting the inputs heavily towards structural preservation (Grad-CAM++: 0.4) while maintaining detailed granular fidelity from perturbation and gradient methods (AdaSISE: 0.3, SmoothIG: 0.3).

### Directory Structure

The core mathematical implementation is separated from the execution runner:

```text
src/xai/CCEM/
  ccem_core.py               # Core algorithm implementation (Rank transform, Penalty, Gating)
  run_maples_ccem.py         # CLI Runner to fuse .npy files and generate metrics/visuals
```
### Example Usage

```text
python src/xai/CCEM/run_maples_ccem.py \
  --xai_dir XAI_10ex_run \
  --messidor_img_dir datasets1 \
  --maples_dir MAPLES-DR \
  --img_size 600
```

### Output

```text
XAI_10ex_run/CCEM_results/
  visuals/                 # 6-panel comparison figures (Original, Mask, 3 XAI, CCEM)
  npy/                     # Final CCEM heatmaps as .npy arrays
  ccem_results.csv         # Per-image localization metrics for CCEM
  ccem_report.txt          # Global mean metrics summary
```

```bash
python src/xai/CCEM/run_maples_ccem.py --xai_dir XAI_10ex_run