"""
evaluation.py
=============
`train.py` has always imported `from evaluation import evaluate_baseline`, but
the module was never in the repository, so `train.py` fails at import time. This
restores it, using the same metric set as `evaluate_model.py` so a classification
run and a regression run are scored on identical definitions.

`train.py` is the softmax/FocalLoss baseline path. The ordinal-regression path
used for the paper is `train_baseline.py` -> `evaluate_model.py`.
"""

import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOTTING = True
except Exception:
    _HAS_PLOTTING = False

from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)


def evaluate_baseline(y_true, y_pred, y_probs=None, save_dir=".", model_tag="baseline"):
    """
    Write evaluation_report.txt (+ a confusion-matrix figure when matplotlib is
    available) and return the metric dictionary.
    """
    os.makedirs(save_dir, exist_ok=True)

    y_true = np.asarray(y_true).astype(int).flatten()
    y_pred = np.asarray(y_pred).astype(int).flatten()

    metrics = dict(
        qwk=float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    lines = [
        "",
        "=" * 65,
        f" EVALUATION METRICS SUMMARY ({model_tag})",
        "=" * 65,
        f" Quadratic Weighted Kappa (QWK):    {metrics['qwk']:.5f}",
        f" Accuracy:                          {metrics['accuracy']:.5f}",
        f" Macro Precision:                   {metrics['macro_precision']:.5f}",
        f" Macro Recall (Sensitivity):        {metrics['macro_recall']:.5f}",
        f" Macro F1-Score:                    {metrics['macro_f1']:.5f}",
        f" Weighted F1-Score:                 {metrics['weighted_f1']:.5f}",
        "=" * 65,
        "",
        " CLASSIFICATION REPORT:",
        classification_report(y_true, y_pred, digits=4, zero_division=0),
        "",
        " CONFUSION MATRIX:",
        str(cm),
        "=" * 65,
    ]
    text = "\n".join(lines)

    with open(os.path.join(save_dir, "evaluation_report.txt"), "w") as f:
        f.write(text)
    print(text)

    if _HAS_PLOTTING:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted grade")
        ax.set_ylabel("True grade")
        ax.set_title(f"Confusion matrix — {model_tag}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)

    return metrics
