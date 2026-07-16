"""
compare_models.py
=================
Turns the per-model `metrics.json` files into the architecture-comparison table
the paper needs, in Markdown and LaTeX, and answers the question a reviewer will
actually ask: *is the proposed model significantly better, or just numerically
higher?*

It reports three things per architecture:

1. Accuracy metrics  - QWK (with bootstrap 95% CI), accuracy, macro/weighted F1.
2. Complexity        - parameters, GFLOPs at the evaluation resolution, and
                       measured inference latency. This is what justifies a
                       heavyweight backbone over MobileNet: if B7+CBAM buys
                       +0.10 QWK for 20x the FLOPs, the paper must say so and
                       argue that screening accuracy is worth it.
3. Significance      - paired bootstrap of the QWK difference against the
                       reference model, on the same validation images.

Usage
-----
python src/training/compare_models.py \
    --exp_dirs src/experiments/Stage2_chexnet \
               src/experiments/Stage2_densenet121 \
               src/experiments/Stage2_mobilenetv3 \
               src/experiments/Stage2_b7 \
               src/experiments/Stage2_b7_CBAM \
    --reference src/experiments/Stage2_b7_CBAM \
    --out results/model_comparison
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

from backbones import get_backbone_config
from model import build_model, count_parameters


# ==========================================================================
# COMPLEXITY
# ==========================================================================
def measure_complexity(arch, use_cbam, img_size, device="cpu", n_warmup=3, n_runs=10):
    """Parameters, GFLOPs and per-image latency at the evaluation resolution."""
    model = build_model(
        arch=arch, num_classes=1, pretrained=False, use_cbam=use_cbam,
        img_size=img_size, verbose=False, require_chexnet=False,
    ).to(device).eval()

    params_m = count_parameters(model) / 1e6
    x = torch.zeros(1, 3, img_size, img_size, device=device)

    gflops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                model(x)
        gflops = fc.get_total_flops() / 1e9
    except Exception:
        try:
            from thop import profile
            macs, _ = profile(model, inputs=(x,), verbose=False)
            gflops = 2 * macs / 1e9
        except Exception:
            pass

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / n_runs * 1000

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(params_millions=params_m, gflops=gflops, latency_ms=latency_ms)


# ==========================================================================
# SIGNIFICANCE
# ==========================================================================
def paired_bootstrap_qwk(y_true, pred_a, pred_b, n_boot=2000, seed=42):
    """
    Bootstrap the QWK difference (a - b) over the SAME resampled images.
    Returns (mean_diff, ci_low, ci_high, p_two_sided).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        ka = cohen_kappa_score(yt, pred_a[idx], weights="quadratic")
        kb = cohen_kappa_score(yt, pred_b[idx], weights="quadratic")
        diffs.append(ka - kb)

    if not diffs:
        return float("nan"), float("nan"), float("nan"), float("nan")

    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap p: how often the sign flips relative to the mean effect.
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), float(lo), float(hi), float(min(p, 1.0))


# ==========================================================================
# TABLES
# ==========================================================================
def to_markdown(rows, ref_label):
    head = ("| Model | Init | Input | Params (M) | GFLOPs | Latency (ms) | "
            "QWK [95% CI] | Accuracy | Macro F1 | Weighted F1 | ΔQWK vs ref | p |")
    sep = "|" + "---|" * 12
    lines = [head, sep]
    for r in rows:
        gf = f"{r['gflops']:.1f}" if r.get("gflops") else "n/a"
        d = r.get("delta_qwk")
        dq = "reference" if r["is_ref"] else (f"{d:+.4f}" if d is not None and d == d else "n/a")
        pv = "-" if r["is_ref"] else (f"{r['p_value']:.3f}" if r.get("p_value") == r.get("p_value") else "n/a")
        lines.append(
            f"| {r['paper_name']}{' + CBAM' if r['use_cbam'] else ''} | {r['init']} | "
            f"{r['img_size']} | {r['params_millions']:.1f} | {gf} | {r['latency_ms']:.0f} | "
            f"{r['qwk']:.4f} [{r['qwk_lo']:.3f}, {r['qwk_hi']:.3f}] | {r['accuracy']:.4f} | "
            f"{r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | {dq} | {pv} |"
        )
    lines.append("")
    lines.append(f"Reference model: **{ref_label}**. "
                 "ΔQWK and p come from a paired bootstrap (2000 resamples) over the same "
                 "validation images. CIs are percentile bootstrap intervals.")
    return "\n".join(lines)


def to_latex(rows):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Architecture comparison on the APTOS 2019 validation fold. All models share the "
        r"same preprocessing, augmentation, ordinal-regression head, TTA and threshold optimisation; "
        r"only the encoder differs.}",
        r"\label{tab:arch_comparison}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Initialisation & Params (M) & GFLOPs & QWK & Accuracy & Macro F1 & $\Delta$QWK \\",
        r"\midrule",
    ]
    for r in rows:
        gf = f"{r['gflops']:.1f}" if r.get("gflops") else "--"
        d = r.get("delta_qwk")
        dq = "--" if r["is_ref"] else (f"{d:+.4f}" if d is not None and d == d else "--")
        name = r["paper_name"].replace("&", r"\&") + (" + CBAM" if r["use_cbam"] else "")
        lines.append(
            f"{name} & {r['init']} & {r['params_millions']:.1f} & {gf} & "
            f"{r['qwk']:.4f} & {r['accuracy']:.4f} & {r['macro_f1']:.4f} & {dq} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ==========================================================================
# MAIN
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Build the architecture-comparison table")
    p.add_argument("--exp_dirs", nargs="+", required=True)
    p.add_argument("--reference", type=str, default=None,
                   help="Experiment dir treated as the proposed model (default: highest QWK)")
    p.add_argument("--out", type=str, default="results/model_comparison")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--complexity_device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--skip_complexity", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    records = []
    for d in args.exp_dirs:
        mpath = os.path.join(d, "metrics.json")
        if not os.path.exists(mpath):
            print(f"   [SKIP] no metrics.json in {d} — run evaluate_model.py on it first.")
            continue
        with open(mpath) as f:
            m = json.load(f)
        m["_dir"] = d
        records.append(m)

    if not records:
        print("Nothing to compare. Run evaluate_model.py on at least one experiment.")
        return

    # Every row must be scored on the same validation set, or the table lies.
    n_valids = {r["n_valid"] for r in records}
    datasets = {r["dataset"] for r in records}
    if len(n_valids) > 1 or len(datasets) > 1:
        print(f"   [WARNING] Rows differ in evaluation set (datasets={datasets}, "
              f"n_valid={n_valids}). Paired testing is disabled and the table is not "
              f"a like-for-like comparison. Re-evaluate every model on one fold.")

    ref_dir = args.reference or max(records, key=lambda r: r["metrics"]["qwk"])["_dir"]
    ref = next((r for r in records if os.path.abspath(r["_dir"]) == os.path.abspath(ref_dir)), None)
    if ref is None:
        print(f"Reference {ref_dir} is not among the evaluated experiments.")
        return

    ref_true = np.load(os.path.join(ref["_dir"], "eval_y_true.npy"))
    ref_pred = np.load(os.path.join(ref["_dir"], "eval_y_pred.npy"))
    ref_label = ref["paper_name"] + (" + CBAM" if ref["use_cbam"] else "")

    rows = []
    for r in records:
        is_ref = os.path.abspath(r["_dir"]) == os.path.abspath(ref["_dir"])

        complexity = dict(params_millions=r.get("params_millions", float("nan")),
                          gflops=None, latency_ms=float("nan"))
        if not args.skip_complexity:
            print(f"   [*] Measuring complexity: {r['arch']} @ {r['img_size']}px ...")
            complexity = measure_complexity(
                r["arch"], r["use_cbam"], r["img_size"], device=args.complexity_device
            )

        delta = lo = hi = p = float("nan")
        if not is_ref:
            y_true = np.load(os.path.join(r["_dir"], "eval_y_true.npy"))
            y_pred = np.load(os.path.join(r["_dir"], "eval_y_pred.npy"))
            if len(y_true) == len(ref_true) and np.array_equal(y_true, ref_true):
                # Sign convention: reference minus this model, so a positive
                # delta means the proposed model wins.
                delta, lo, hi, p = paired_bootstrap_qwk(
                    ref_true, ref_pred, y_pred, n_boot=args.n_boot
                )
            else:
                print(f"   [WARNING] {r['arch']} was evaluated on a different image set; "
                      f"skipping the paired test for this row.")

        rows.append(dict(
            arch=r["arch"], paper_name=r["paper_name"], init=r.get("init") or "n/a",
            use_cbam=r["use_cbam"], img_size=r["img_size"], is_ref=is_ref,
            qwk=r["metrics"]["qwk"], qwk_lo=r["ci"]["qwk"][0], qwk_hi=r["ci"]["qwk"][1],
            accuracy=r["metrics"]["accuracy"], macro_f1=r["metrics"]["macro_f1"],
            weighted_f1=r["metrics"]["weighted_f1"],
            delta_qwk=delta, ci_low=lo, ci_high=hi, p_value=p,
            **complexity,
        ))

    rows.sort(key=lambda x: (x["is_ref"], x["qwk"]))

    md = to_markdown(rows, ref_label)
    tex = to_latex(rows)

    with open(args.out + ".md", "w") as f:
        f.write("# Architecture comparison\n\n" + md + "\n")
    with open(args.out + ".tex", "w") as f:
        f.write(tex + "\n")
    pd.DataFrame(rows).to_csv(args.out + ".csv", index=False)

    print("\n" + md + "\n")
    print(f"Wrote {args.out}.md / .tex / .csv")


if __name__ == "__main__":
    main()
