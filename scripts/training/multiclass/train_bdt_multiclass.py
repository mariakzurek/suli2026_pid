"""
train_bdt.py — Fit a LightGBM multiclass BDT and Platt-calibrate on the training split.

WHAT IT DOES
------------
Loads the training and validation parquet files produced by build_dataset.py,
fits a LightGBM multiclass classifier (π/K/proton), applies Platt calibration
on a held-out slice of the training set (row-level, stratified, never val or
test), and writes the fitted model, calibrator, evaluation metrics, and
diagnostic plots.

CHANGED FOR MULTICLASS (BDT-2):
--------------------------------
* objective='multiclass', num_class=3 (0=pi, 1=K, 2=proton)
* predict_proba now returns shape (n, 3); all downstream scoring updated
* AUC computed one-vs-rest (multi_class='ovr')
* Brier score generalized to multiclass (mean sum-of-squares vs one-hot truth)
* Reliability diagram now one panel per class instead of a single K-vs-pi panel
* ROC plot now one curve per class (one-vs-rest) instead of a single curve
"""

import argparse
import datetime
import json
import pathlib
import sys
from typing import Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    auc,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


try:
    from lightgbm import LGBMClassifier
except ImportError:
    print(
        "ERROR: lightgbm not installed in the current environment.\n"
        "       On ifarm: conda activate suli2026_pid && conda install -c conda-forge lightgbm",
        file=sys.stderr,
    )
    sys.exit(1)


# CHANGED: class label -> name mapping, used for plot legends/titles.
CLASS_NAMES = {0: "pi+", 1: "K+", 2: "proton"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from binary version)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_features_file(path: pathlib.Path) -> List[str]:
    lines = []
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    if not lines:
        print(
            f"ERROR: --features-file is empty or contains only comments: {path}\n"
            f"       Fill it with column names from scripts/training/columns_maximal.txt.\n"
            f"       One column name per line, # for comments.",
            file=sys.stderr,
        )
        sys.exit(1)
    return lines


def _validate_features_against_parquet(
    feature_list: List[str],
    parquet_path: pathlib.Path,
) -> None:
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(str(parquet_path))
        parquet_cols = set(schema.names)
    except Exception:
        try:
            parquet_cols = set(pd.read_parquet(str(parquet_path), columns=[]).columns)
            if not parquet_cols:
                parquet_cols = set(pd.read_parquet(str(parquet_path)).columns)
        except Exception as e:
            print(
                f"WARNING: Could not read parquet schema from {parquet_path}: {e}\n"
                f"         Skipping feature validation — training may fail later.",
                file=sys.stderr,
            )
            return

    missing = [f for f in feature_list if f not in parquet_cols]
    if missing:
        print(
            f"ERROR: The following features from --features-file are not columns "
            f"in {parquet_path}:\n"
            f"  {missing}\n\n"
            f"Available columns in parquet: {sorted(parquet_cols)}",
            file=sys.stderr,
        )
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics / plotting helpers — CHANGED for multiclass
# ──────────────────────────────────────────────────────────────────────────────

def _plot_reliability_diagram(
    y_true_val: np.ndarray,
    scores_uncal: np.ndarray,
    scores_cal: np.ndarray,
    out_path: pathlib.Path,
    n_bins: int = 10,
) -> None:
    """
    One row per class (pi/K/proton), pre-cal (left) vs post-cal (right),
    using a one-vs-rest reliability curve per class.
    """
    n_classes = scores_uncal.shape[1]
    fig, axes = plt.subplots(n_classes, 2, figsize=(10, 4 * n_classes), sharey=True)

    if n_classes == 1:
        axes = axes.reshape(1, 2)

    for c in range(n_classes):
        y_true_c = (y_true_val == c).astype(int)
        for col, (scores, title) in enumerate(
            zip(
                (scores_uncal[:, c], scores_cal[:, c]),
                ("Pre-calibration (raw BDT)", "Post-calibration (Platt)"),
            )
        ):
            ax = axes[c, col]
            frac_pos, mean_pred = calibration_curve(y_true_c, scores, n_bins=n_bins)
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
            ax.plot(mean_pred, frac_pos, "o-", lw=2, markersize=5, label="BDT")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel(f"Fraction of positives ({CLASS_NAMES[c]})")
            ax.set_title(f"{CLASS_NAMES[c]} — {title}")
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    fig.suptitle("Reliability diagrams (one-vs-rest, evaluated on validation set)", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(
    y_true: np.ndarray,
    scores: np.ndarray,
    out_path: pathlib.Path,
    label: str = "BDT (calibrated)",
) -> Dict[int, float]:
    """
    One-vs-rest ROC curve per class, all on one plot. Returns per-class AUC dict.
    """
    n_classes = scores.shape[1]
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=(6, 5))
    per_class_auc = {}

    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], scores[:, c])
        roc_auc_c = auc(fpr, tpr)
        per_class_auc[c] = float(roc_auc_c)
        ax.plot(fpr, tpr, lw=2, label=f"{CLASS_NAMES[c]} (AUC = {roc_auc_c:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — validation set ({label}, one-vs-rest)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return per_class_auc


def _plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    out_path: pathlib.Path,
    top_n: int = 15,
) -> None:
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(7, max(3, top_n * 0.35)))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals[::-1], align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("LightGBM gain importance")
    ax.set_title(f"Feature importance (top {top_n} by gain)")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_readme(
    outdir: pathlib.Path,
    dataset_dir: pathlib.Path,
    feature_list: List[str],
    lgb_params: dict,
    calibration_frac: float,
    reweight_map: Optional[str],
    metrics: dict,
    timestamp: str,
) -> None:
    lines = [
        "# BDT (multiclass) training run provenance\n",
        f"Generated: {timestamp}\n\n",
        "## Dataset\n",
        f"- Source: `{dataset_dir}`\n",
        f"- Features ({len(feature_list)}): {', '.join(feature_list)}\n",
        f"- Classes: {CLASS_NAMES}\n",
        f"- Reweight map: {reweight_map or 'None (unweighted)'}\n\n",
        "## Hyperparameters\n",
    ]
    for k, v in lgb_params.items():
        lines.append(f"- `{k}`: {v}\n")
    lines += [
        f"- calibration_frac: {calibration_frac}\n\n",
        "## Metrics (validation set)\n",
    ]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.5f}\n")
        else:
            lines.append(f"- {k}: {v}\n")
    lines += [
        "\n## Outputs\n",
        "- `model.joblib` — wrapper dict {\"model\": calibrated LightGBM + Platt calibrator, "
        "\"features\": list of training feature names, \"class_names\": label map}\n",
        "- `training_summary.csv` — AUC (per-class + macro), Brier, log-loss for train/val pre/post cal\n",
        "- `reliability_diagram.png` — calibration quality per class (on val set)\n",
        "- `roc_val.png` — one-vs-rest ROC curves on val set\n",
        "- `feature_importance.png` / `.csv` — top-15 features by gain\n",
    ]
    (outdir / "README.md").write_text("".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def train_bdt(
    train_path: pathlib.Path,
    val_path: pathlib.Path,
    feature_list: List[str],
    outdir: pathlib.Path,
    reweight_map: Optional[pathlib.Path] = None,
    calibration_frac: float = 0.2,
    lgb_kwargs: Optional[dict] = None,
    random_state: int = 42,
    overwrite: bool = False,
) -> Dict:
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / "model.joblib"
    if model_path.exists() and not overwrite:
        raise FileExistsError(
            f"Model already exists: {model_path}. Pass overwrite=True to replace."
        )

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading training data ...")
    df_train_full = pd.read_parquet(str(train_path))
    df_val = pd.read_parquet(str(val_path))

    missing_feats = [f for f in feature_list if f not in df_train_full.columns]
    if missing_feats:
        print(
            f"WARNING: {len(missing_feats)} feature(s) not found in training parquet "
            f"(will be all-NaN): {missing_feats}",
            file=sys.stderr,
        )

    X_full = df_train_full[feature_list].to_numpy(dtype=np.float32, na_value=np.nan)
    y_full = df_train_full["label"].astype(np.int8).to_numpy()

    X_val = df_val[feature_list].to_numpy(dtype=np.float32, na_value=np.nan)
    y_val = df_val["label"].astype(np.int8).to_numpy()

    # CHANGED: fixed n_classes (not data-dependent) + explicit presence check.
    n_classes = len(CLASS_NAMES)
    present_train = set(np.unique(y_full).tolist())
    present_val = set(np.unique(y_val).tolist())
    expected = set(CLASS_NAMES.keys())
    if present_train != expected:
        print(
            f"WARNING: train labels present {sorted(present_train)} != expected "
            f"{sorted(expected)}. Missing classes will make LightGBM's "
            f"num_class={n_classes} setting inconsistent with the data. "
            f"Check build_dataset.py's truth_breakdown before proceeding.",
            file=sys.stderr,
        )
    if present_val != expected:
        print(
            f"WARNING: val labels present {sorted(present_val)} != expected "
            f"{sorted(expected)}.",
            file=sys.stderr,
        )

    n_train_full = len(X_full)
    n_val = len(X_val)
    print(f"  Train (full): {n_train_full:,} rows, Val: {n_val:,} rows")
    print(f"  Train class fractions: {np.bincount(y_full, minlength=n_classes) / len(y_full)}")
    print(f"  Val class fractions: {np.bincount(y_val, minlength=n_classes) / len(y_val)}")

    # ── Calibration split ──────────────────────────────────────────────────────
    print(f"Carving calibration slice ({calibration_frac:.0%} of train, stratified) ...")
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_full, y_full,
        test_size=calibration_frac,
        random_state=random_state,
        stratify=y_full,
    )
    print(f"  BDT fit rows: {len(X_fit):,}  |  Calibration rows: {len(X_cal):,}")

    # ── Optional reweight map ──────────────────────────────────────────────────
    w_fit = None
    if reweight_map is not None:
        print(f"Loading reweight map: {reweight_map}")
        rw = np.load(str(reweight_map))
        p_edges = rw["p_edges"]
        theta_edges = rw["theta_edges"]
        weights_grid = rw["weights"]

        p_arr = df_train_full["p"].to_numpy(dtype=np.float32)
        theta_arr = df_train_full["theta"].to_numpy(dtype=np.float32)

        p_bin = np.searchsorted(p_edges[1:-1], p_arr)
        theta_bin = np.searchsorted(theta_edges[1:-1], theta_arr)
        p_bin = np.clip(p_bin, 0, weights_grid.shape[0] - 1)
        theta_bin = np.clip(theta_bin, 0, weights_grid.shape[1] - 1)

        w_full = weights_grid[p_bin, theta_bin]

        idx_all = np.arange(n_train_full)
        idx_fit, idx_cal_unused = train_test_split(
            idx_all,
            test_size=calibration_frac,
            random_state=random_state,
            stratify=y_full,
        )
        w_fit = w_full[idx_fit]
        print(f"  Reweight map applied; mean weight = {w_fit.mean():.3f}")

    # ── Fit LightGBM ──────────────────────────────────────────────────────────
    # CHANGED: objective + num_class for multiclass.
    default_lgb = dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        objective="multiclass",
        num_class=n_classes,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    if lgb_kwargs:
        default_lgb.update(lgb_kwargs)

    print(f"Fitting LightGBM: {default_lgb}")
    clf = LGBMClassifier(**default_lgb)
    clf.fit(X_fit, y_fit, sample_weight=w_fit)
    print("  LightGBM fit complete.")

    # ── Pre-calibration scores ─────────────────────────────────────────────────
    # CHANGED: no [:, 1] indexing — keep full (n, n_classes) score matrix.
    scores_train_uncal = clf.predict_proba(X_full)
    scores_val_uncal = clf.predict_proba(X_val)

    # CHANGED: AUC via one-vs-rest; guard against a degenerate class count.
    def _safe_auc(y_true, y_score):
        n_present = len(np.unique(y_true))
        if n_present < 2:
            return float("nan")
        if y_score.shape[1] == 2:
            return roc_auc_score(y_true, y_score[:, 1])
        return roc_auc_score(y_true, y_score, multi_class="ovr")

    auc_train_uncal = _safe_auc(y_full, scores_train_uncal)
    auc_val_uncal = _safe_auc(y_val, scores_val_uncal)

    # CHANGED: multiclass Brier score (mean sum-of-squares vs one-hot truth).
    def _multiclass_brier(y_true, y_score):
        classes = np.arange(y_score.shape[1])
        y_onehot = label_binarize(y_true, classes=classes)
        return float(np.mean(np.sum((y_score - y_onehot) ** 2, axis=1)))

    brier_val_uncal = _multiclass_brier(y_val, scores_val_uncal)
    logloss_val_uncal = log_loss(y_val, scores_val_uncal, labels=np.arange(n_classes))
    print(f"  Pre-cal AUC  train={auc_train_uncal:.4f}  val={auc_val_uncal:.4f}")

    # ── Platt calibration ──────────────────────────────────────────────────────
    print("Fitting Platt calibration on held-out calibration slice ...")
    calibrated_clf = CalibratedClassifierCV(
        estimator=clf, method="sigmoid", cv="prefit"
    )
    calibrated_clf.fit(X_cal, y_cal)
    print("  Calibration fit complete.")

    # ── Post-calibration scores ────────────────────────────────────────────────
    scores_train_cal = calibrated_clf.predict_proba(X_full)
    scores_val_cal = calibrated_clf.predict_proba(X_val)

    auc_train_cal = _safe_auc(y_full, scores_train_cal)
    auc_val_cal = _safe_auc(y_val, scores_val_cal)
    brier_val_cal = _multiclass_brier(y_val, scores_val_cal)
    logloss_val_cal = log_loss(y_val, scores_val_cal, labels=np.arange(n_classes))
    print(f"  Post-cal AUC train={auc_train_cal:.4f}  val={auc_val_cal:.4f}")
    print(f"  Post-cal Brier val={brier_val_cal:.4f}  LogLoss val={logloss_val_cal:.4f}")

    metrics = {
        "auc_train_uncal": auc_train_uncal,
        "auc_val_uncal": auc_val_uncal,
        "auc_train_cal": auc_train_cal,
        "auc_val_cal": auc_val_cal,
        "brier_val_uncal": brier_val_uncal,
        "brier_val_cal": brier_val_cal,
        "logloss_val_uncal": logloss_val_uncal,
        "logloss_val_cal": logloss_val_cal,
        "n_train": n_train_full,
        "n_cal": len(X_cal),
        "n_fit": len(X_fit),
        "n_val": n_val,
        "n_classes": n_classes,
        "class_names": CLASS_NAMES,
        "train_class_counts": np.bincount(y_full, minlength=n_classes).tolist(),
        "val_class_counts": np.bincount(y_val, minlength=n_classes).tolist(),
    }

    # ── Save model ─────────────────────────────────────────────────────────────
    # CHANGED: include class_names in the wrapper dict.
    model_bundle = {
        "model": calibrated_clf,
        "features": feature_list,
        "class_names": CLASS_NAMES,
    }
    joblib.dump(model_bundle, str(model_path), compress=3)
    print(f"  Model saved → {model_path}")
    print(f"  (wrapper dict: model + features={feature_list} + class_names={CLASS_NAMES})")

    # ── Training summary CSV ───────────────────────────────────────────────────
    pd.DataFrame([metrics]).to_csv(str(outdir / "training_summary.csv"), index=False)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("Generating reliability diagram ...")
    _plot_reliability_diagram(
        y_val, scores_val_uncal, scores_val_cal,
        outdir / "reliability_diagram.png",
    )

    print("Generating ROC curve ...")
    per_class_auc = _plot_roc(y_val, scores_val_cal, outdir / "roc_val.png")
    print(f"  Per-class AUC (val, calibrated): {per_class_auc}")

    print("Generating feature importance plot ...")
    feature_importances = clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_list,
        "importance_gain": feature_importances,
    }).sort_values("importance_gain", ascending=False)
    feat_imp_df.to_csv(str(outdir / "feature_importance.csv"), index=False)
    _plot_feature_importance(feature_list, feature_importances, outdir / "feature_importance.png")

    # ── README ─────────────────────────────────────────────────────────────────
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    _write_readme(
        outdir=outdir,
        dataset_dir=train_path.parent,
        feature_list=feature_list,
        lgb_params=default_lgb,
        calibration_frac=calibration_frac,
        reweight_map=str(reweight_map) if reweight_map else None,
        metrics=metrics,
        timestamp=timestamp,
    )

    return {
        "model_path": model_path,
        "metrics": metrics,
        "feature_importance": feat_imp_df,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point (unchanged from binary version)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset-dir", required=True,
        help="Directory produced by build_dataset.py; must contain train.parquet, "
             "val.parquet, and manifest.json.")
    p.add_argument("--features-file", required=True, metavar="PATH",
        help="Text file listing column names to use as training features.")
    p.add_argument("--outdir", required=True,
        help="Output directory for model.joblib, plots, and CSVs.")
    p.add_argument("--reweight-map", default=None, metavar="PATH",
        help="Optional .npz reweight map (p_edges, theta_edges, weights).")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--calibration-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    dataset_dir = pathlib.Path(args.dataset_dir)
    outdir = pathlib.Path(args.outdir)
    features_file = pathlib.Path(args.features_file)

    if not features_file.exists():
        print(f"ERROR: --features-file not found: {features_file}", file=sys.stderr)
        sys.exit(1)

    feature_list = _parse_features_file(features_file)
    print(f"Features loaded ({len(feature_list)}) from {features_file}: {feature_list}")

    train_path = dataset_dir / "train.parquet"
    val_path = dataset_dir / "val.parquet"
    for path in (train_path, val_path):
        if not path.exists():
            print(f"ERROR: Required parquet not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"Validating features against parquet schema: {train_path}")
    _validate_features_against_parquet(feature_list, train_path)
    print(f"  All {len(feature_list)} features present in parquet. OK.")

    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        p_max = manifest.get("p_max")
        print(f"p_max from manifest: {p_max} GeV/c")
    else:
        print(f"WARNING: manifest.json not found in {dataset_dir}.", file=sys.stderr)

    lgb_kwargs = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
    )

    reweight_map = pathlib.Path(args.reweight_map) if args.reweight_map else None

    result = train_bdt(
        train_path=train_path,
        val_path=val_path,
        feature_list=feature_list,
        outdir=outdir,
        reweight_map=reweight_map,
        calibration_frac=args.calibration_frac,
        lgb_kwargs=lgb_kwargs,
        random_state=args.seed,
        overwrite=args.overwrite,
    )

    print("\nDone.")
    print(f"  Model      : {result['model_path']}")
    print(f"  AUC (val)  : {result['metrics']['auc_val_cal']:.4f}")
    print(f"  Brier(val) : {result['metrics']['brier_val_cal']:.4f}")


if __name__ == "__main__":
    main()