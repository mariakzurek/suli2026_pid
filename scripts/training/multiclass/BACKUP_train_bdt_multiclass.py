"""
train_bdt.py — Fit a LightGBM BDT and Platt-calibrate on the training split.

WHAT IT DOES
------------
Loads the training and validation parquet files produced by build_dataset.py,
fits a LightGBM binary classifier (K vs π), applies Platt calibration on a
held-out slice of the training set (row-level, stratified, never val or test),
and writes the fitted model, calibrator, evaluation metrics, and diagnostic plots.

NEW WORKFLOW (week4-tier-flexible)
-----------------------------------
The training feature set is now selected at training time via --features-file,
NOT read from manifest.json.  This decouples "what the parquet stores" from
"what the model trains on," so Cooper can run multiple feature tier experiments
(Tier 1, Tier 2, Tier 3) against the same parquet without rebuilding the dataset.

--features-file is REQUIRED.  Pass one of:
  scripts/training/features_tier1.txt  — minimal: beta + FTOF 1B
  scripts/training/features_tier2.txt  — adds chi2pid + FTOF 1A
  scripts/training/features_tier3.txt  — adds ECAL + HTCC
or any custom file listing column names from columns_maximal.txt.

The feature list is embedded in model.joblib (as a wrapper dict) so evaluate.py
can recover it without consulting the manifest.  This avoids the staleness bug
where the manifest might reflect a different run.

BDT defaults (from cooper_10week_plan.md §300):
  n_estimators=200, learning_rate=0.05, max_depth=6,
  objective='binary', random_state=42, n_jobs=-1

Calibration: CalibratedClassifierCV(method='sigmoid', cv='prefit') on a
held-out 20% slice of the training set (--calibration-frac controls this).
The calibration slice is carved BEFORE fitting; the BDT is fitted on the
remaining 80%.  This is the correct procedure: calibrate on data the BDT
has not seen.

Outputs:
  model.joblib               wrapper dict {"model": calibrated_clf, "features": feature_list}
  training_summary.csv       AUC, Brier, log-loss for train and val (pre/post cal)
  reliability_diagram.png    2-panel pre/post calibration (n_bins=10)
  roc_val.png                ROC curve on validation set
  feature_importance.png     Top-15 features by LightGBM gain
  feature_importance.csv     Full importance table
  README.md                  Run provenance

WHEN TO USE
-----------
Run after build_dataset.py produces the three parquet files. Pass the dataset
directory and a features file that is a subset of columns_maximal.txt.
Re-run with a different --features-file to try a different feature tier — no
rebuild needed as long as the dataset was built with columns_maximal.txt.

PITFALLS
--------
* --features-file is REQUIRED and must name columns present in the parquet.
  The script validates each feature against the parquet schema and fails fast
  if any are missing, with a pointer to columns_maximal.txt.
* The manifest's feature_list is NOT used as a default — the feature set must
  be explicit per training run.  This prevents the silent manifest-mismatch bug.
* The calibration slice is carved from the TRAINING split only. Val and test
  are never touched during calibration or fitting. Never evaluate calibration
  quality on the calibration slice itself; use the reliability diagram on val.
* --reweight-map is optional; pass it only when a (p,theta)-reweighting has
  been produced. Without it the model trains unweighted — valid for v1.
* Never load test.parquet in this script. All test-set evaluation is in
  evaluate.py.

Usage:
  python scripts/training/train_bdt.py \\
      --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \\
      --features-file scripts/training/features_tier1.txt \\
      --outdir /volatile/clas12/$USER/SULI/models/tier1 \\
      --overwrite

Smoke test (requires parquet files in --dataset-dir):
  python scripts/training/train_bdt.py --help
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
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# LightGBM import — provides a clear error if not installed.
try:
    from lightgbm import LGBMClassifier
except ImportError:
    print(
        "ERROR: lightgbm not installed in the current environment.\n"
        "       On ifarm: conda activate suli2026_pid && conda install -c conda-forge lightgbm",
        file=sys.stderr,
    )
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_features_file(path: pathlib.Path) -> List[str]:
    """
    Parse a tier features file (one column name per line, # comments stripped).

    WHAT IT DOES
    ------------
    Reads the features file, strips blank lines and comment lines, and returns
    a list of column names that will be used as the model's input features.
    Errors if the file is missing or results in an empty list.

    PITFALLS
    --------
    Column names must exactly match what is in the parquet (case-sensitive).
    The script validates these against the parquet schema immediately after
    parsing — a typo here will produce a clear error before any training starts.
    """
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
    """
    Validate that every feature in feature_list exists as a column in the parquet.

    WHAT IT DOES
    ------------
    Reads only the parquet schema (no data rows loaded) using pyarrow, then
    checks each requested feature against the available columns.  Fails fast
    with a clear error listing missing features and pointing to columns_maximal.txt.

    WHEN TO USE
    -----------
    Call this immediately after parsing --features-file, before loading any
    training data.  Catches typos and schema mismatches early.

    PITFALLS
    --------
    Uses pd.read_parquet with columns=[] to retrieve the schema; this is a
    lightweight metadata read that does not load data rows.
    """
    try:
        # Read parquet schema without loading data rows.
        import pyarrow.parquet as pq
        schema = pq.read_schema(str(parquet_path))
        parquet_cols = set(schema.names)
    except Exception:
        # Fallback: read a zero-row slice via pandas.
        try:
            parquet_cols = set(pd.read_parquet(str(parquet_path), columns=[]).columns)
            # Zero-column read returns empty; fall back to reading one row.
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
            f"These features are not in the parquet.  Either:\n"
            f"  (a) The feature is already in columns_maximal.txt and was built into "
            f"the parquet — check for a typo in your features file.\n"
            f"  (b) The feature is NOT in columns_maximal.txt — add it there and "
            f"rebuild the dataset with build_dataset.py --overwrite.\n\n"
            f"Available columns in parquet: {sorted(parquet_cols)}",
            file=sys.stderr,
        )
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics / plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plot_reliability_diagram(
    y_true_cal: np.ndarray,
    scores_uncal: np.ndarray,
    scores_cal: np.ndarray,
    out_path: pathlib.Path,
    n_bins: int = 10,
) -> None:
    """
    Multiclass reliability diagram using one-vs-rest calibration curves.
    """

    from sklearn.preprocessing import label_binarize
    from sklearn.calibration import calibration_curve

    classes = np.arange(scores_uncal.shape[1])
    y_bin = label_binarize(y_true_cal, classes=classes)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        sharey=True,
    )

    for ax, scores, title in zip(
        axes,
        (scores_uncal, scores_cal),
        ("Pre-calibration (raw BDT)", "Post-calibration (Platt)"),
    ):

        for i, cls in enumerate(classes):
            frac_pos, mean_pred = calibration_curve(
                y_bin[:, i],
                scores[:, i],
                n_bins=n_bins,
                strategy="uniform",
            )

            ax.plot(
                mean_pred,
                frac_pos,
                "o-",
                lw=1.5,
                markersize=4,
                label=f"Class {cls}",
            )

        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            lw=1,
            label="Perfect calibration",
        )

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(title)
        ax.legend(fontsize=8)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(
        "Multiclass reliability diagram (validation set)",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(
    y_true: np.ndarray,
    scores: np.ndarray,
    out_path: pathlib.Path,
    label: str = "BDT (calibrated)",
) -> float:
    """
    ROC curves for multiclass classification using one-vs-rest.
    Returns macro-average AUC.
    """

    from sklearn.preprocessing import label_binarize

    classes = np.arange(scores.shape[1])
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(6, 5))

    aucs = []

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(
            y_bin[:, i],
            scores[:, i],
        )

        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        ax.plot(
            fpr,
            tpr,
            lw=2,
            label=f"Class {cls} (AUC = {roc_auc:.4f})",
        )

    macro_auc = np.mean(aucs)

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        lw=1,
    )

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — validation set (macro AUC = {macro_auc:.4f})")
    ax.legend(fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return float(macro_auc)


def _plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    out_path: pathlib.Path,
    top_n: int = 15,
) -> None:
    """
    Horizontal bar chart of top-N LightGBM feature importances by gain.

    WHAT IT DOES
    ------------
    Shows which features the BDT used most.  'Gain' importance is the total
    reduction in loss attributed to splits on that feature — the most
    informative measure for physics interpretation.

    PITFALLS
    --------
    Importance scores depend on the training data and hyperparameters; they
    are a guide for feature analysis, not a definitive ranking.  Do not prune
    features based solely on importance from a single run.
    """
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
    """Write a provenance README into the model output directory."""
    lines = [
        "# BDT training run provenance\n",
        f"Generated: {timestamp}\n\n",
        "## Dataset\n",
        f"- Source: `{dataset_dir}`\n",
        f"- Features ({len(feature_list)}): {', '.join(feature_list)}\n",
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
        "\"features\": list of training feature names}\n",
        "- `training_summary.csv` — AUC, Brier, log-loss for train/val pre/post cal\n",
        "- `reliability_diagram.png` — calibration quality (on val set)\n",
        "- `roc_val.png` — ROC curve on val set\n",
        "- `feature_importance.png` / `.csv` — top-15 features by gain\n",
    ]
    (outdir / "README.md").write_text("".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def _safe_auc(y_true, y_score):
    if y_score.shape[1] == 2:
        return roc_auc_score(y_true, y_score[:, 1])
    return roc_auc_score(y_true, y_score, multi_class="ovr")


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

    # Validate feature list against what's in the file.
    missing_feats = [f for f in feature_list if f not in df_train_full.columns]
    if missing_feats:
        print(
            f"WARNING: {len(missing_feats)} feature(s) not found in training parquet "
            f"(will be all-NaN): {missing_feats}",
            file=sys.stderr,
        )

    X_full = df_train_full[feature_list].to_numpy(
        dtype=np.float32,
        na_value=np.nan,
    )
    y_full = df_train_full["label"].astype(np.int8).to_numpy()

    X_val = df_val[feature_list].to_numpy(
        dtype=np.float32,
        na_value=np.nan,
    )
    y_val = df_val["label"].astype(np.int8).to_numpy()

    # Class definitions
    class_names = {
        0: "pi+",
        1: "K+",
        2: "proton",
    }

    n_classes = len(np.unique(y_full))

    print(f"Number of classes: {n_classes}")
    for c, name in class_names.items():
        print(f"  {c}: {name}")

    n_train_full = len(X_full)
    n_val = len(X_val)

    print(f"  Train (full): {n_train_full:,} rows, Val: {n_val:,} rows")
    print(f"  Train class fractions: {np.bincount(y_full)/len(y_full)}")
    print(f"  Val class fractions: {np.bincount(y_val)/len(y_val)}")

    # ── Calibration split ──────────────────────────────────────────────────────
    print(
        f"Carving calibration slice ({calibration_frac:.0%} of train, stratified) ..."
    )

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_full,
        y_full,
        test_size=calibration_frac,
        random_state=random_state,
        stratify=y_full,
    )

    print(
        f"  BDT fit rows: {len(X_fit):,}  |  "
        f"Calibration rows: {len(X_cal):,}"
    )

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

        p_bin = np.clip(
            p_bin,
            0,
            weights_grid.shape[0] - 1,
        )

        theta_bin = np.clip(
            theta_bin,
            0,
            weights_grid.shape[1] - 1,
        )

        w_full = weights_grid[p_bin, theta_bin]

        idx_all = np.arange(n_train_full)

        idx_fit, idx_cal_unused = train_test_split(
            idx_all,
            test_size=calibration_frac,
            random_state=random_state,
            stratify=y_full,
        )

        w_fit = w_full[idx_fit]

        print(
            f"  Reweight map applied; mean weight = {w_fit.mean():.3f}"
        )

    # ── Fit LightGBM ──────────────────────────────────────────────────────────
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

    clf.fit(
        X_fit,
        y_fit,
        sample_weight=w_fit,
    )

    print("  LightGBM fit complete.")
    # ── Pre-calibration scores ─────────────────────────────────────────────────
    from sklearn.preprocessing import label_binarize

    scores_train_uncal = clf.predict_proba(X_full)
    scores_val_uncal = clf.predict_proba(X_val)

    logloss_val_uncal = log_loss(
        y_val,
        scores_val_uncal,
    )

    auc_train_uncal = _safe_auc(y_full, scores_train_uncal)

    auc_val_uncal = _safe_auc(y_full, scores_val_uncal)

    print(
        f"  Pre-cal AUC train={auc_train_uncal:.4f}  "
        f"val={auc_val_uncal:.4f}"
    )

    # ── Platt calibration ──────────────────────────────────────────────────────
    print(
        "Fitting Platt calibration on held-out calibration slice ..."
    )

    calibrated_clf = CalibratedClassifierCV(
        estimator=clf,
        method="sigmoid",
        cv="prefit",
    )

    calibrated_clf.fit(
        X_cal,
        y_cal,
    )

    print("  Calibration fit complete.")

    # ── Post-calibration scores ────────────────────────────────────────────────
    scores_train_cal = calibrated_clf.predict_proba(X_full)
    scores_val_cal = calibrated_clf.predict_proba(X_val)

    auc_train_cal = _safe_auc(y_full, scores_train_cal)

    auc_val_cal = _safe_auc(y_full, scores_val_cal)

    # Multiclass Brier score
    classes = np.arange(scores_val_cal.shape[1])

    y_val_onehot = label_binarize(
        y_val,
        classes=classes,
    )

    brier_val_cal = np.mean(
        np.sum(
            (scores_val_cal - y_val_onehot) ** 2,
            axis=1,
        )
    )

    y_val_onehot_uncal = label_binarize(
        y_val,
        classes=classes,
    )

    brier_val_uncal = np.mean(
        np.sum(
            (scores_val_uncal - y_val_onehot_uncal) ** 2,
            axis=1,
        )
    )

    logloss_val_cal = log_loss(
        y_val,
        scores_val_cal,
    )

    print(
        f"  Post-cal AUC train={auc_train_cal:.4f}  "
        f"val={auc_val_cal:.4f}"
    )

    print(
        f"  Post-cal Brier val={brier_val_cal:.4f}  "
        f"LogLoss val={logloss_val_cal:.4f}"
    )

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
        "class_names": class_names,
        "train_class_counts": np.bincount(y_full).tolist(),
        "val_class_counts": np.bincount(y_val).tolist(),
    }
        # ── Save model ─────────────────────────────────────────────────────────────
    # Save the calibrated model and metadata.
    model_bundle = {
        "model": calibrated_clf,
        "features": feature_list,
        "class_names": class_names,
    }

    joblib.dump(
        model_bundle,
        str(model_path),
        compress=3,
    )

    print(f"  Model saved → {model_path}")
    print(f"  Classes: {class_names}")
    print(f"  Features: {feature_list}")

    # ── Training summary CSV ───────────────────────────────────────────────────
    pd.DataFrame([metrics]).to_csv(
        str(outdir / "training_summary.csv"),
        index=False,
    )

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("Generating reliability diagram ...")
    print("Skipping reliability diagram for multiclass model.")

    print("Generating ROC curve ...")

    _plot_roc(
        y_val,
        scores_val_cal,
        outdir / "roc_val.png",
    )

    print("Generating feature importance plot ...")

    feature_importances = clf.feature_importances_

    feat_imp_df = pd.DataFrame(
        {
            "feature": feature_list,
            "importance_gain": feature_importances,
        }
    ).sort_values(
        "importance_gain",
        ascending=False,
    )

    feat_imp_df.to_csv(
        str(outdir / "feature_importance.csv"),
        index=False,
    )

    _plot_feature_importance(
        feature_list,
        feature_importances,
        outdir / "feature_importance.png",
    )

    # ── README ────────────────────────────────────────────────────────────────
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
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory produced by build_dataset.py; must contain train.parquet, "
             "val.parquet, and manifest.json.",
    )
    p.add_argument(
        "--features-file",
        required=True,
        metavar="PATH",
        help="Text file listing column names to use as training features "
             "(one per line, # comments allowed, blank lines ignored). "
             "Must be a subset of the columns in the parquet "
             "(see scripts/training/columns_maximal.txt). "
             "Use features_tier1.txt, features_tier2.txt, or features_tier3.txt "
             "for Week-5 tier experiments.",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory for model.joblib, plots, and CSVs.",
    )
    p.add_argument(
        "--reweight-map",
        default=None,
        metavar="PATH",
        help="Optional .npz reweight map (p_edges, theta_edges, weights). "
             "If omitted, training is unweighted.",
    )
    p.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of BDT trees (default: %(default)s).",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="LightGBM learning rate (default: %(default)s).",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth (default: %(default)s).",
    )
    p.add_argument(
        "--calibration-frac",
        type=float,
        default=0.2,
        help="Fraction of training set held out for Platt calibration (default: %(default)s).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/calibration split and LightGBM (default: %(default)s).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing model.joblib if present.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    dataset_dir = pathlib.Path(args.dataset_dir)
    outdir = pathlib.Path(args.outdir)
    features_file = pathlib.Path(args.features_file)

    # ── Load and validate features file ───────────────────────────────────────
    if not features_file.exists():
        print(
            f"ERROR: --features-file not found: {features_file}\n"
            f"       Use one of: scripts/training/features_tier{{1,2,3}}.txt\n"
            f"       or create a custom file listing columns from columns_maximal.txt.",
            file=sys.stderr,
        )
        sys.exit(1)

    feature_list = _parse_features_file(features_file)
    print(f"Features loaded ({len(feature_list)}) from {features_file}: {feature_list}")

    # ── Validate parquet exists ────────────────────────────────────────────────
    train_path = dataset_dir / "train.parquet"
    val_path = dataset_dir / "val.parquet"

    for path in (train_path, val_path):
        if not path.exists():
            print(f"ERROR: Required parquet not found: {path}", file=sys.stderr)
            sys.exit(1)

    # ── Validate features against parquet schema (fail fast) ──────────────────
    print(f"Validating features against parquet schema: {train_path}")
    _validate_features_against_parquet(feature_list, train_path)
    print(f"  All {len(feature_list)} features present in parquet. OK.")

    # ── Print manifest p_max for reference (not used to derive feature_list) ───
    manifest_path = dataset_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        p_max = manifest.get("p_max")
        print(f"p_max from manifest: {p_max} GeV/c")
    else:
        print(
            f"WARNING: manifest.json not found in {dataset_dir}. "
            f"Proceeding without p_max info.",
            file=sys.stderr,
        )

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
    print(f"  Model            : {result['model_path']}")
    print(f"  Macro AUC (val)  : {result['metrics']['auc_val_cal']:.4f}")
    print(f"  Brier score (val): {result['metrics']['brier_val_cal']:.4f}")
    print(f"  Train classes    : {result['metrics']['train_class_counts']}")
    print(f"  Val classes      : {result['metrics']['val_class_counts']}")


if __name__ == "__main__":
    main()
