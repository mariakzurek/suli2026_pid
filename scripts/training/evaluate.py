"""
evaluate.py — Per-(p,θ)-bin threshold sweep + baseline comparison on the test set.

WHAT IT DOES
------------
Loads the calibrated model from train_bdt.py and the test parquet from
build_dataset.py.  Sweeps classification thresholds in each (p, θ) bin and
produces:

  per_bin_sweep.csv
    One row per (p_bin, theta_bin, threshold) with:
      eff_K  = N(score>t & label==1) / N(label==1)
      C_pi   = N(score>t & label==0) / N(score>t & label.notna())
      C_p    = N(score>t & mc_matching_pid==2212) / N(score>t)

  comparison_summary.csv
    Per-bin matched-efficiency comparison and matched-contamination comparison
    between the baseline chi2pid cut and the BDT.  Matched-eff: find the BDT
    threshold that achieves the same eff_K as the baseline; quote BDT C_pi at
    that threshold.  Matched-contam: reverse.

  contam_vs_ptheta_baseline_vs_bdt.png
    1×2 viridis heatmap (shared color scale) of C_pi at the threshold that
    matches the baseline eff_K.  Bins with n_label<50 overlaid with '///'
    hatching (n<50 low-stat policy).  Headline plot for Cooper's Week-4 deliverable.

  cp_to_K_map.png
    1×1 viridis heatmap of C_p = C^{p→K} at matched-eff threshold — Cooper's
    Phase-4 input for the proton-contamination decision.

WHEN TO USE
-----------
Run after train_bdt.py. Does not require rerunning if you want to re-plot with
different edges — just pass new --p-edges / --theta-edges and --overwrite.

PITFALLS
--------
* The test set has nullable Int8 labels; eff_K and C_pi are computed only over
  label.notna() rows (i.e., EB-K+ that are true K or true π).  Proton rows
  (mc_matching_pid==2212) contribute to C_p but not to eff_K or C_pi.
* Bins with n_label < 50 (labeled rows) are marked low-stat and shown with
  '///' hatching in heatmaps. Do not quote numbers from these bins.
* Baseline comparison uses passes_kplus_chi2pid_cut from
  scripts.baseline_chi2pid; it expects two numpy arrays (chi2pid, p).
* This script evaluates on the TEST set only. It never loads train.parquet
  or val.parquet.

Usage:
  python scripts/training/evaluate.py \\
      --model /volatile/clas12/$USER/SULI/models/v01/model.joblib \\
      --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \\
      --outdir /volatile/clas12/$USER/SULI/eval/v01 \\
      --p-edges 1.0 2.0 3.0 4.0 5.0 \\
      --theta-edges 5 15 25 35 \\
      --overwrite
"""

import argparse
import pathlib
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# Must be invoked from the repo root (~/CLAS/SULI/suli2026_pid/) so that the
# package import below resolves.  The SLURM worker (cd "${REPO_ROOT}/suli2026_pid")
# and the interactive Tier-2 workflow both satisfy this requirement.
from scripts.baseline_chi2pid import passes_kplus_chi2pid_cut

# ──────────────────────────────────────────────────────────────────────────────
# Default (p, θ) bin edges from Week 2 convention.
# These match the audit grid; change with --p-edges and --theta-edges.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_P_EDGES     = [1.0, 2.0, 3.0, 4.0, 5.0]   # GeV/c
DEFAULT_THETA_EDGES = [5.0, 15.0, 25.0, 35.0]      # degrees

# Minimum labeled rows (label.notna()) per bin to report metrics.
LOW_STAT_THRESHOLD = 50

PID_KPLUS  = 321
PID_PIPLUS = 211
PID_PROTON = 2212


# ──────────────────────────────────────────────────────────────────────────────
# Per-bin metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bin_label(p_lo, p_hi, t_lo, t_hi):
    """Human-readable bin label for plot axes."""
    return f"p=[{p_lo:.1f},{p_hi:.1f})\nθ=[{t_lo:.0f},{t_hi:.0f})"


def _bin_metrics_at_threshold(df_bin: pd.DataFrame, threshold: float) -> dict:
    """
    Compute eff_K, C_pi, C_p at a single threshold for one (p,θ) bin.

    WHAT IT DOES
    ------------
    eff_K = N(score>t & label==1) / N(label==1)   [K efficiency]
    C_pi  = N(score>t & label==0) / N(score>t & label.notna())   [π contamination]
    C_p   = N(score>t & mc_matching_pid==2212) / N(score>t)      [proton contamination]

    Denominator conventions:
    - eff_K: denominator is all true-K rows (label==1), regardless of score.
    - C_pi: denominator is all rows accepted by the BDT with known label
      (not proton/unmatched). This is the contamination among labeled rows.
    - C_p: denominator is ALL rows accepted by the BDT (including protons
      and unmatched). This is the proton-to-K mis-ID rate in the BDT output.

    PITFALLS
    --------
    * label.notna() in Int8 nullable arrays requires pd.notna() or .isna().
    * Returns NaN for any metric whose denominator is 0.
    """
    score = df_bin["score"].to_numpy(dtype=np.float64)
    label = df_bin["label"]
    mcp = df_bin["mc_matching_pid"].to_numpy(dtype=np.int32)

    label_notna = pd.notna(label)
    label_vals = np.where(label_notna, label.astype("float64"), np.nan)

    accepted = score > threshold

    n_K        = np.sum(label_vals == 1)
    n_acc      = np.sum(accepted)
    n_acc_lab  = np.sum(accepted & label_notna.to_numpy())

    eff_K = np.sum(accepted & (label_vals == 1)) / n_K if n_K > 0 else np.nan
    C_pi  = np.sum(accepted & (label_vals == 0)) / n_acc_lab \
            if n_acc_lab > 0 else np.nan
    C_p   = np.sum(accepted & (mcp == PID_PROTON)) / n_acc \
            if n_acc > 0 else np.nan

    return {
        "eff_K": eff_K,
        "C_pi": C_pi,
        "C_p": C_p,
        "n_accepted": int(n_acc),
        "n_label": int(n_K + np.sum(label_vals == 0)),
    }


def _baseline_metrics(df_bin: pd.DataFrame) -> dict:
    """
    Compute baseline chi2pid cut metrics for one bin.

    Uses passes_kplus_chi2pid_cut(chi2pid, p) — the pass-2 K+ cut.
    Returns eff_K, C_pi, C_p at the baseline cut.
    """
    chi2 = df_bin["chi2pid"].to_numpy(dtype=np.float64)
    p    = df_bin["p"].to_numpy(dtype=np.float64)
    label = df_bin["label"]
    mcp  = df_bin["mc_matching_pid"].to_numpy(dtype=np.int32)

    accepted = passes_kplus_chi2pid_cut(chi2, p)

    label_notna = pd.notna(label)
    label_vals = np.where(label_notna, label.astype("float64"), np.nan)

    n_K       = np.sum(label_vals == 1)
    n_acc     = np.sum(accepted)
    n_acc_lab = np.sum(accepted & label_notna.to_numpy())

    eff_K = np.sum(accepted & (label_vals == 1)) / n_K if n_K > 0 else np.nan
    C_pi  = np.sum(accepted & (label_vals == 0)) / n_acc_lab \
            if n_acc_lab > 0 else np.nan
    C_p   = np.sum(accepted & (mcp == PID_PROTON)) / n_acc \
            if n_acc > 0 else np.nan

    return {
        "baseline_eff_K": eff_K,
        "baseline_C_pi": C_pi,
        "baseline_C_p": C_p,
        "baseline_n_accepted": int(n_acc),
    }


def _interpolate_threshold(sweep_df: pd.DataFrame, target_metric: str,
                            target_value: float, return_metric: str) -> tuple:
    """
    Interpolate: find the threshold where target_metric == target_value
    and return the corresponding value of return_metric.

    Returns (threshold_found, return_metric_value) or (NaN, NaN) if
    interpolation is not possible (e.g., target_value out of range).

    Uses linear interpolation between the two sweep rows that bracket
    the target value.  The sweep_df must be sorted by threshold ascending,
    so target_metric decreases monotonically (eff_K is a non-increasing
    function of threshold).
    """
    vals = sweep_df[target_metric].dropna()
    if len(vals) < 2:
        return np.nan, np.nan

    # eff_K is non-increasing with threshold; C_pi also non-increasing.
    # Find the pair of rows that brackets target_value.
    thresholds = sweep_df["threshold"].to_numpy()
    metric_vals = sweep_df[target_metric].to_numpy()
    ret_vals = sweep_df[return_metric].to_numpy()

    # Walk from low to high threshold (eff_K decreasing).
    for i in range(len(metric_vals) - 1):
        v0, v1 = metric_vals[i], metric_vals[i + 1]
        if np.isnan(v0) or np.isnan(v1):
            continue
        # Check if target is bracketed
        if (v0 >= target_value >= v1) or (v0 <= target_value <= v1):
            # Linear interpolation
            if abs(v1 - v0) < 1e-12:
                t_interp = thresholds[i]
                r_interp = ret_vals[i]
            else:
                frac = (target_value - v0) / (v1 - v0)
                t_interp = thresholds[i] + frac * (thresholds[i + 1] - thresholds[i])
                r_interp = ret_vals[i] + frac * (ret_vals[i + 1] - ret_vals[i])
            return float(t_interp), float(r_interp)

    return np.nan, np.nan


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _heatmap(
    ax,
    data: np.ndarray,
    mask_lowstat: np.ndarray,
    p_edges: list,
    theta_edges: list,
    vmin: float,
    vmax: float,
    title: str,
    cbar_ax=None,
    fig=None,
) -> None:
    """
    Draw a (n_p_bins × n_theta_bins) viridis heatmap with low-stat masking.

    Low-stat bins (mask_lowstat==True) are overlaid with a '///' hatch
    pattern (unfilled, drawn on top of the colormap cell) so the reader
    immediately sees that the cell should not be quoted.  Valid cells show
    their numeric value as white text.

    WHAT IT DOES
    ------------
    Draws one panel of the contamination heatmap.  Each cell shows the C_pi
    (or C_p) value at the matched-efficiency threshold for that (p, θ) bin.
    Low-stat bins (n_label < 50) and NaN cells get a '///' hatch overlay so
    the reader doesn't accidentally quote uninformative numbers.

    PITFALLS
    --------
    * data shape must be (n_p_bins, n_theta_bins) with p indexing rows and
      theta indexing columns.
    * NaN cells (metric computation failed) are also hatched.
    """
    n_p = len(p_edges) - 1
    n_t = len(theta_edges) - 1

    cmap = plt.cm.viridis.copy()

    # Plot valid cells; low-stat / NaN cells still receive a colormap value
    # (the nearest valid value or 0) but are visually overridden by the hatch.
    data_plot = np.where(mask_lowstat | np.isnan(data), 0.0, data)

    im = ax.imshow(
        data_plot,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=[0, n_t, 0, n_p],
    )

    # Overlay hatching on low-stat / NaN bins; annotate valid bins with value.
    for pi in range(n_p):
        for ti in range(n_t):
            if mask_lowstat[pi, ti] or np.isnan(data[pi, ti]):
                # Hatch patch: no fill, just the '///' pattern over the cell.
                ax.add_patch(plt.Rectangle(
                    (ti, pi), 1, 1,
                    fill=False,
                    hatch="///",
                    edgecolor="white",
                    linewidth=0.5,
                ))
            else:
                ax.text(ti + 0.5, pi + 0.5, f"{data[pi, ti]:.2f}",
                        ha="center", va="center", fontsize=7, color="white")

    # Axis labels: theta on x, p on y.
    ax.set_xticks(np.arange(n_t) + 0.5)
    ax.set_xticklabels(
        [f"{theta_edges[j]:.0f}–{theta_edges[j+1]:.0f}°" for j in range(n_t)],
        fontsize=8,
    )
    ax.set_yticks(np.arange(n_p) + 0.5)
    ax.set_yticklabels(
        [f"{p_edges[i]:.1f}–{p_edges[i+1]:.1f}" for i in range(n_p)],
        fontsize=8,
    )
    ax.set_xlabel("θ bin (deg)", fontsize=9)
    ax.set_ylabel("p bin (GeV/c)", fontsize=9)
    ax.set_title(title, fontsize=10)

    if fig is not None and cbar_ax is not None:
        fig.colorbar(im, cax=cbar_ax)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_path: pathlib.Path,
    test_path: pathlib.Path,
    p_edges: list,
    theta_edges: list,
    outdir: pathlib.Path,
    threshold_grid=None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Per-bin threshold sweep and baseline comparison on the test set.

    WHAT IT DOES
    ------------
    1. Loads the model and test parquet.
    2. Adds BDT scores to the test set.
    3. For each (p, θ) bin: sweeps thresholds, computes (eff_K, C_pi, C_p),
       and computes baseline chi2pid metrics.
    4. For each bin: interpolates the BDT threshold that matches baseline
       eff_K; records matched BDT C_pi.  Does the reverse (matched-contam).
    5. Writes per_bin_sweep.csv and comparison_summary.csv.
    6. Produces both headline heatmap PNGs and READMEs.

    Parameters
    ----------
    model_path : path to model.joblib produced by train_bdt.py
    test_path : path to test.parquet produced by build_dataset.py
    p_edges : list of momentum bin edges (GeV/c); default [1,2,3,4,5]
    theta_edges : list of polar-angle bin edges (deg); default [5,15,25,35]
    outdir : directory to write outputs
    threshold_grid : optional (low, high, n) tuple for threshold sweep;
        default: np.linspace(0.01, 0.99, 99)
    overwrite : if False, error if per_bin_sweep.csv already exists

    Returns
    -------
    comparison_summary DataFrame

    PITFALLS
    --------
    * Never use train.parquet or val.parquet here — test only.
    * Bins where n_label < LOW_STAT_THRESHOLD are overlaid with '///' hatching.
    * The headline plot uses shared color scale (vmin=0, vmax determined by
      the non-low-stat BDT bin with the highest C_pi); if all bins are
      low-stat, the plot has no colored cells.
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sweep_csv_path = outdir / "per_bin_sweep.csv"
    if sweep_csv_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {sweep_csv_path}. Pass overwrite=True."
        )

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    model = joblib.load(str(model_path))

    # ── Load test set ──────────────────────────────────────────────────────────
    print(f"Loading test set: {test_path}")
    df = pd.read_parquet(str(test_path))
    print(f"  Test rows: {len(df):,}")

    # Read feature list from model (via the underlying LightGBM booster).
    # The calibrated wrapper stores feature names on the base estimator.
    try:
        # CalibratedClassifierCV stores the base estimator as .estimator
        base_clf = model.estimator
        feature_names = base_clf.booster_.feature_name()
    except AttributeError:
        # Fallback: try model directly
        try:
            feature_names = model.booster_.feature_name()
        except AttributeError:
            # Last resort: use all float columns except known non-feature ones.
            non_feat = {"p", "theta", "phi", "vz", "sector", "chi2pid",
                        "pid", "mc_matching_pid", "label"}
            feature_names = [c for c in df.columns
                             if c not in non_feat and df[c].dtype in (np.float32, np.float64)]
            print(f"  WARNING: Could not read feature names from model; "
                  f"using {len(feature_names)} float columns as features.",
                  file=sys.stderr)

    print(f"  Features ({len(feature_names)}): {feature_names}")

    X_test = df[feature_names].to_numpy(dtype=np.float32, na_value=np.nan)
    scores = model.predict_proba(X_test)[:, 1]
    df = df.copy()
    df["score"] = scores

    # ── Threshold grid ─────────────────────────────────────────────────────────
    if threshold_grid is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    else:
        lo, hi, n = threshold_grid
        thresholds = np.linspace(lo, hi, n)

    # ── Per-bin sweep ──────────────────────────────────────────────────────────
    sweep_rows = []
    comparison_rows = []

    n_p = len(p_edges) - 1
    n_t = len(theta_edges) - 1

    # Arrays for heatmaps: C_pi at matched-eff threshold (BDT vs baseline)
    bdt_cpi_matched     = np.full((n_p, n_t), np.nan)
    baseline_cpi_arr    = np.full((n_p, n_t), np.nan)
    bdt_cp_matched      = np.full((n_p, n_t), np.nan)
    mask_lowstat        = np.ones((n_p, n_t), dtype=bool)   # True = low-stat

    for pi, (p_lo, p_hi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
        for ti, (t_lo, t_hi) in enumerate(zip(theta_edges[:-1], theta_edges[1:])):
            bin_mask = (
                (df["p"] >= p_lo) & (df["p"] < p_hi) &
                (df["theta"] >= t_lo) & (df["theta"] < t_hi)
            )
            df_bin = df[bin_mask].copy()

            label_notna = pd.notna(df_bin["label"])
            n_label = int(label_notna.sum())

            # Baseline metrics
            bl = _baseline_metrics(df_bin)

            # Sweep
            bin_sweep = []
            for t in thresholds:
                m = _bin_metrics_at_threshold(df_bin, t)
                row = {
                    "p_lo": p_lo, "p_hi": p_hi,
                    "theta_lo": t_lo, "theta_hi": t_hi,
                    "threshold": t,
                    "n_label": n_label,
                    **m,
                }
                bin_sweep.append(row)
                sweep_rows.append(row)

            sweep_bin_df = pd.DataFrame(bin_sweep).sort_values("threshold")

            # Interpolate matched-eff: BDT threshold where eff_K == baseline eff_K
            bl_eff = bl["baseline_eff_K"]
            t_matched_eff, bdt_cpi_at_matched_eff = _interpolate_threshold(
                sweep_bin_df, "eff_K", bl_eff if not np.isnan(bl_eff) else -1,
                "C_pi"
            )
            _, bdt_cp_at_matched_eff = _interpolate_threshold(
                sweep_bin_df, "eff_K", bl_eff if not np.isnan(bl_eff) else -1,
                "C_p"
            )

            # Interpolate matched-contam: threshold where C_pi == baseline C_pi
            bl_cpi = bl["baseline_C_pi"]
            t_matched_cpi, bdt_eff_at_matched_cpi = _interpolate_threshold(
                sweep_bin_df, "C_pi", bl_cpi if not np.isnan(bl_cpi) else -1,
                "eff_K"
            )

            comparison_rows.append({
                "p_lo": p_lo, "p_hi": p_hi,
                "theta_lo": t_lo, "theta_hi": t_hi,
                "n_label": n_label,
                "low_stat": n_label < LOW_STAT_THRESHOLD,
                **bl,
                "bdt_cpi_at_matched_eff":   bdt_cpi_at_matched_eff,
                "bdt_cp_at_matched_eff":    bdt_cp_at_matched_eff,
                "threshold_matched_eff":    t_matched_eff,
                "bdt_eff_at_matched_cpi":   bdt_eff_at_matched_cpi,
                "threshold_matched_cpi":    t_matched_cpi,
            })

            # Populate heatmap arrays
            if n_label >= LOW_STAT_THRESHOLD:
                mask_lowstat[pi, ti] = False
                bdt_cpi_matched[pi, ti] = bdt_cpi_at_matched_eff
                baseline_cpi_arr[pi, ti] = bl["baseline_C_pi"]
                bdt_cp_matched[pi, ti] = bdt_cp_at_matched_eff

            print(
                f"  bin p=[{p_lo:.1f},{p_hi:.1f}) θ=[{t_lo:.0f},{t_hi:.0f}): "
                f"n_label={n_label} | "
                f"baseline eff_K={bl_eff:.3f} C_pi={bl_cpi:.3f} | "
                f"BDT C_pi(matched-eff)={bdt_cpi_at_matched_eff:.3f}"
                if not np.isnan(bl_eff) and not np.isnan(bdt_cpi_at_matched_eff)
                else
                f"  bin p=[{p_lo:.1f},{p_hi:.1f}) θ=[{t_lo:.0f},{t_hi:.0f}): "
                f"n_label={n_label} (low-stat or NaN)"
            )

    # ── Write CSVs ─────────────────────────────────────────────────────────────
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(str(sweep_csv_path), index=False)
    print(f"  per_bin_sweep.csv → {sweep_csv_path}")

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = outdir / "comparison_summary.csv"
    comp_df.to_csv(str(comp_csv_path), index=False)
    print(f"  comparison_summary.csv → {comp_csv_path}")

    # ── Headline heatmap: C_pi at matched-eff (baseline vs BDT) ───────────────
    # Shared color scale: vmin=0, vmax=max of non-low-stat BDT values.
    valid_vals = bdt_cpi_matched[~mask_lowstat & ~np.isnan(bdt_cpi_matched)]
    vmax = float(valid_vals.max()) if len(valid_vals) > 0 else 1.0
    vmax = max(vmax, float(
        np.nanmax(baseline_cpi_arr[~mask_lowstat]) if np.any(~mask_lowstat) else vmax
    ))

    fig, axes = plt.subplots(
        1, 2, figsize=(12, max(3, n_p * 1.2)),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    _heatmap(
        axes[0], baseline_cpi_arr, mask_lowstat,
        p_edges, theta_edges,
        vmin=0.0, vmax=vmax,
        title="Baseline chi2pid\nC_π→K at baseline eff_K",
        cbar_ax=None, fig=None,
    )
    im_bdt = None
    _heatmap(
        axes[1], bdt_cpi_matched, mask_lowstat,
        p_edges, theta_edges,
        vmin=0.0, vmax=vmax,
        title="BDT (calibrated)\nC_π→K at matched eff_K",
        cbar_ax=None, fig=None,
    )

    # Add shared colorbar manually.
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="C_π→K (π contamination)")

    fig.suptitle(
        "π→K contamination at matched eff_K: baseline vs BDT\n"
        "(gray = n_label < 50; values shown in cells)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.91, 1])
    heatmap_path = outdir / "contam_vs_ptheta_baseline_vs_bdt.png"
    fig.savefig(str(heatmap_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  contam_vs_ptheta_baseline_vs_bdt.png → {heatmap_path}")

    # ── C_p map: proton contamination at matched-eff ───────────────────────────
    valid_cp = bdt_cp_matched[~mask_lowstat & ~np.isnan(bdt_cp_matched)]
    vmax_cp = float(valid_cp.max()) if len(valid_cp) > 0 else 1.0

    fig2, ax2 = plt.subplots(figsize=(max(4, n_t * 1.5), max(3, n_p * 1.2)))
    _heatmap(
        ax2, bdt_cp_matched, mask_lowstat,
        p_edges, theta_edges,
        vmin=0.0, vmax=vmax_cp,
        title="C^{p→K} (proton contamination in BDT output at matched eff_K)\n"
              "Cooper's Phase-4 decision input",
        cbar_ax=None, fig=None,
    )
    norm_cp = mcolors.Normalize(vmin=0.0, vmax=vmax_cp)
    sm_cp = plt.cm.ScalarMappable(cmap="viridis", norm=norm_cp)
    sm_cp.set_array([])
    fig2.colorbar(sm_cp, ax=ax2, label="C_p→K (proton contamination)")
    fig2.tight_layout()
    cp_map_path = outdir / "cp_to_K_map.png"
    fig2.savefig(str(cp_map_path), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  cp_to_K_map.png → {cp_map_path}")

    # ── README ─────────────────────────────────────────────────────────────────
    readme = "\n".join([
        "# evaluate.py run provenance",
        "",
        f"Model: `{model_path}`",
        f"Test set: `{test_path}`",
        f"p edges: {p_edges}",
        f"theta edges: {theta_edges} deg",
        f"Threshold grid: {len(thresholds)} points [{thresholds[0]:.3f}, {thresholds[-1]:.3f}]",
        f"Low-stat threshold: n_label < {LOW_STAT_THRESHOLD}",
        "",
        "## Outputs",
        "- `per_bin_sweep.csv` — eff_K, C_pi, C_p at each threshold for each bin",
        "- `comparison_summary.csv` — matched-eff and matched-contam comparison",
        "- `contam_vs_ptheta_baseline_vs_bdt.png` — headline heatmap (shared scale)",
        "- `cp_to_K_map.png` — C^{p→K} at matched eff_K (Phase-4 input)",
        "",
        "## Metric definitions",
        "eff_K  = N(score>t & label==1) / N(label==1)",
        "C_pi   = N(score>t & label==0) / N(score>t & label.notna())",
        "C_p    = N(score>t & mc_matching_pid==2212) / N(score>t)",
        "",
        "Baseline: passes_kplus_chi2pid_cut (scripts/baseline_chi2pid.py).",
        "Matched-eff: BDT threshold where eff_K equals baseline eff_K; report BDT C_pi.",
        "Matched-contam: BDT threshold where C_pi equals baseline C_pi; report BDT eff_K.",
    ])
    (outdir / "README.md").write_text(readme + "\n")

    return comp_df


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
        "--model",
        required=True,
        help="Path to model.joblib produced by train_bdt.py.",
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory produced by build_dataset.py; must contain test.parquet.",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory for CSVs and plots.",
    )
    p.add_argument(
        "--p-edges",
        type=float,
        nargs="+",
        default=DEFAULT_P_EDGES,
        metavar="EDGE",
        help=f"Momentum bin edges in GeV/c (default: {DEFAULT_P_EDGES}).",
    )
    p.add_argument(
        "--theta-edges",
        type=float,
        nargs="+",
        default=DEFAULT_THETA_EDGES,
        metavar="EDGE",
        help=f"Polar-angle bin edges in degrees (default: {DEFAULT_THETA_EDGES}).",
    )
    p.add_argument(
        "--threshold-grid",
        type=float,
        nargs=3,
        default=None,
        metavar=("LOW", "HIGH", "N"),
        help="Threshold sweep: LOW HIGH N (default: 0.01 0.99 99).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing outputs.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    model_path   = pathlib.Path(args.model)
    dataset_dir  = pathlib.Path(args.dataset_dir)
    outdir       = pathlib.Path(args.outdir)
    test_path    = dataset_dir / "test.parquet"

    for path in (model_path, test_path):
        if not path.exists():
            print(f"ERROR: Required file not found: {path}", file=sys.stderr)
            sys.exit(1)

    threshold_grid = None
    if args.threshold_grid is not None:
        lo, hi, n = args.threshold_grid
        threshold_grid = (lo, hi, int(n))

    comp_df = evaluate_model(
        model_path=model_path,
        test_path=test_path,
        p_edges=args.p_edges,
        theta_edges=args.theta_edges,
        outdir=outdir,
        threshold_grid=threshold_grid,
        overwrite=args.overwrite,
    )

    print("\nDone.")
    print(f"  Comparison summary: {outdir / 'comparison_summary.csv'}")
    print(f"  Headline plot:      {outdir / 'contam_vs_ptheta_baseline_vs_bdt.png'}")
    print(f"  C_p map:            {outdir / 'cp_to_K_map.png'}")


if __name__ == "__main__":
    main()
