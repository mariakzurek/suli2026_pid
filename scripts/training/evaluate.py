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
* Feature names are read from model.joblib (wrapper dict {"model": ...,
  "features": [...]}) produced by the updated train_bdt.py.  Old bare-estimator
  model.joblib files fall back to manifest.json with a deprecation warning.

Usage:
  python scripts/training/evaluate.py \\
      --model /volatile/clas12/$USER/SULI/models/v01/model.joblib \\
      --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \\
      --outdir /volatile/clas12/$USER/SULI/eval/v01 \\
      --p-edges 1.0 2.0 3.0 4.0 5.0 \\
      --theta-edges 5 15 25 35 \\
      --overwrite
"""
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import pathlib


import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import json

import sys
from pathlib import Path

allPlots=[]

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Must be invoked from the repo root (~/CLAS/SULI/suli2026_pid/) so that the
# package import below resolves.  The SLURM worker (cd "${REPO_ROOT}/suli2026_pid")
# and the interactive Tier-2 workflow both satisfy this requirement.
from scripts.baseline_chi2pid import passes_kplus_chi2pid_cut


# ──────────────────────────────────────────────────────────────────────────────
# Default (p, θ) bin edges from Week 2 convention.
# These match the audit grid; change with --p-edges and --theta-edges.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_P_EDGES     = [0.5, 1.4, 2.3, 3.2]   # GeV/c
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
    thresholds=None,
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
                if thresholds is not None and not np.isnan(thresholds[pi, ti]):
                    text = f"Cont:{data[pi, ti]:.2f} \n BDT:{thresholds[pi, ti]:.2f}"
                else:
                    text = f"{data[pi, ti]:.2f}"

                ax.text(
                    ti + 0.5,
                    pi + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

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


def plot_ml_contamination_simple(matched,
                                 pStart, pEnd, pStep,
                                 direct,
                                 bdt_cut):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # fixed theta bins: 5 bins from 5 to 35 degrees
    t_edges = np.linspace(5, 35, 6)

    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    n_p_bins = int((pEnd - pStart) / pStep)

    for i in range(5):

        theta_lo = t_edges[i]
        theta_hi = t_edges[i + 1]

        vals = []
        errs = []

        p = pStart

        for j in range(n_p_bins):

            pCut = matched[
                (matched["p"] >= p) &
                (matched["p"] < p + pStep) &
                (matched["theta"] >= theta_lo) &
                (matched["theta"] < theta_hi) &
                (matched["score"] > bdt_cut)
            ]

            a = ((pCut["mc_matching_pid"] != 321) &
                 (pCut["pid"] == 321)).sum()

            b = (pCut["pid"] == 321).sum()

            if b != 0:
                r = a / b
            else:
                r = 0

            if a != 0 and b != 0:
                rErr = r * np.sqrt((1/a) + (1/b))
            else:
                rErr = 0

            vals.append(r)
            errs.append(rErr)

            p += pStep

        edges = np.linspace(pStart, pEnd, len(vals) + 1)
        x = (edges[:-1] + edges[1:]) / 2

        mask = np.array(vals) != 0

        ax.errorbar(
            x[mask],
            np.array(vals)[mask],
            yerr=np.array(errs)[mask],
            fmt='o',
            capsize=3,
            color=colors[i],
            label=f"{theta_lo:.0f}–{theta_hi:.0f}°"
        )

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("Contamination")
    ax.set_title("ML-based K⁺ contamination vs momentum")

    ax.plot([], [], ' ', label=f"BDT cut = {bdt_cut:.3f}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(direct + "contaminationK_ML.png", dpi=150)
    plt.close(fig)
    allPlots.append(fig)
    return fig


def plot_ml_contamination_fixed_efficiency_theta(
    matched,
    pStart, pEnd, pStep,
    target_eff=0.8,
    direct="."
):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # fixed theta bins: 5 bins from 5–35°
    t_edges = np.linspace(5, 35, 6)
    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    n_p_bins = int((pEnd - pStart) / pStep)

    for i in range(5):

        theta_lo = t_edges[i]
        theta_hi = t_edges[i + 1]

        vals = []
        errs = []

        for j in range(n_p_bins):

            p_lo = pStart + j * pStep
            p_hi = p_lo + pStep

            df_bin = matched[
                (matched["p"] >= p_lo) &
                (matched["p"] < p_hi) &
                (matched["theta"] >= theta_lo) &
                (matched["theta"] < theta_hi)
            ]

            if len(df_bin) == 0:
                vals.append(0)
                errs.append(0)
                continue

            # --- FIXED EFFICIENCY CUT ---
            k_scores = df_bin.loc[df_bin["label"] == 1, "score"]

            if len(k_scores) == 0:
                vals.append(0)
                errs.append(0)
                continue

            # threshold that gives desired efficiency
            t_cut = np.quantile(k_scores, 1 - target_eff)

            accepted = df_bin["score"] > t_cut

            # contamination: NOT true K in accepted K sample
            a = ((accepted) &
                 (df_bin["pid"] == 321) &
                 (df_bin["mc_matching_pid"] != 321)).sum()

            b = ((accepted) &
                 (df_bin["pid"] == 321)).sum()

            if b > 0:
                r = a / b
                rErr = r * np.sqrt((1/a if a > 0 else 0) + (1/b))
            else:
                r = 0
                rErr = 0

            vals.append(r)
            errs.append(rErr)

        # --- plotting ---
        edges = np.linspace(pStart, pEnd, len(vals) + 1)
        x = (edges[:-1] + edges[1:]) / 2

        mask = np.array(vals) != 0

        ax.errorbar(
            x[mask],
            np.array(vals)[mask],
            yerr=np.array(errs)[mask],
            fmt="o",
            capsize=3,
            color=colors[i],
            label=f"{theta_lo:.0f}–{theta_hi:.0f}°"
        )

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel(f"Contamination")
    ax.set_title("ML contamination at fixed K⁺ efficiency")

    ax.plot([], [], " ", label=f"efficiency = {target_eff:.2f}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{direct}/contamination_fixed_eff_theta.png", dpi=150)
    plt.close(fig)
    allPlots.append(fig)
    return fig

def plot_ml_contamination_matched_theta(
    df,
    comp_df,
    p_edges,
    theta_edges,
    direct="."
):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    # fixed theta bins (must match evaluate.py)
    n_theta = len(theta_edges) - 1
    n_p = len(p_edges) - 1

    for ti in range(n_theta):

        theta_lo = theta_edges[ti]
        theta_hi = theta_edges[ti + 1]

        vals = []
        errs = []

        for pi in range(n_p):

            p_lo = p_edges[pi]
            p_hi = p_edges[pi + 1]

            # get matching threshold from evaluate.py output
            row = comp_df[
                (comp_df["p_lo"] == p_lo) &
                (comp_df["p_hi"] == p_hi) &
                (comp_df["theta_lo"] == theta_lo) &
                (comp_df["theta_hi"] == theta_hi)
            ]

            if len(row) == 0 or np.isnan(row["threshold_matched_eff"].values[0]):
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            t_cut = row["threshold_matched_eff"].values[0]

            df_bin = df[
                (df["p"] >= p_lo) & (df["p"] < p_hi) &
                (df["theta"] >= theta_lo) & (df["theta"] < theta_hi)
            ]

            if len(df_bin) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            accepted = df_bin["score"] > t_cut

            N_acc = accepted.sum()
            if N_acc == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            N_bad = (accepted & (df_bin["mc_matching_pid"] != 321)).sum()

            r = N_bad / N_acc
            rErr = np.sqrt(r * (1 - r) / N_acc)

            vals.append(r)
            errs.append(rErr)

        # plot per theta bin
        x = np.arange(len(vals)) + 0.5

        vals = np.array(vals)
        errs = np.array(errs)

        mask = ~np.isnan(vals)

        ax.errorbar(
            x[mask],
            vals[mask],
            yerr=errs[mask],
            fmt="o",
            capsize=3,
            color=colors[ti % len(colors)],
            label=f"{theta_lo:.0f}–{theta_hi:.0f}°"
        )

    ax.set_xlabel("p bin index")
    ax.set_ylabel("Contamination")
    ax.set_title("ML contamination (matched-eff thresholds from heatmap)")
    ax.set_ylim(0, 1.1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{direct}/contamination_1d_matched_consistent.png", dpi=150)
    plt.close(fig)

    return fig

def plot_ml_efficiency_fixed_efficiency_theta(
    matched,
    pStart, pEnd, pStep,
    target_eff=0.8,
    direct="."
):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # fixed theta bins: 5 bins from 5–35°
    t_edges = np.linspace(5, 35, 6)
    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    n_p_bins = int((pEnd - pStart) / pStep)

    for i in range(5):

        theta_lo = t_edges[i]
        theta_hi = t_edges[i + 1]

        vals = []
        errs = []

        for j in range(n_p_bins):

            p_lo = pStart + j * pStep
            p_hi = p_lo + pStep

            df_bin = matched[
                (matched["p"] >= p_lo) &
                (matched["p"] < p_hi) &
                (matched["theta"] >= theta_lo) &
                (matched["theta"] < theta_hi)
            ]

            if len(df_bin) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # --- FIXED EFFICIENCY CUT ---
            k_scores = df_bin.loc[df_bin["label"] == 1, "score"]

            if len(k_scores) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # threshold that gives desired efficiency
            t_cut = np.quantile(k_scores, 1 - target_eff)

            accepted = df_bin["score"] > t_cut

            # =====================================================
            # EFFICIENCY (FIXED PART)
            # =====================================================
            mc = df_bin["mc_matching_pid"].to_numpy()

            true_K = (mc == 321)
            denom = np.sum(true_K)

            if denom == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            num = np.sum(accepted & true_K)

            r = num / denom
            rErr = np.sqrt(r * (1 - r) / denom)

            vals.append(r)
            errs.append(rErr)

        # --- plotting ---
        edges = np.linspace(pStart, pEnd, len(vals) + 1)
        x = (edges[:-1] + edges[1:]) / 2

        vals = np.array(vals)
        errs = np.array(errs)
        mask = ~np.isnan(vals)

        ax.errorbar(
            x[mask],
            vals[mask],
            yerr=errs[mask],
            fmt="o",
            capsize=3,
            color=colors[i],
            label=f"{theta_lo:.0f}–{theta_hi:.0f}°"
        )

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("Efficiency")
    ax.set_title("ML efficiency at fixed efficiency")

    ax.plot([], [], " ", label=f"target efficiency = {target_eff:.2f}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{direct}/efficiency_fixed_eff_theta.png", dpi=150)
    plt.close(fig)

    return fig

def plot_ml_contamination_matched_chi2pid_theta(
    df,
    pStart, pEnd, pStep,
    theta_edges=None,
    direct="."
):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    if theta_edges is None:
        theta_edges = np.linspace(5, 35, 6)

    fig, ax = plt.subplots()

    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    n_p_bins = int((pEnd - pStart) / pStep)

    for i in range(len(theta_edges) - 1):

        theta_lo = theta_edges[i]
        theta_hi = theta_edges[i + 1]

        vals = []
        errs = []

        for j in range(n_p_bins):

            p_lo = pStart + j * pStep
            p_hi = p_lo + pStep

            df_bin = df[
                (df["p"] >= p_lo) &
                (df["p"] < p_hi) &
                (df["theta"] >= theta_lo) &
                (df["theta"] < theta_hi)
            ]

            if len(df_bin) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # =====================================================
            # TRUE K DEFINITION (PID = 321)
            # =====================================================
            label = df_bin["label"]
            pid = df_bin["pid"]

            valid = pd.notna(label) & pd.notna(pid)

            true_k = valid.to_numpy() & (label.astype(float).to_numpy() == 1) & (pid.to_numpy() == 321)

            n_true_k = np.sum(true_k)

            if n_true_k == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # =====================================================
            # χ²PID efficiency (REFERENCE CUT)
            # =====================================================
            chi2_mask = passes_kplus_chi2pid_cut(
                df_bin["chi2pid"].to_numpy(),
                df_bin["p"].to_numpy()
            )

            chi2_eff = np.sum(chi2_mask & true_k) / n_true_k

            # =====================================================
            # BDT SCORES
            # =====================================================
            scores = df_bin["score"].to_numpy()

            if len(scores) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # =====================================================
            # FIND MATCHED THRESHOLD
            # =====================================================
            thresholds = np.linspace(0.01, 0.99, 100)

            best_t = thresholds[0]
            best_diff = 1e9

            for t in thresholds:
                bdt_eff = np.sum((scores > t) & true_k) / n_true_k
                diff = abs(bdt_eff - chi2_eff)

                if diff < best_diff:
                    best_diff = diff
                    best_t = t

            # =====================================================
            # APPLY THRESHOLD
            # =====================================================
            accepted = scores > best_t

            n_acc_true_k = np.sum(accepted & true_k)

            n_wrong = np.sum(
                accepted &
                valid.to_numpy() &
                (label.astype(float).to_numpy() == 0)
            )

            r = n_wrong / n_acc_true_k
            rErr = np.sqrt(r * (1 - r) / n_acc_true_k)

            vals.append(r)
            errs.append(rErr)

        # =========================================================
        # plotting
        # =========================================================
        edges = np.linspace(pStart, pEnd, len(vals) + 1)
        x = (edges[:-1] + edges[1:]) / 2

        vals = np.array(vals)
        errs = np.array(errs)
        mask = ~np.isnan(vals)

        ax.errorbar(
            x[mask],
            vals[mask],
            yerr=errs[mask],
            fmt="o",
            capsize=3,
            color=colors[i % len(colors)],
            label=f"{theta_lo:.0f}–{theta_hi:.0f}°"
        )

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("Contamination")
    ax.set_title("BDT contamination at CHI2PID-matched efficiency")

    ax.legend()
    fig.tight_layout()

    outpath = f"{direct}/contamination_matched_chi2pid_theta.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return fig
    
#######
def plot_ml_contamination_matched_chi2pid_theta(
    df,
    pStart, pEnd, pStep,
    theta_edges=None,
    direct="."
):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    if theta_edges is None:
        theta_edges = np.linspace(5, 35, 6)

    fig, ax = plt.subplots()

    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    n_p_bins = int((pEnd - pStart) / pStep)

    for i in range(len(theta_edges) - 1):

        theta_lo = theta_edges[i]
        theta_hi = theta_edges[i + 1]

        vals = []
        errs = []

        for j in range(n_p_bins):

            p_lo = pStart + j * pStep
            p_hi = p_lo + pStep

            df_bin = df[
                (df["p"] >= p_lo) &
                (df["p"] < p_hi) &
                (df["theta"] >= theta_lo) &
                (df["theta"] < theta_hi)
            ]

            if len(df_bin) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # =====================================================
            # TRUE K DEFINITION
            # =====================================================
            label = df_bin["label"]
            label_notna = pd.notna(label)

            true_k = label_notna.to_numpy() & (label.astype(float).to_numpy() == 1)

            n_true_k = np.sum(true_k)

            if n_true_k == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # =====================================================
            # BDT SCORES
            # =====================================================
            scores = df_bin["score"].to_numpy()

            if len(scores) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # =====================================================
            # FIND MATCHED BDT THRESHOLD
            # =====================================================
            thresholds = np.linspace(0.01, 0.99, 100)

            best_t = thresholds[0]
            best_diff = 1e9

            for t in thresholds:
                eff = np.mean(scores > t)
                diff = abs(eff - (np.sum(true_k & (scores > 0)) / n_true_k if n_true_k > 0 else 0))

                if diff < best_diff:
                    best_diff = diff
                    best_t = t

            # =====================================================
            # APPLY THRESHOLD
            # =====================================================
            accepted = scores > best_t

            n_acc_true_k = np.sum(accepted & true_k)
            n_wrong = np.sum(accepted & (label_notna.to_numpy() & (label.astype(float).to_numpy() == 0)))

            if n_true_k == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            r = n_acc_true_k / n_true_k
            rErr = np.sqrt(r * (1 - r) / n_true_k)

            vals.append(r)
            errs.append(rErr)

        # =========================================================
        # plotting
        # =========================================================
        edges = np.linspace(pStart, pEnd, len(vals) + 1)
        x = (edges[:-1] + edges[1:]) / 2

        vals = np.array(vals)
        errs = np.array(errs)
        mask = ~np.isnan(vals)

        ax.errorbar(
            x[mask],
            vals[mask],
            yerr=errs[mask],
            fmt="o",
            capsize=3,
            color=colors[i % len(colors)],
            label=f"{theta_lo:.0f}–{theta_hi:.0f}°"
        )

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("Efficiency (true K accepted / true K)")
    ax.set_title("BDT efficiency matched to true-K selection")

    ax.legend()
    fig.tight_layout()

    outpath = f"{direct}/efficiency_matched_theta.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return fig





#######


def plot_ml_eff_matched_chi2pid_theta(
    df,
    pStart, pEnd, pStep,
    theta_edges=None,
    direct="."
):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    if theta_edges is None:
        theta_edges = np.linspace(5, 35, 6)

    fig, ax = plt.subplots()

    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    n_p_bins = int((pEnd - pStart) / pStep)

    for i in range(len(theta_edges) - 1):

        theta_lo = theta_edges[i]
        theta_hi = theta_edges[i + 1]

        vals = []
        errs = []

        for j in range(n_p_bins):

            p_lo = pStart + j * pStep
            p_hi = p_lo + pStep

            df_bin = df[
                (df["p"] >= p_lo) &
                (df["p"] < p_hi) &
                (df["theta"] >= theta_lo) &
                (df["theta"] < theta_hi)
            ]

            if len(df_bin) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # =====================================================
            # 1. χ²PID efficiency (target)
            # =====================================================
            chi2_mask = passes_kplus_chi2pid_cut(
                df_bin["chi2pid"].to_numpy(),
                df_bin["p"].to_numpy()
            )
            chi2_eff = chi2_mask.mean()

            # =====================================================
            # 2. BDT scores
            # =====================================================
            scores = df_bin["score"].to_numpy()

            if len(scores) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # =====================================================
            # 3. FIND MATCHED BDT THRESHOLD
            # =====================================================
            thresholds = np.linspace(0.01, 0.99, 100)

            best_t = thresholds[0]
            best_diff = 1e9

            for t in thresholds:
                eff = np.mean(scores > t)
                diff = abs(eff - chi2_eff)

                if diff < best_diff:
                    best_diff = diff
                    best_t = t

            # =====================================================
            # 4. APPLY THAT EXACT BIN-SPECIFIC THRESHOLD
            # =====================================================
            accepted = scores > best_t

            mc = df_bin["mc_matching_pid"].to_numpy()
            valid = ~np.isnan(mc)

            # =====================================================
            # 5. TRUE EFFICIENCY (FIXED PART)
            # =====================================================
            n_true = np.sum(mc == 321)   # true K+
            if n_true == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            n_pass = np.sum(accepted & (mc == 321))

            r = n_pass / n_true
            rErr = np.sqrt(r * (1 - r) / n_true)

            vals.append(r)
            errs.append(rErr)

        # =========================================================
        # plotting
        # =========================================================
        edges = np.linspace(pStart, pEnd, len(vals) + 1)
        x = (edges[:-1] + edges[1:]) / 2

        vals = np.array(vals)
        errs = np.array(errs)
        mask = ~np.isnan(vals)

        ax.errorbar(
            x[mask],
            vals[mask],
            yerr=errs[mask],
            fmt="o",
            capsize=3,
            color=colors[i % len(colors)],
            label=f"{theta_lo:.0f}–{theta_hi:.0f}°"
        )

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("Efficiency")
    ax.set_title("BDT efficiency at chi2pid-matched")

    ax.legend()
    fig.tight_layout()

    outpath = f"{direct}/efficiency_matched_chi2pid_theta.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return fig


def LESSplot_ml_eff_matched_chi2pid_theta(
    df,
    pStart, pEnd, pStep,
    theta_edges=None,
    direct="."
):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    if theta_edges is None:
        theta_edges = np.linspace(5, 35, 6)

    fig, ax = plt.subplots()

    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    n_p_bins = int((pEnd - pStart) / pStep)

    for i in range(len(theta_edges) - 1):

        theta_lo = theta_edges[i]
        theta_hi = theta_edges[i + 1]

        vals = []
        errs = []

        for j in range(n_p_bins):

            p_lo = pStart + j * pStep
            p_hi = p_lo + pStep

            df_bin = df[
                (df["p"] >= p_lo) &
                (df["p"] < p_hi) &
                (df["theta"] >= theta_lo) &
                (df["theta"] < theta_hi)
            ]

            if len(df_bin) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # =====================================================
            # 1. χ²PID efficiency (target)
            # =====================================================
            chi2_mask = np.abs(df_bin["chi2pid"].to_numpy()) < 3
            chi2_eff = chi2_mask.mean()

            # =====================================================
            # 2. BDT scores
            # =====================================================
            scores = df_bin["score"].to_numpy()

            if len(scores) == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            # =====================================================
            # 3. FIND MATCHED BDT THRESHOLD
            # =====================================================
            thresholds = np.linspace(0.01, 0.99, 100)

            best_t = thresholds[0]
            best_diff = 1e9

            for t in thresholds:
                eff = np.mean(scores > t)
                diff = abs(eff - chi2_eff)

                if diff < best_diff:
                    best_diff = diff
                    best_t = t

            # =====================================================
            # 4. APPLY THAT EXACT BIN-SPECIFIC THRESHOLD
            # =====================================================
            accepted = scores > best_t

            mc = df_bin["mc_matching_pid"].to_numpy()
            valid = ~np.isnan(mc)

            # =====================================================
            # 5. TRUE EFFICIENCY (FIXED PART)
            # =====================================================
            n_true = np.sum(mc == 321)   # true K+
            if n_true == 0:
                vals.append(np.nan)
                errs.append(np.nan)
                continue

            n_pass = np.sum(accepted & (mc == 321))

            r = n_pass / n_true
            rErr = np.sqrt(r * (1 - r) / n_true)

            vals.append(r)
            errs.append(rErr)

        # =========================================================
        # plotting
        # =========================================================
        edges = np.linspace(pStart, pEnd, len(vals) + 1)
        x = (edges[:-1] + edges[1:]) / 2

        vals = np.array(vals)
        errs = np.array(errs)
        mask = ~np.isnan(vals)

        ax.errorbar(
            x[mask],
            vals[mask],
            yerr=errs[mask],
            fmt="o",
            capsize=3,
            color=colors[i % len(colors)],
            label=f"{theta_lo:.0f}–{theta_hi:.0f}°"
        )

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("Efficiency")
    ax.set_title("BDT efficiency at |chi2pid|<3-matched")

    ax.legend()
    fig.tight_layout()

    outpath = f"{direct}/LESSefficiency_matched_chi2pid_theta.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return fig




######################chi2pid<3


##############################################
#chipid<3





#############################################



def plot_contamination_chi2pid_vs_bdt3(
    df,
    pStart,
    pEnd,
    pStep,
    direct="."
):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    theCut = df[(df["theta"] >= 17) & (df["theta"] <= 23)]
    df = theCut

    fig, ax = plt.subplots()

    ml_vals = []
    ml_errs = []

    chi2_vals = []
    chi2_errs = []

    chi2hard_vals = []
    chi2hard_errs = []

    n_p_bins = int((pEnd - pStart) / pStep)

    for j in range(n_p_bins):

        p_lo = pStart + j * pStep
        p_hi = p_lo + pStep

        df_bin = df[
            (df["p"] >= p_lo) &
            (df["p"] < p_hi)
        ]

        if len(df_bin) == 0:
            ml_vals.append(np.nan)
            ml_errs.append(np.nan)
            chi2_vals.append(np.nan)
            chi2_errs.append(np.nan)
            chi2hard_vals.append(np.nan)
            chi2hard_errs.append(np.nan)
            continue

        mc = df_bin["mc_matching_pid"].to_numpy()
        pid = df_bin["pid"].to_numpy()

        n_true_k = np.sum(mc == 321)

        if n_true_k == 0:
            ml_vals.append(np.nan)
            ml_errs.append(np.nan)
            chi2_vals.append(np.nan)
            chi2_errs.append(np.nan)
            chi2hard_vals.append(np.nan)
            chi2hard_errs.append(np.nan)
            continue

        # =====================================================
        # χ²PID (baseline)
        # =====================================================
        chi2_mask = passes_kplus_chi2pid_cut(
            df_bin["chi2pid"].to_numpy(),
            df_bin["p"].to_numpy()
        )

        chi2_eff = np.sum(chi2_mask & (mc == 321)) / n_true_k

        # =====================================================
        # HARD χ²PID < 3 cut (NEW THIRD CURVE)
        # =====================================================
        chi2_hard_mask = abs(df_bin["chi2pid"].to_numpy()) < 3
        chi2hard_eff = np.sum(chi2_hard_mask & (mc == 321)) / n_true_k

        # =====================================================
        # BDT threshold matching χ²PID efficiency
        # =====================================================
        scores = df_bin["score"].to_numpy()
        thresholds = np.linspace(0.01, 0.99, 100)

        best_t = thresholds[0]
        best_diff = 1e9

        for t in thresholds:

            accepted = scores > t

            eff = np.sum(accepted & (mc == 321)) / n_true_k

            diff = abs(eff - chi2_eff)

            if diff < best_diff:
                best_diff = diff
                best_t = t

        # =====================================================
        # ML contamination (matched efficiency)
        # =====================================================
        accepted_ml = scores > best_t

        n_acc_ml = np.sum(accepted_ml & (pid == 321))

        if n_acc_ml > 0:
            n_bad_ml = np.sum(
                accepted_ml &
                (pid == 321) &
                (mc == 211)
            )
            r_ml = n_bad_ml / n_acc_ml
            rErr_ml = np.sqrt(r_ml * (1 - r_ml) / n_acc_ml)
        else:
            r_ml = np.nan
            rErr_ml = np.nan

        ml_vals.append(r_ml)
        ml_errs.append(rErr_ml)

        # =====================================================
        # χ²PID contamination
        # =====================================================
        n_acc_chi2 = np.sum(chi2_mask & (pid == 321))

        if n_acc_chi2 > 0:
            n_bad_chi2 = np.sum(
                chi2_mask &
                (pid == 321) &
                (mc == 211)
            )
            r_chi2 = n_bad_chi2 / n_acc_chi2
            rErr_chi2 = np.sqrt(r_chi2 * (1 - r_chi2) / n_acc_chi2)
        else:
            r_chi2 = np.nan
            rErr_chi2 = np.nan

        chi2_vals.append(r_chi2)
        chi2_errs.append(rErr_chi2)

        # =====================================================
        # HARD χ²PID < 3 contamination (NEW)
        # =====================================================
        n_acc_hard = np.sum(chi2_hard_mask & (pid == 321))

        if n_acc_hard > 0:
            n_bad_hard = np.sum(
                chi2_hard_mask &
                (pid == 321) &
                (mc == 211)
            )
            r_hard = n_bad_hard / n_acc_hard
            rErr_hard = np.sqrt(r_hard * (1 - r_hard) / n_acc_hard)
        else:
            r_hard = np.nan
            rErr_hard = np.nan

        chi2hard_vals.append(r_hard)
        chi2hard_errs.append(rErr_hard)

    # =========================================================
    # Plot
    # =========================================================
    edges = np.linspace(pStart, pEnd, len(ml_vals) + 1)
    x = (edges[:-1] + edges[1:]) / 2

    ml_vals = np.array(ml_vals)
    ml_errs = np.array(ml_errs)

    chi2_vals = np.array(chi2_vals)
    chi2_errs = np.array(chi2_errs)

    chi2hard_vals = np.array(chi2hard_vals)
    chi2hard_errs = np.array(chi2hard_errs)

    mask_ml = ~np.isnan(ml_vals)
    mask_chi2 = ~np.isnan(chi2_vals)
    mask_hard = ~np.isnan(chi2hard_vals)

    ax.errorbar(
        x[mask_ml],
        ml_vals[mask_ml],
        yerr=ml_errs[mask_ml],
        fmt="o",
        capsize=3,
        color="tab:blue",
        label="BDT (matched efficiency)"
    )

    ax.errorbar(
        x[mask_chi2],
        chi2_vals[mask_chi2],
        yerr=chi2_errs[mask_chi2],
        fmt="o",
        capsize=3,
        color="black",
        label="Baseline"
    )

    ax.errorbar(
        x[mask_hard],
        chi2hard_vals[mask_hard],
        yerr=chi2hard_errs[mask_hard],
        fmt="o",
        capsize=3,
        color="tab:red",
        label="|chi2pid| < 3 cut"
    )

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("Contamination")
    ax.set_title(
        "Contamination Baseline Comparisons with BDT "
        "(17 < theta < 23)"
    )

    ax.legend()
    fig.tight_layout()

    outpath = f"{direct}/three_contamination_chi2pid_vs_bdt_highBin.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return fig


###### Efficiency #########






###### Mis-ID ##########



def plot_efficiency_vs_contamination_theta(sweep_csv, outdir):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df = pd.read_csv(sweep_csv)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [
        "black",
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red"
    ]

    # unique theta bins
    theta_bins = (
        df[["theta_lo", "theta_hi"]]
        .drop_duplicates()
        .sort_values("theta_lo")
        .to_numpy()
    )

    for i, (theta_lo, theta_hi) in enumerate(theta_bins):

        theta_df = df[
            (df["theta_lo"] == theta_lo) &
            (df["theta_hi"] == theta_hi)
        ]

        # average over all momentum bins at each threshold
        grouped = (
            theta_df
            .groupby("threshold")
            .agg({
                "eff_K": "mean",
                "C_pi": "mean"
            })
            .reset_index()
            .sort_values("eff_K")
        )

        ax.plot(
            grouped["eff_K"],
            grouped["C_pi"],
            lw=2,
            color=colors[i % len(colors)],
            label=f"{theta_lo:.0f}–{theta_hi:.0f}°"
        )

    ax.set_xlabel("K⁺ Efficiency")
    ax.set_ylabel("π → K Contamination")
    ax.set_title("Efficiency vs Contamination by θ Bin")

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    outfile = f"{outdir}/efficiency_vs_contamination_theta.png"
    fig.savefig(outfile, dpi=150)

    plt.close(fig)
    allPlots.append(fig)
    print(f"Saved: {outfile}")
# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


#-----------------------------------------
#Helper functions
#-----------------------------------------

def computeEffChiPID(df, pStart, pEnd, pBinN):
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------------------------
    # keep only valid MC-matched tracks
    # -------------------------------------------------
    df = df[df["mc_matching_pid"] != -9999].copy()

    step = (pEnd - pStart) / pBinN
    eff = []

    for i in range(pBinN):

        p_lo = pStart + i * step
        p_hi = pStart + (i + 1) * step

        binp = df[
            (df["p"] >= p_lo) &
            (df["p"] < p_hi)
        ]

        if len(binp) == 0:
            eff.append(np.nan)
            continue

        # -------------------------------------------------
        # TRUE K+ (DENOMINATOR ONLY)
        # -------------------------------------------------
        true_k = (binp["mc_matching_pid"] == 321)
        n_true_k = np.sum(true_k)

        if n_true_k == 0:
            eff.append(np.nan)
            continue

        # -------------------------------------------------
        # χ²PID CUT
        # -------------------------------------------------
        chi2_mask = passes_kplus_chi2pid_cut(
            binp["chi2pid"].to_numpy(),
            binp["p"].to_numpy()
        )

        # -------------------------------------------------
        # NUMERATOR: true K+ that PASS cut AND are reconstructed as K+
        # -------------------------------------------------
        reconstructed_k = (binp["pid"] == 321)

        passed = chi2_mask & reconstructed_k & true_k.to_numpy()

        eff.append(np.sum(passed) / n_true_k)

    # -------------------------------------------------
    # plotting
    # -------------------------------------------------
    eff = np.array(eff, dtype=float)

    p_centers = np.linspace(
        pStart + 0.5*(pEnd - pStart)/pBinN,
        pEnd - 0.5*(pEnd - pStart)/pBinN,
        pBinN
    )

    mask = np.isfinite(eff) & np.isfinite(p_centers)

    plt.figure()

    if np.sum(mask) == 0:
        print("TEST plot skipped: no valid data")
        return eff

    plt.plot(p_centers[mask], eff[mask], "o-", label="χ²PID efficiency")

    plt.xlabel("Momentum (GeV/c)")
    plt.ylabel("Efficiency")
    plt.ylim(0, 1.05)
    plt.title("TEST: χ²PID Efficiency vs Momentum")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        "/volatile/clas12/cooperb/SULI/tier2All/eval_v01/TEST_chi2pid_efficiency.png",
        dpi=150
    )
    plt.close()

    return eff



def MatchEfficiency(df, pStart, pEnd, pBinN):
    import numpy as np

    df = df[df["mc_matching_pid"] != -9999].copy()

    # correct χ²PID efficiencies
    effs = computeEffChiPID(df, pStart, pEnd, pBinN)

    step = (pEnd - pStart) / pBinN
    thresholds_out = []

    for i in range(pBinN):

        if np.isnan(effs[i]):
            thresholds_out.append(np.nan)
            continue

        p_lo = pStart + i * step
        p_hi = pStart + (i + 1) * step

        binp = df[
            (df["p"] >= p_lo) &
            (df["p"] < p_hi)
        ]

        if len(binp) == 0:
            thresholds_out.append(np.nan)
            continue

        true_k = binp["mc_matching_pid"] == 321
        n_true_k = np.sum(true_k)

        if n_true_k == 0:
            thresholds_out.append(np.nan)
            continue

        scores = binp["score"].to_numpy()
        target_eff = effs[i]

        thresholds = np.linspace(0.01, 0.99, 200)

        best_t = np.nan
        best_diff = 1e9

        for t in thresholds:

            accepted = scores > t
            eff = np.sum(accepted & true_k.to_numpy()) / n_true_k

            diff = abs(eff - target_eff)

            if diff < best_diff:
                best_diff = diff
                best_t = t

        thresholds_out.append(best_t)

    return np.array(thresholds_out)

def ComputeMLMATCHeff(
    df,
    pStart,
    pEnd,
    pBinN,
    score_col="score"
):
    import numpy as np

    df = df[df["mc_matching_pid"] != -9999].copy()

    thresholds = MatchEfficiency(df, pStart, pEnd, pBinN)

    step = (pEnd - pStart) / pBinN

    effs = []
    errors = []

    for i in range(pBinN):

        p_lo = pStart + i * step
        p_hi = pStart + (i + 1) * step

        binp = df[
            (df["p"] >= p_lo) &
            (df["p"] < p_hi)
        ]

        if len(binp) == 0:
            effs.append(np.nan)
            errors.append(np.nan)
            continue

        scores = binp[score_col].to_numpy()
        accepted = scores > thresholds[i]

        true_k = binp["mc_matching_pid"] == 321

        n_true_k = np.sum(true_k)

        if n_true_k == 0:
            effs.append(np.nan)
            errors.append(np.nan)
            continue

        n_true_k_acc = np.sum(true_k & accepted)

        eff = n_true_k_acc / n_true_k

        err = np.sqrt(eff * (1 - eff) / n_true_k)

        effs.append(eff)
        errors.append(err)

    return np.array(effs), np.array(errors)

def ComputeMLMATCHcontams(
    df,
    pStart,
    pEnd,
    pBinN,
    score_col="score"
):
    import numpy as np

    df = df[df["mc_matching_pid"] != -9999].copy()

    thresholds = MatchEfficiency(df, pStart, pEnd, pBinN)

    step = (pEnd - pStart) / pBinN

    contams = []
    errors = []

    for i in range(pBinN):

        p_lo = pStart + i * step
        p_hi = pStart + (i + 1) * step

        binp = df[
            (df["p"] >= p_lo) &
            (df["p"] < p_hi)
        ]

        if len(binp) == 0:
            contams.append(np.nan)
            errors.append(np.nan)
            continue

        scores = binp[score_col].to_numpy()
        accepted = scores > thresholds[i]

        n_acc = np.sum(accepted)

        if n_acc == 0:
            contams.append(np.nan)
            errors.append(np.nan)
            continue

        true_k = binp["mc_matching_pid"] == 321
        fake_k = ~true_k

        n_fake_acc = np.sum(fake_k & accepted)

        C = n_fake_acc / n_acc

        # -------------------------------------------------
        # binomial error
        # -------------------------------------------------
        err = np.sqrt(C * (1 - C) / n_acc)

        contams.append(C)
        errors.append(err)

    return np.array(contams), np.array(errors)

def ComputeChi2PIDcontams(
    df,
    pStart,
    pEnd,
    pBinN
):
    import numpy as np

    df = df[df["mc_matching_pid"] != -9999].copy()
    df2=df

    step = (pEnd - pStart) / pBinN

    contams = []
    errors = []

    for i in range(pBinN):

        p_lo = pStart + i * step
        p_hi = pStart + (i + 1) * step

        binp = df[
            (df["p"] >= p_lo) &
            (df["p"] < p_hi)
        ]

        if len(binp) == 0:
            contams.append(np.nan)
            errors.append(np.nan)
            continue

        # -------------------------------------------------
        # χ²PID CUT (baseline selection)
        # -------------------------------------------------
        chi2_mask = passes_kplus_chi2pid_cut(
            binp["chi2pid"].to_numpy(),
            binp["p"].to_numpy()
        )

        accepted = chi2_mask  # THIS is your selection

        n_acc = np.sum(accepted)

        if n_acc == 0:
            contams.append(np.nan)
            errors.append(np.nan)
            continue

        # -------------------------------------------------
        # truth definition
        # -------------------------------------------------
        true_k = binp["mc_matching_pid"] == 321
        fake_k = ~true_k

        n_fake_acc = np.sum(fake_k & accepted)

        C = n_fake_acc / n_acc

        # -------------------------------------------------
        # binomial error
        # -------------------------------------------------
        err = np.sqrt(C * (1 - C) / n_acc)

        contams.append(C)
        errors.append(err)

    return np.array(contams), np.array(errors)

def PlotMLMATCHcontams_theta(
    df,
    pStart,
    pEnd,
    pBinN,
    theta_edges,
    outdir
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots()

    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    step = (pEnd - pStart) / pBinN

    # shared p-bin centers
    p_centers = np.array([
        pStart + (i + 0.5) * step for i in range(pBinN)
    ])

    for i in range(len(theta_edges) - 1):

        th_lo = theta_edges[i]
        th_hi = theta_edges[i + 1]

        df_theta = df[
            (df["theta"] >= th_lo) &
            (df["theta"] < th_hi)
        ]

        if len(df_theta) == 0:
            continue

        # -------------------------------------------------
        # full pipeline happens inside this call
        # -------------------------------------------------
        contams, errs = ComputeMLMATCHcontams(
            df_theta,
            pStart,
            pEnd,
            pBinN
        )

        contams = np.array(contams)
        errs = np.array(errs)

        mask = ~np.isnan(contams)

        ax.errorbar(
            p_centers[mask],
            contams[mask],
            yerr=errs[mask],
            fmt="o",
            capsize=3,
            color=colors[i % len(colors)],
            label=f"{th_lo:.0f}–{th_hi:.0f}°"
        )

    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("ML Contamination")
    ax.set_title("BDT Contamination vs Momentum (θ-binned)")
    ax.set_ylim(0, 1.1)

    ax.legend()
    fig.tight_layout()

    outpath = os.path.join(outdir, "ml_contamination_chi2PID_Matched.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return fig

def PlotMLMATCHeffs_theta(
    df,
    pStart,
    pEnd,
    pBinN,
    theta_edges,
    outdir
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots()

    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    step = (pEnd - pStart) / pBinN

    # shared p-bin centers
    p_centers = np.array([
        pStart + (i + 0.5) * step for i in range(pBinN)
    ])

    for i in range(len(theta_edges) - 1):

        th_lo = theta_edges[i]
        th_hi = theta_edges[i + 1]

        df_theta = df[
            (df["theta"] >= th_lo) &
            (df["theta"] < th_hi)
        ]

        if len(df_theta) == 0:
            continue

        # -------------------------------------------------
        # full pipeline happens inside this call
        # -------------------------------------------------
        contams, errs = ComputeMLMATCHeff(
            df_theta,
            pStart,
            pEnd,
            pBinN
        )

        contams = np.array(contams)
        errs = np.array(errs)

        mask = ~np.isnan(contams)

        ax.errorbar(
            p_centers[mask],
            contams[mask],
            yerr=errs[mask],
            fmt="o",
            capsize=3,
            color=colors[i % len(colors)],
            label=f"{th_lo:.0f}–{th_hi:.0f}°"
        )

    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("ML Efficiciency")
    ax.set_title("BDT Efficiency vs Momentum (θ-binned)")
    ax.set_ylim(0, 1.1)

    ax.legend()
    fig.tight_layout()

    outpath = os.path.join(outdir, "ml_efficiency_chi2PID_Matched.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return fig
    
def PlotMLContamCompare(
    df,
    pStart,
    pEnd,
    pBinN,
    outdir,
    name,
    tStart,
    tEnd,
    label
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs(outdir, exist_ok=True)
    df=df[(df["theta"]>=tStart)&(df["theta"]<=tEnd)]
    step = (pEnd - pStart) / pBinN
    p_centers = np.array([
        pStart + (i + 0.5) * step for i in range(pBinN)
    ])

    # -------------------------------------------------
    # compute both contaminations
    # -------------------------------------------------
    chi2_contam, chi2_err = ComputeChi2PIDcontams(
        df, pStart, pEnd, pBinN
    )

    ml_contam, ml_err = ComputeMLMATCHcontams(
        df, pStart, pEnd, pBinN
    )

    chi2_contam = np.array(chi2_contam, dtype=float)
    ml_contam = np.array(ml_contam, dtype=float)

    chi2_err = np.array(chi2_err, dtype=float)
    ml_err = np.array(ml_err, dtype=float)

    # masks (avoid NaNs crashing matplotlib)
    mask_chi2 = np.isfinite(chi2_contam)
    mask_ml = np.isfinite(ml_contam)

    # -------------------------------------------------
    # plot
    # -------------------------------------------------
    plt.figure()

    plt.errorbar(
        p_centers[mask_chi2],
        chi2_contam[mask_chi2],
        yerr=chi2_err[mask_chi2],
        fmt="o",
        label="Chi2pid",
        alpha=0.8
    )

    plt.errorbar(
        p_centers[mask_ml],
        ml_contam[mask_ml],
        yerr=ml_err[mask_ml],
        fmt="o",
        label="ML (BDT) efficiency Matched",
        alpha=0.8
    )

    plt.xlabel("Momentum (GeV/c)")
    plt.ylabel("Contamination")
    plt.title("ML vs Chi2PID Contamination Comparison "+label)
    plt.ylim(0, 0.51)
    plt.legend()
    plt.grid(False)
    outpath = os.path.join(outdir, name+"ml_compare_chi2pid_contam.png")
    plt.savefig(outpath, dpi=150)
    plt.close()

    return None



########################## |chi2pid|<3 cut versions of pipeline

def LESScomputeEffChiPID(df, pStart, pEnd, pBinN):
    import numpy as np
    import matplotlib.pyplot as plt

    df = df[df["mc_matching_pid"] != -9999].copy()

    step = (pEnd - pStart) / pBinN
    eff = []

    for i in range(pBinN):

        p_lo = pStart + i * step
        p_hi = pStart + (i + 1) * step

        binp = df[
            (df["p"] >= p_lo) &
            (df["p"] < p_hi)
        ]

        if len(binp) == 0:
            eff.append(np.nan)
            continue

        # -------------------------------------------------
        # TRUE K+ (DENOMINATOR ONLY)
        # -------------------------------------------------
        true_k = (binp["mc_matching_pid"] == 321)
        n_true_k = np.sum(true_k)

        if n_true_k == 0:
            eff.append(np.nan)
            continue

        # -------------------------------------------------
        # χ²PID CUT: |chi2| < 3
        # -------------------------------------------------
        chi2 = binp["chi2pid"].to_numpy()
        chi2_mask = np.abs(chi2) < 3

        # -------------------------------------------------
        # NUMERATOR: true K+ passing cut
        # -------------------------------------------------
        passed = chi2_mask & true_k.to_numpy()

        eff.append(np.sum(passed) / n_true_k)

    eff = np.array(eff, dtype=float)

    # -------------------------------------------------
    # p-centers
    # -------------------------------------------------
    p_centers = np.linspace(
        pStart + 0.5*(pEnd - pStart)/pBinN,
        pEnd - 0.5*(pEnd - pStart)/pBinN,
        pBinN
    )

    mask = np.isfinite(eff) & np.isfinite(p_centers)

    plt.figure()

    if np.sum(mask) == 0:
        print("TEST plot skipped: no valid data")
        return eff

    plt.plot(
        p_centers[mask],
        eff[mask],
        "o-",
        label="|χ²PID| < 3 efficiency"
    )

    plt.xlabel("Momentum (GeV/c)")
    plt.ylabel("Efficiency")
    plt.ylim(0, 1.05)
    plt.title("TEST: χ²PID Efficiency vs Momentum")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        "/volatile/clas12/cooperb/SULI/tier2All/eval_v01/TEST_LESSchi2pid_efficiency.png",
        dpi=150
    )
    plt.close()

    return eff

def LESSMatchEfficiency(df, pStart, pEnd, pBinN):
    import numpy as np

    df = df[df["mc_matching_pid"] != -9999].copy()

    # correct χ²PID efficiencies
    effs = LESScomputeEffChiPID(df, pStart, pEnd, pBinN)

    step = (pEnd - pStart) / pBinN
    thresholds_out = []

    for i in range(pBinN):

        if np.isnan(effs[i]):
            thresholds_out.append(np.nan)
            continue

        p_lo = pStart + i * step
        p_hi = pStart + (i + 1) * step

        binp = df[
            (df["p"] >= p_lo) &
            (df["p"] < p_hi)
        ]

        if len(binp) == 0:
            thresholds_out.append(np.nan)
            continue

        true_k = binp["mc_matching_pid"] == 321
        n_true_k = np.sum(true_k)

        if n_true_k == 0:
            thresholds_out.append(np.nan)
            continue

        scores = binp["score"].to_numpy()
        target_eff = effs[i]

        thresholds = np.linspace(0.01, 0.99, 200)

        best_t = np.nan
        best_diff = 1e9

        for t in thresholds:

            accepted = scores > t
            eff = np.sum(accepted & true_k.to_numpy()) / n_true_k

            diff = abs(eff - target_eff)

            if diff < best_diff:
                best_diff = diff
                best_t = t

        thresholds_out.append(best_t)

    return np.array(thresholds_out)

def LESSComputeMLMATCHeff(
    df,
    pStart,
    pEnd,
    pBinN,
    score_col="score"
):
    import numpy as np

    df = df[df["mc_matching_pid"] != -9999].copy()

    thresholds = LESSMatchEfficiency(df, pStart, pEnd, pBinN)

    step = (pEnd - pStart) / pBinN

    effs = []
    errors = []

    for i in range(pBinN):

        p_lo = pStart + i * step
        p_hi = pStart + (i + 1) * step

        binp = df[
            (df["p"] >= p_lo) &
            (df["p"] < p_hi)
        ]

        if len(binp) == 0:
            effs.append(np.nan)
            errors.append(np.nan)
            continue

        scores = binp[score_col].to_numpy()
        accepted = scores > thresholds[i]

        true_k = binp["mc_matching_pid"] == 321

        n_true_k = np.sum(true_k)

        if n_true_k == 0:
            effs.append(np.nan)
            errors.append(np.nan)
            continue

        n_true_k_acc = np.sum(true_k & accepted)

        eff = n_true_k_acc / n_true_k

        err = np.sqrt(eff * (1 - eff) / n_true_k)

        effs.append(eff)
        errors.append(err)

    return np.array(effs), np.array(errors)

def LESSComputeMLMATCHcontams(
    df,
    pStart,
    pEnd,
    pBinN,
    score_col="score"
):
    import numpy as np

    df = df[df["mc_matching_pid"] != -9999].copy()

    thresholds = LESSMatchEfficiency(df, pStart, pEnd, pBinN)

    step = (pEnd - pStart) / pBinN

    contams = []
    errors = []

    for i in range(pBinN):

        p_lo = pStart + i * step
        p_hi = pStart + (i + 1) * step

        binp = df[
            (df["p"] >= p_lo) &
            (df["p"] < p_hi)
        ]

        if len(binp) == 0:
            contams.append(np.nan)
            errors.append(np.nan)
            continue

        scores = binp[score_col].to_numpy()
        accepted = scores > thresholds[i]

        n_acc = np.sum(accepted)

        if n_acc == 0:
            contams.append(np.nan)
            errors.append(np.nan)
            continue

        true_k = binp["mc_matching_pid"] == 321
        fake_k = ~true_k

        n_fake_acc = np.sum(fake_k & accepted)

        C = n_fake_acc / n_acc

        # -------------------------------------------------
        # binomial error
        # -------------------------------------------------
        err = np.sqrt(C * (1 - C) / n_acc)

        contams.append(C)
        errors.append(err)

    return np.array(contams), np.array(errors)

def LESSComputeChi2PIDcontams(df, pStart, pEnd, pBinN):
    import numpy as np

    df = df.copy()

    step = (pEnd - pStart) / pBinN

    contams = []
    errors = []

    for i in range(pBinN):

        p_lo = pStart + i * step
        p_hi = pStart + (i + 1) * step

        # -------------------------------------------------
        # bin selection
        # -------------------------------------------------
        binp = df[(df["p"] >= p_lo) & (df["p"] < p_hi)]

        if len(binp) == 0:
            contams.append(np.nan)
            errors.append(np.nan)
            continue

        # -------------------------------------------------
        # χ²PID selection (THIS defines the sample)
        # -------------------------------------------------
        chi2_mask = passes_kplus_chi2pid_cut(
            binp["chi2pid"].to_numpy(),
            binp["p"].to_numpy()
        ).astype(bool)

        n_sel = np.sum(chi2_mask)

        if n_sel == 0:
            contams.append(np.nan)
            errors.append(np.nan)
            continue

        # -------------------------------------------------
        # truth split (ONLY used AFTER selection)
        # -------------------------------------------------
        mc = binp["mc_matching_pid"].to_numpy()

        true_k = (mc == 321)
        fake_k = (mc != 321)

        # -------------------------------------------------
        # contamination definition
        # -------------------------------------------------
        n_fake_sel = np.sum(fake_k & chi2_mask)

        C = n_fake_sel / n_sel
        err = np.sqrt(C * (1 - C) / n_sel)

        contams.append(C)
        errors.append(err)

    return np.array(contams), np.array(errors)

def LESSPlotMLMATCHcontams_theta(
    df,
    pStart,
    pEnd,
    pBinN,
    theta_edges,
    outdir
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots()

    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    step = (pEnd - pStart) / pBinN

    # shared p-bin centers
    p_centers = np.array([
        pStart + (i + 0.5) * step for i in range(pBinN)
    ])

    for i in range(len(theta_edges) - 1):

        th_lo = theta_edges[i]
        th_hi = theta_edges[i + 1]

        df_theta = df[
            (df["theta"] >= th_lo) &
            (df["theta"] < th_hi)
        ]

        if len(df_theta) == 0:
            continue

        # -------------------------------------------------
        # full pipeline happens inside this call
        # -------------------------------------------------
        contams, errs = LESSComputeMLMATCHcontams(
            df_theta,
            pStart,
            pEnd,
            pBinN
        )

        contams = np.array(contams)
        errs = np.array(errs)

        mask = ~np.isnan(contams)

        ax.errorbar(
            p_centers[mask],
            contams[mask],
            yerr=errs[mask],
            fmt="o",
            capsize=3,
            color=colors[i % len(colors)],
            label=f"{th_lo:.0f}–{th_hi:.0f}°"
        )

    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("ML Contamination")
    ax.set_title("BDT Contamination vs Momentum (θ-binned)")
    ax.set_ylim(0, 1.1)

    ax.legend()
    fig.tight_layout()

    outpath = os.path.join(outdir, "LESSml_contamination_chi2PID_Matched.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return fig

def LESSPlotMLMATCHeffs_theta(
    df,
    pStart,
    pEnd,
    pBinN,
    theta_edges,
    outdir
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots()

    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    step = (pEnd - pStart) / pBinN

    # shared p-bin centers
    p_centers = np.array([
        pStart + (i + 0.5) * step for i in range(pBinN)
    ])

    for i in range(len(theta_edges) - 1):

        th_lo = theta_edges[i]
        th_hi = theta_edges[i + 1]

        df_theta = df[
            (df["theta"] >= th_lo) &
            (df["theta"] < th_hi)
        ]

        if len(df_theta) == 0:
            continue

        # -------------------------------------------------
        # full pipeline happens inside this call
        # -------------------------------------------------
        contams, errs = LESSComputeMLMATCHeff(
            df_theta,
            pStart,
            pEnd,
            pBinN
        )

        contams = np.array(contams)
        errs = np.array(errs)

        mask = ~np.isnan(contams)

        ax.errorbar(
            p_centers[mask],
            contams[mask],
            yerr=errs[mask],
            fmt="o",
            capsize=3,
            color=colors[i % len(colors)],
            label=f"{th_lo:.0f}–{th_hi:.0f}°"
        )

    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("ML Efficiciency")
    ax.set_title("BDT Efficiency vs Momentum (θ-binned)")
    ax.set_ylim(0, 1.1)

    ax.legend()
    fig.tight_layout()

    outpath = os.path.join(outdir, "LESSml_efficiency_chi2PID_Matched.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return fig
    
def LESSPlotMLContamCompare(
    df,
    pStart,
    pEnd,
    pBinN,
    outdir,
    name,
    tStart,
    tEnd,
    label
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    df=df[(df["theta"]>=tStart)&(df["theta"]<=tEnd)]
    os.makedirs(outdir, exist_ok=True)

    step = (pEnd - pStart) / pBinN
    p_centers = np.array([
        pStart + (i + 0.5) * step for i in range(pBinN)
    ])

    # -------------------------------------------------
    # compute both contaminations
    # -------------------------------------------------
    chi2_contam, chi2_err = LESSComputeChi2PIDcontams(
        df, pStart, pEnd, pBinN
    )

    ml_contam, ml_err = LESSComputeMLMATCHcontams(
        df, pStart, pEnd, pBinN
    )

    chi2_contam = np.array(chi2_contam, dtype=float)
    ml_contam = np.array(ml_contam, dtype=float)

    chi2_err = np.array(chi2_err, dtype=float)
    ml_err = np.array(ml_err, dtype=float)

    # masks (avoid NaNs crashing matplotlib)
    mask_chi2 = np.isfinite(chi2_contam)
    mask_ml = np.isfinite(ml_contam)

    # -------------------------------------------------
    # plot
    # -------------------------------------------------
    plt.figure()

    plt.errorbar(
        p_centers[mask_chi2],
        chi2_contam[mask_chi2],
        yerr=chi2_err[mask_chi2],
        fmt="o",
        label="χ²PID",
        alpha=0.8
    )

    plt.errorbar(
        p_centers[mask_ml],
        ml_contam[mask_ml],
        yerr=ml_err[mask_ml],
        fmt="o",
        label="ML (BDT) efficiency matched",
        alpha=0.8
    )

    plt.xlabel("Momentum (GeV/c)")
    plt.ylabel("Contamination")
    plt.title("ML vs chi2PID Contamination Comparison "+label)
    plt.ylim(0, 1.1)
    plt.legend()

    outpath = os.path.join(outdir, name+"LESSml_compare_chi2pid_contam.png")
    plt.savefig(outpath, dpi=150)
    plt.close()

    return None






#############################

    
    
def evaluate_model(
    model_path: pathlib.Path,
    test_path: pathlib.Path,
    p_edges: list,
    theta_edges: list,
    outdir: pathlib.Path,
    threshold_grid=None,
    overwrite: bool = False,
) -> pd.DataFrame:

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sweep_csv_path = outdir / "per_bin_sweep.csv"
    if sweep_csv_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {sweep_csv_path}. Pass overwrite=True."
        )

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    _model_raw = joblib.load(str(model_path))

    # Support both the new wrapper dict {"model": ..., "features": [...]} and
    # old bare-estimator joblib files (backward compat).
    if isinstance(_model_raw, dict) and "model" in _model_raw and "features" in _model_raw:
        model = _model_raw["model"]
        feature_names = _model_raw["features"]
        print(f"  Features (from model.joblib wrapper): {len(feature_names)}")
    else:
        # Old bare-estimator model — fall back to manifest.
        model = _model_raw
        import warnings
        warnings.warn(
            "model.joblib does not contain a feature list (old format — bare estimator). "
            "Falling back to manifest.json feature_list. "
            "Rebuild the model with the new train_bdt.py to embed features in model.joblib.",
            DeprecationWarning,
            stacklevel=2,
        )
        manifest_path = pathlib.Path(test_path).resolve().parents[0] / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"model.joblib has no embedded feature list and manifest.json not found at "
                f"{manifest_path}. Cannot determine feature names."
            )
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        feature_names = manifest.get("feature_list") or manifest.get("columns")
        if not feature_names:
            raise ValueError(
                f"manifest.json at {manifest_path} has no feature_list or columns field. "
                f"Rebuild the dataset and model."
            )
        print(f"  Features (from manifest fallback — DEPRECATED): {len(feature_names)}")

    # ── Load test set ──────────────────────────────────────────────────────────
    print(f"Loading test set: {test_path}")
    df = pd.read_parquet(str(test_path))
    print(f"  Test rows: {len(df):,}")

    # ── Build feature matrix ───────────────────────────────────────────────────
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

# matched-efficiency BDT threshold
    bdt_thresholds      = np.full((n_p, n_t), np.nan)

    mask_lowstat        = np.ones((n_p, n_t), dtype=bool)

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
                bdt_thresholds[pi, ti] = t_matched_eff
                #bdt_eff_matched[pi, ti] = bl["baseline_eff_K"]

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
    plot_efficiency_vs_contamination_theta(
        sweep_csv_path,
        outdir
    )

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
        thresholds=bdt_thresholds,
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

    plot_bdt_1d(df, outdir / "bdt_score_1d_full.png")
    plot_ml_contamination_simple(
        matched=df,
        pStart=0.5,
        pEnd=3.2,
        pStep=0.27,
        #bins=10,
        direct=str(outdir) + "/",
        bdt_cut=t_matched_eff
    )
    plot_ml_contamination_fixed_efficiency_theta(
        matched=df,
        pStart=0.5,
        pEnd=3.2,
        pStep=0.27,
        target_eff=0.8,
        direct=str(outdir)
    )
  
    #plot_ml_contamination_matched_chi2pidLess_theta(
     #   df,
      ## direct=str(outdir)
    #)
    #plot_ml_eff_matched_chi2pid_theta(
      ## 0.5, 3.2, 0.27,
        #direct=str(outdir)
    #)
    plot_ml_efficiency_fixed_efficiency_theta(
        matched=df,
        pStart=0.5,
        pEnd=3.2,
        pStep=0.27,
        target_eff=0.8,
        direct=str(outdir)
    )


    

    
     #LESSplot_ml_eff_matched_chi2pid_theta(
      #  df,
      #  0.5, 3.2, 0.27,
      #  direct=str(outdir)
    #)

    #plot_contamination_chi2pid_vs_bdt(
     #   df=df,
      #  pStart=0.5,
       # pEnd=3.2,
        #pStep=0.27,
       # direct=str(outdir)
    #)
    #LESSplot_contamination_chi2pid_vs_bdt(
     #   df=df,
      #  pStart=0.5,
       # pEnd=3.2,
        #pStep=0.27,
        #direct=str(outdir)
    #)

    #plot_contamination_chi2pid_vs_bdt3(
     #   df=df,
      #  pStart=0.5,
       # pEnd=3.2,
        #pStep=0.27,
        #direct=str(outdir)
    #)

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

def plot_bdt_1d(df, outpath):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    score = df["score"].to_numpy()

    label = df["label"]
    mcp = df["mc_matching_pid"].to_numpy()

    label_notna = pd.notna(label)
    label_vals = np.where(label_notna, label.astype(float), np.nan)

    scores_K = score[label_vals == 1]
    scores_pi = score[label_vals == 0]
    scores_p = score[mcp == 2212]

    bins = np.linspace(0, 1, 80)

    plt.figure(figsize=(8, 5))

    plt.hist(scores_K, bins=bins, density=True, alpha=0.6, label="K⁺ (signal)", color="green")
    plt.hist(scores_pi, bins=bins, density=True, alpha=0.6, label="π⁺ (background)", color="blue")
    #plt.hist(scores_p, bins=bins, density=True, alpha=0.6, label="p (background)", color ="red")

    plt.xlabel("BDT score")
    plt.ylabel("Normalized density")
    plt.title("BDT score distribution (1D)")

    plt.legend()
    plt.tight_layout()

    plt.savefig(outpath, dpi=200)
    plt.close()

def load_model_and_data(
    model_path,
    test_path,
    outdir
):
    import pathlib
    import numpy as np
    import pandas as pd
    import joblib
    import json

    # -------------------------------------------------
    # 1. LOAD MODEL
    # -------------------------------------------------
    print(f"Loading model: {model_path}")
    model_obj = joblib.load(str(model_path))

    if isinstance(model_obj, dict) and "model" in model_obj:
        model = model_obj["model"]
        feature_names = model_obj["features"]
        print(f"  Loaded wrapped model with {len(feature_names)} features")
    else:
        model = model_obj

        manifest_path = pathlib.Path(test_path).resolve().parents[0] / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json at {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        feature_names = manifest.get("feature_list") or manifest.get("columns")
        if feature_names is None:
            raise ValueError("manifest.json missing feature_list/columns")

        print(f"  Loaded legacy model with {len(feature_names)} features")

    # -------------------------------------------------
    # 2. LOAD DATA
    # -------------------------------------------------
    print(f"Loading data: {test_path}")
    df = pd.read_parquet(str(test_path))
    #df = uproot.open("/volatile/clas12/cooperb/SULI/pid_training_v2.root:PhysicsEvents").arrays(library="pd")
    print(f"  Rows: {len(df):,}")

    # -------------------------------------------------
    # 3. SCORE MODEL
    # -------------------------------------------------
    X = df[feature_names].to_numpy(dtype=np.float32, na_value=np.nan)
    scores = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["score"] = scores

    # -------------------------------------------------
    # 4. BASIC SANITY CHECK (IMPORTANT FOR YOUR ISSUE)
    # -------------------------------------------------
    print("\n--- SANITY CHECK ---")
    print("Total events:", len(df))
    print("True K:", (df["mc_matching_pid"] == 321).sum())
    print("Reco K (pid=321):", (df["pid"] == 321).sum())
    print("--------------------\n")
    print("unique pid values:", df["pid"].value_counts().head(10))
    print("\n--- SANITY CHECK ---POST CUT chi<3")
    print("Total events:", len(df))
    print("True K:", ((df["mc_matching_pid"] == 321)&(abs(df["chi2pid"])<3)).sum())
    print("Reco K (pid=321):", ((df["pid"] == 321)&(abs(df["chi2pid"])<3)).sum())
    print("--------------------\n")
    print("unique pid values:", df["pid"].value_counts().head(10))

    print("true K & pid==321:",
      ((df["mc_matching_pid"] == 321) & (df["pid"] == 321)).sum())

    print("true K & pid!=321:",
      ((df["mc_matching_pid"] == 321) & (df["pid"] != 321)).sum())

    print("non-true K but pid==321:",
      ((df["mc_matching_pid"] != 321) & (df["pid"] == 321)).sum())

    PlotMLMATCHcontams_theta(
        df=df,
        pStart=0.5,
        pEnd=3.2,
        pBinN=10,
        theta_edges=np.linspace(5, 35, 6),
        outdir=outdir
    )
    PlotMLMATCHeffs_theta(
        df=df,
        pStart=0.5,
        pEnd=3.2,
        pBinN=10,
        theta_edges=np.linspace(5, 35, 6),
        outdir=outdir
    )
    PlotMLContamCompare(
        df=df,
        pStart=0.5,
        pEnd=3.2,
        pBinN=10,
        outdir=outdir,
        name="low_5-11_",
        tStart=5,
        tEnd=11,
        label="5<theta<11"
    )
    PlotMLContamCompare(
        df=df,
        pStart=0.5,
        pEnd=3.2,
        pBinN=10,
        outdir=outdir,
        name="high_17-23_",
        tStart=17,
        tEnd=23,
        label="17<theta<23"
    )

  


    LESSPlotMLMATCHcontams_theta(
        df=df,
        pStart=0.5,
        pEnd=3.2,
        pBinN=10,
        theta_edges=np.linspace(5, 35, 6),
        outdir=outdir
    )
    LESSPlotMLMATCHeffs_theta(
        df=df,
        pStart=0.5,
        pEnd=3.2,
        pBinN=10,
        theta_edges=np.linspace(5, 35, 6),
        outdir=outdir
    )
    LESSPlotMLContamCompare(
        df=df,
        pStart=0.5,
        pEnd=3.2,
        pBinN=10,
        outdir=outdir,
        name="low_5-11_",
        tStart=5,
        tEnd=11,
        label="5<theta<11"
    )
    LESSPlotMLContamCompare(
        df=df,
        pStart=0.5,
        pEnd=3.2,
        pBinN=10,
        outdir=outdir,
        name="high_17-23_",
        tStart=17,
        tEnd=23,
        label="17<theta<23"
    )
    


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

    
    load_model_and_data(
        model_path=model_path,
        test_path=test_path,
        outdir=outdir
    )



    
    print("\nDone.")
    print(f"  Comparison summary: {outdir / 'comparison_summary.csv'}")
    print(f"  Headline plot:      {outdir / 'contam_vs_ptheta_baseline_vs_bdt.png'}")
    print(f"  C_p map:            {outdir / 'cp_to_K_map.png'}")




if __name__ == "__main__":
    main()