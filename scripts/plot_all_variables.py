"""
plot_all_variables.py
---------------------
Diagnostic plots for ALL 53 variables in the ML PID training ntuple produced by
processing_mc_pid_training.groovy + convert_txt_to_root.cpp (script_index=7).

Produces one PNG per variable, grouped into subdirectories by variable family.
Also writes a plain-text summary of row counts and per-variable missing fractions.

Usage
~~~~~
    python plot_all_variables.py <root_file> [--output-dir <dir>] [--max-rows N]

    root_file    Path to a ROOT file containing the PhysicsEvents TTree.
                 If you need a different tree name, use the standard
                 uproot syntax:  file.root:TreeName
    --output-dir Output directory (default: ./figures/variable_check/)
    --max-rows   Load only the first N rows (useful for quick tests)

Column map (53 branches, script_index == 7)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Event-level (1-8):
    runnum, evnum, helicity, Q2, W, x, y, nu
  Per-track kinematics (9-15):
    pid, p, theta, phi, vz, sector, status
  ML features (16-30):
    beta, chi2pid,
    ftof_energy_1A, ftof_energy_1B, ftof_time_1A, ftof_time_1B,
    ftof_path_1A, ftof_path_1B,
    ecin_energy, ecout_energy, ecin_time, ecout_time, ecin_path, ecout_path,
    nphe_htcc
  PCAL + FTOF layer 2 (31-36):
    pcal_energy, pcal_time, pcal_path,
    ftof_energy_2, ftof_time_2, ftof_path_2
  RICH (37-50):
    rich_emilay, rich_emico, rich_emqua, rich_best_PID,
    rich_RQ, rich_ReQ, rich_el_logl, rich_pi_logl, rich_k_logl, rich_pr_logl,
    rich_best_ch, rich_best_c2, rich_best_RL, rich_best_ntot
  MC truth (51-53):
    mc_matching_pid, mc_parent_pid, mc_match_quality

Missing-value convention:  -9999 for all columns.
chi2pid also has a +9999 sentinel (absent/bad fit); both are excluded for
chi2pid distributions.
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot


# ── Column catalogue ──────────────────────────────────────────────────────────

# Columns that are logically integer-valued and plotted as bar charts.
# (rich_emilay, rich_emico, rich_emqua, rich_best_PID are stored as /D in ROOT
# but are integer-valued PIDs/layer indices → also bar-chart treatment.)
INTEGER_COLS = {
    "runnum", "evnum", "helicity",
    "pid", "sector", "status",
    "mc_matching_pid", "mc_parent_pid",
    "rich_emilay", "rich_emico", "rich_emqua", "rich_best_PID",
}

# Four truth classes used in per-class overlays.
TRUTH_CLASSES = [211, 321, 2212, -9999]
TRUTH_COLORS  = {211: "tab:blue", 321: "tab:orange", 2212: "tab:green", -9999: "tab:red"}
TRUTH_LABELS  = {211: "π⁺ (211)", 321: "K⁺ (321)", 2212: "p (2212)", -9999: "unmatched"}

# Subdirectory groups: (subdir_name, [column_names_in_order])
GROUPS = [
    ("event_level", [
        "runnum", "evnum", "helicity",
        "Q2", "W", "x", "y", "nu",
    ]),
    ("kinematics", [
        "pid", "p", "theta", "phi", "vz", "sector", "status",
    ]),
    ("ml_features", [
        "beta", "chi2pid",
        "ftof_energy_1A", "ftof_energy_1B",
        "ftof_time_1A",   "ftof_time_1B",
        "ftof_path_1A",   "ftof_path_1B",
        "ecin_energy",    "ecout_energy",
        "ecin_time",      "ecout_time",
        "ecin_path",      "ecout_path",
        "nphe_htcc",
    ]),
    ("pcal_ftof2", [
        "pcal_energy", "pcal_time", "pcal_path",
        "ftof_energy_2", "ftof_time_2", "ftof_path_2",
    ]),
    ("rich", [
        "rich_emilay", "rich_emico", "rich_emqua", "rich_best_PID",
        "rich_RQ",     "rich_ReQ",
        "rich_el_logl", "rich_pi_logl", "rich_k_logl", "rich_pr_logl",
        "rich_best_ch", "rich_best_c2", "rich_best_RL", "rich_best_ntot",
    ]),
    ("mc_truth", [
        "mc_matching_pid", "mc_parent_pid", "mc_match_quality",
    ]),
]

# Columns that should NOT get the per-truth-class split panel
# (event-level identifiers and the truth labels themselves).
NO_SPLIT_COLS = {
    "runnum", "evnum", "helicity",
    "Q2", "W", "x", "y", "nu",
    "mc_matching_pid", "mc_parent_pid", "mc_match_quality",
}

SENTINEL = -9999


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_dataframe(root_path: str, max_rows: int | None) -> pd.DataFrame:
    """Open a ROOT file with uproot and return a pandas DataFrame."""
    # Support  "file.root:TreeName"  syntax; default tree is PhysicsEvents.
    if ":" in root_path and not root_path.startswith("/"):
        file_path, tree_name = root_path.rsplit(":", 1)
    else:
        # Could be an absolute path with a colon-separated tree at the end
        parts = root_path.rsplit(":", 1)
        if len(parts) == 2 and not os.path.exists(root_path) and os.path.exists(parts[0]):
            file_path, tree_name = parts
        else:
            file_path  = root_path
            tree_name  = "PhysicsEvents"

    print(f"Opening: {file_path}  (tree: {tree_name})")
    with uproot.open(file_path) as f:
        tree = f[tree_name]
        available = set(tree.keys())
        # Build the full expected column list from GROUPS
        expected = [col for _, cols in GROUPS for col in cols]
        missing_in_file = [c for c in expected if c not in available]
        if missing_in_file:
            print(f"  WARNING: the following expected columns are absent from "
                  f"the ROOT file and will be skipped:\n  {missing_in_file}")
        to_load = [c for c in expected if c in available]
        entry_stop = max_rows if max_rows is not None else None
        df = tree.arrays(to_load, library="pd", entry_stop=entry_stop)

    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.")
    return df


# ── Missing-value helpers ─────────────────────────────────────────────────────

def is_missing(series: pd.Series, col: str) -> pd.Series:
    """Return boolean mask for sentinel / missing values."""
    mask = series == SENTINEL
    if col == "chi2pid":
        mask = mask | (series == 9999)
    return mask


def non_missing(series: pd.Series, col: str) -> pd.Series:
    return series[~is_missing(series, col)]


def missing_frac(series: pd.Series, col: str) -> float:
    return is_missing(series, col).mean()


# ── Plotting functions ─────────────────────────────────────────────────────────

def _percentile_range(arr: np.ndarray) -> tuple[float, float]:
    """Return 1st–99th percentile x-range for an array of non-missing values."""
    if len(arr) == 0:
        return 0.0, 1.0
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    return lo, hi


def plot_integer_col(col: str, series: pd.Series, out_path: str) -> None:
    """Bar chart of value_counts (top 15), missing count in corner."""
    fig, ax = plt.subplots(figsize=(10, 4))

    miss_n = int(is_missing(series, col).sum())
    valid  = series[~is_missing(series, col)]

    if len(valid) == 0:
        ax.text(0.5, 0.5, "All values missing", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
    else:
        counts = valid.value_counts().head(15).sort_index()
        ax.bar([str(int(v)) for v in counts.index], counts.values, color="steelblue",
               edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    ax.set_title(col, fontsize=13, fontweight="bold")
    ax.text(0.98, 0.97, f"missing (={SENTINEL}): {miss_n:,}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="firebrick",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_continuous_col(col: str, series: pd.Series,
                        mc_pid: pd.Series | None,
                        out_path: str) -> None:
    """
    1D histogram (left panel) + per-truth-class overlay (right panel, if applicable).
    """
    do_split = (mc_pid is not None) and (col not in NO_SPLIT_COLS)
    figsize  = (12, 4) if do_split else (10, 4)
    n_axes   = 2 if do_split else 1

    fig, axes = plt.subplots(1, n_axes, figsize=figsize)
    if n_axes == 1:
        axes = [axes]

    # --- Left panel: overall 1D histogram ---
    ax = axes[0]
    valid = non_missing(series, col).to_numpy(dtype=float)
    n_total   = len(series)
    n_missing = int(is_missing(series, col).sum())
    frac_miss = 100.0 * n_missing / n_total if n_total > 0 else 0.0

    if len(valid) == 0:
        ax.text(0.5, 0.5, "All values missing", ha="center", va="center",
                transform=ax.transAxes, fontsize=13)
    else:
        lo, hi = _percentile_range(valid)
        ax.hist(valid, bins=80, range=(lo, hi),
                color="steelblue", alpha=0.85, edgecolor="none")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    title = (f"{col}\n"
             f"[{n_total - n_missing:,} non-missing, {frac_miss:.1f}% missing]")
    ax.set_title(title, fontsize=10, fontweight="bold")

    # --- Right panel: per-truth-class overlay ---
    if do_split:
        ax2 = axes[1]
        valid_all = non_missing(series, col)
        if mc_pid is not None:
            mc_pid_valid = mc_pid.loc[valid_all.index]
        plotted_any = False
        for pid_val in TRUTH_CLASSES:
            mask = mc_pid_valid == pid_val
            vals = valid_all[mask].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            lo, hi = _percentile_range(vals)
            # Use a shared range for cleaner comparison: overall non-missing range
            ax2.hist(vals, bins=60, density=True,
                     histtype="stepfilled", alpha=0.40,
                     color=TRUTH_COLORS[pid_val],
                     label=f"{TRUTH_LABELS[pid_val]} (n={len(vals):,})")
            ax2.hist(vals, bins=60, density=True,
                     histtype="step", linewidth=1.2,
                     color=TRUTH_COLORS[pid_val])
            plotted_any = True

        if plotted_any:
            # Set x-range to overall valid range
            lo_all, hi_all = _percentile_range(valid_all.to_numpy(dtype=float))
            ax2.set_xlim(lo_all, hi_all)
            ax2.legend(fontsize=7, loc="upper right")
        else:
            ax2.text(0.5, 0.5, "No data per class", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=12)

        ax2.set_xlabel(col)
        ax2.set_ylabel("Density")
        ax2.set_title(f"{col} — by mc_matching_pid", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ── Summary writer ────────────────────────────────────────────────────────────

def write_summary(df: pd.DataFrame, out_path: str) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append("NTUPLE DIAGNOSTIC SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Total rows: {len(df):,}")
    lines.append("")

    # EB pid counts
    if "pid" in df.columns:
        lines.append("EB-reconstructed pid counts:")
        for pid_val, cnt in df["pid"].value_counts().sort_index().items():
            lines.append(f"  pid={int(pid_val):6d}  {cnt:>10,}")
        lines.append("")

    # Truth class counts
    if "mc_matching_pid" in df.columns:
        lines.append("mc_matching_pid counts (truth labels):")
        for pid_val, cnt in df["mc_matching_pid"].value_counts().sort_index().items():
            lines.append(f"  mc_matching_pid={int(pid_val):6d}  {cnt:>10,}")
        lines.append("")

    # Per-variable missing fractions, sorted descending
    lines.append("Missing-value fractions per column (sorted descending):")
    lines.append(f"  {'column':<22}  {'missing_n':>10}  {'missing_%':>9}")
    lines.append("  " + "-" * 46)
    fracs = []
    all_cols = [col for _, cols in GROUPS for col in cols if col in df.columns]
    for col in all_cols:
        miss_n = int(is_missing(df[col], col).sum())
        frac   = 100.0 * miss_n / len(df) if len(df) > 0 else 0.0
        fracs.append((col, miss_n, frac))
    fracs.sort(key=lambda t: t[2], reverse=True)
    for col, miss_n, frac in fracs:
        lines.append(f"  {col:<22}  {miss_n:>10,}  {frac:>8.2f}%")

    lines.append("=" * 72)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote summary → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce diagnostic plots for all variables in the ML PID training ntuple."
    )
    parser.add_argument("root_file",
                        help="Path to ROOT file (PhysicsEvents tree assumed unless "
                             "you use file.root:TreeName syntax).")
    parser.add_argument("--output-dir", default="./figures/variable_check/",
                        help="Output directory for PNGs and summary.txt "
                             "(default: ./figures/variable_check/).")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Load only this many rows (for quick tests).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    df = load_dataframe(args.root_file, args.max_rows)

    # mc_matching_pid column used for per-class splits
    mc_pid_col = df["mc_matching_pid"] if "mc_matching_pid" in df.columns else None

    # ── Create output directory tree ───────────────────────────────────────
    out_root = args.output_dir
    for subdir, _ in GROUPS:
        os.makedirs(os.path.join(out_root, subdir), exist_ok=True)

    # ── Enumerate all columns with their subdir ────────────────────────────
    col_subdir: list[tuple[str, str]] = []
    for subdir, cols in GROUPS:
        for col in cols:
            col_subdir.append((col, subdir))

    total = len(col_subdir)

    # ── Plot each column ───────────────────────────────────────────────────
    for i, (col, subdir) in enumerate(col_subdir, start=1):
        print(f"  Plotting {i}/{total}: {col}  [{subdir}]")

        if col not in df.columns:
            print(f"    WARNING: '{col}' not in DataFrame — skipping.")
            continue

        series   = df[col]
        out_path = os.path.join(out_root, subdir, f"{col}.png")

        if col in INTEGER_COLS:
            plot_integer_col(col, series, out_path)
        else:
            plot_continuous_col(col, series, mc_pid_col, out_path)

    # ── Summary ────────────────────────────────────────────────────────────
    write_summary(df, os.path.join(out_root, "summary.txt"))
    print(f"\nDone. All plots saved under: {out_root}")


if __name__ == "__main__":
    main()
