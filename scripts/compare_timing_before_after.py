"""
compare_timing_before_after.py — Side-by-side comparison of feature-audit
summaries before and after a preprocessing change (e.g. negative-time removal).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads two feature_audit_summary.csv files — one produced *before* a data-
cleaning step ("before") and one produced *after* ("after") — and inner-joins
them on the five kinematic-cell key columns:

    variable, p_lo, p_hi, theta_lo, theta_hi

For each matched row it emits a single wide output row containing:
  • The cell key columns (variable, p_lo, p_hi, theta_lo, theta_hi)
  • For the BEFORE audit: wasserstein_norm, psi, max_local_residual,
    drift_decision, n_mc, n_data, hit_frac_mc, hit_frac_data
  • For the AFTER  audit: same eight columns (suffixed _after)
  • Columns present only in the AFTER CSV (e.g. neg_time_frac_removed_mc,
    neg_time_frac_removed_data) are included as-is from the after row; they
    are silently skipped if absent.
  • Delta columns (after − before) for the three primary drift metrics:
    wasserstein_norm_delta, psi_delta, max_local_residual_delta

The join is restricted to a user-specified list of variable names (default:
ecin_time, pcal_time) so the output stays focused on the variables of interest.
Rows where the variable is not in the requested list are dropped before joining.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHEN TO USE IT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use this script to quantify the impact of a preprocessing change on timing-
variable drift decisions.  Typical workflow:

  1. Run audit_species.py on the original sample → feature_audit_summary.csv A
  2. Apply preprocessing (e.g. remove negative ECAL/PCAL times in ROOT)
  3. Re-run audit_species.py on the cleaned sample → feature_audit_summary.csv B
  4. python scripts/compare_timing_before_after.py \\
         --before A/feature_audit_summary.csv \\
         --after  B/feature_audit_summary.csv \\
         --vars   ecin_time,pcal_time \\
         --output figures/timing_cleanup_comparison.csv

The delta columns show how much drift improved (or worsened).  A drop in
wasserstein_norm_delta toward 0 means the preprocessing reduced MC/data
disagreement for that cell.

Sanity-check (self-join): passing the same CSV for both --before and --after
should produce delta columns that are identically 0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PITFALLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• The join is an inner join: cells present in only one CSV are silently
  dropped.  Run with --verbose to see a count of dropped rows.
• If the two audits used different p/theta bin grids the join will be empty.
  Ensure both audits were run with the same --p-bins and --theta-bins.
• neg_time_frac_removed_* columns are taken from the AFTER CSV only.  If the
  AFTER CSV does not contain them (e.g. the audit was run without
  --remove-neg-time) they are silently omitted from the output.
• drift_decision is a categorical string (KEEP / CANDIDATE / DROP).  The
  script does not compute a numeric delta for it; inspect the _before and
  _after columns manually.
• Column names in feature_audit_summary.csv are fixed by compare_mc_data.py.
  If that engine is updated the JOIN_KEYS and METRIC_COLS constants below
  may need to be updated to match.
"""

import argparse
import sys

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Constants — keep in sync with compare_mc_data.py output schema
# ──────────────────────────────────────────────────────────────────────────────

# Columns used to match rows between the two CSVs.
JOIN_KEYS = ["variable", "p_lo", "p_hi", "theta_lo", "theta_hi"]

# Columns carried from each CSV into the output (suffixed _before / _after).
CARRY_COLS = [
    "wasserstein_norm",
    "psi",
    "max_local_residual",
    "drift_decision",
    "n_mc",
    "n_data",
    "hit_frac_mc",
    "hit_frac_data",
]

# Numeric drift metrics for which delta = after − before is computed.
DELTA_METRICS = ["wasserstein_norm", "psi", "max_local_residual"]

# Columns that may appear only in the AFTER CSV (e.g. added by neg-time removal
# preprocessing).  Included as-is from the after row if present.
AFTER_ONLY_COLS = [
    "neg_time_frac_removed_mc",
    "neg_time_frac_removed_data",
]

DEFAULT_VARS = ["ecin_time", "pcal_time"]


# ──────────────────────────────────────────────────────────────────────────────
# Core comparison logic
# ──────────────────────────────────────────────────────────────────────────────

def compare_audits(before_path, after_path, variables, output_path, verbose=False):
    """Load two feature_audit_summary CSVs, join them, and write a comparison.

    Parameters
    ----------
    before_path : str
        Path to the "before" feature_audit_summary.csv.
    after_path : str
        Path to the "after" feature_audit_summary.csv.
    variables : list of str
        Variable names to include (rows not in this list are dropped).
    output_path : str
        Destination CSV path for the side-by-side comparison table.
    verbose : bool
        If True, print extra join diagnostics to stdout.

    Returns
    -------
    pd.DataFrame
        The comparison DataFrame (also written to output_path).

    Raises
    ------
    SystemExit
        On missing input files or missing required columns.
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        before = pd.read_csv(before_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: --before file not found: {before_path}")
    try:
        after = pd.read_csv(after_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: --after file not found: {after_path}")

    # ── Validate required columns ──────────────────────────────────────────────
    for label, df, path in [("before", before, before_path),
                             ("after",  after,  after_path)]:
        missing = [c for c in JOIN_KEYS + CARRY_COLS if c not in df.columns]
        if missing:
            sys.exit(
                f"ERROR: {label} CSV ({path}) is missing required columns:\n"
                f"  {missing}\n"
                f"  Found columns: {list(df.columns)}"
            )

    # ── Filter to requested variables ─────────────────────────────────────────
    before_filt = before[before["variable"].isin(variables)].copy()
    after_filt  = after[after["variable"].isin(variables)].copy()

    if verbose:
        print(f"  before rows (all vars): {len(before)}  → after variable filter: {len(before_filt)}")
        print(f"  after  rows (all vars): {len(after)}   → after variable filter: {len(after_filt)}")

    if before_filt.empty:
        sys.exit(
            f"ERROR: No rows in --before CSV match requested variables: {variables}\n"
            f"  Variables present: {sorted(before['variable'].unique())}"
        )
    if after_filt.empty:
        sys.exit(
            f"ERROR: No rows in --after CSV match requested variables: {variables}\n"
            f"  Variables present: {sorted(after['variable'].unique())}"
        )

    # ── Select columns to carry from each side ────────────────────────────────
    before_cols = JOIN_KEYS + CARRY_COLS
    after_extra = [c for c in AFTER_ONLY_COLS if c in after_filt.columns]
    after_cols  = JOIN_KEYS + CARRY_COLS + after_extra

    before_sel = before_filt[before_cols].copy()
    after_sel  = after_filt[after_cols].copy()

    # ── Inner join ────────────────────────────────────────────────────────────
    merged = before_sel.merge(
        after_sel,
        on=JOIN_KEYS,
        how="inner",
        suffixes=("_before", "_after"),
    )

    n_before = len(before_filt)
    n_after  = len(after_filt)
    n_merged = len(merged)
    n_dropped_before = n_before - n_merged
    n_dropped_after  = n_after  - n_merged

    if verbose:
        print(f"  Inner join: {n_before} before × {n_after} after → {n_merged} matched rows")
        if n_dropped_before or n_dropped_after:
            print(f"  Dropped (not in both): {n_dropped_before} before rows, {n_dropped_after} after rows")

    if merged.empty:
        sys.exit(
            "ERROR: Inner join produced no rows.  Check that --before and --after\n"
            "  were produced with the same p/theta bin grids and the same variable names.\n"
            f"  before variables: {sorted(before_filt['variable'].unique())}\n"
            f"  after  variables: {sorted(after_filt['variable'].unique())}"
        )

    # ── Delta columns ─────────────────────────────────────────────────────────
    for metric in DELTA_METRICS:
        col_b = f"{metric}_before"
        col_a = f"{metric}_after"
        if col_b in merged.columns and col_a in merged.columns:
            merged[f"{metric}_delta"] = merged[col_a] - merged[col_b]

    # ── Reorder columns for readability ───────────────────────────────────────
    key_block    = JOIN_KEYS
    before_block = [f"{c}_before" for c in CARRY_COLS if f"{c}_before" in merged.columns]
    after_block  = [f"{c}_after"  for c in CARRY_COLS if f"{c}_after"  in merged.columns]
    extra_block  = [c for c in after_extra if c in merged.columns]
    delta_block  = [f"{m}_delta" for m in DELTA_METRICS if f"{m}_delta" in merged.columns]

    ordered_cols = key_block + before_block + after_block + extra_block + delta_block
    # Safety: include any remaining columns we may have missed.
    remaining = [c for c in merged.columns if c not in ordered_cols]
    merged = merged[ordered_cols + remaining]

    # ── Write output ──────────────────────────────────────────────────────────
    merged.to_csv(output_path, index=False)
    print(f"Wrote {n_merged} rows × {len(merged.columns)} columns → {output_path}")

    # ── Summary printout ──────────────────────────────────────────────────────
    print("\nDelta summary (after − before):")
    for metric in DELTA_METRICS:
        dcol = f"{metric}_delta"
        if dcol in merged.columns:
            vals = merged[dcol]
            print(
                f"  {dcol:30s}  mean={vals.mean():+.4f}  "
                f"min={vals.min():+.4f}  max={vals.max():+.4f}  "
                f"all_zero={vals.abs().max() < 1e-12}"
            )

    drift_b = f"drift_decision_before"
    drift_a = f"drift_decision_after"
    if drift_b in merged.columns and drift_a in merged.columns:
        changed = merged[merged[drift_b] != merged[drift_a]]
        print(f"\nDrift decision changes: {len(changed)} / {n_merged} cells")
        if not changed.empty:
            for _, row in changed.iterrows():
                print(
                    f"  {row['variable']}  p=[{row['p_lo']},{row['p_hi']}]  "
                    f"theta=[{row['theta_lo']},{row['theta_hi']}]  "
                    f"{row[drift_b]} → {row[drift_a]}"
                )

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Side-by-side comparison of feature-audit summaries before and after\n"
            "a preprocessing change (e.g. negative-time removal).\n\n"
            "Inner-joins two feature_audit_summary.csv files on the kinematic-cell\n"
            "key (variable, p_lo, p_hi, theta_lo, theta_hi) and computes delta\n"
            "columns (after − before) for the three primary drift metrics.\n\n"
            "Example:\n"
            "  python scripts/compare_timing_before_after.py \\\n"
            "      --before figures/feature_audit_week3/kp/feature_audit_summary.csv \\\n"
            "      --after  figures/feature_audit_week4/kp/feature_audit_summary.csv \\\n"
            "      --vars   ecin_time,pcal_time \\\n"
            "      --output figures/timing_cleanup_comparison.csv\n\n"
            "Self-join sanity check (all deltas should be ~0):\n"
            "  python scripts/compare_timing_before_after.py \\\n"
            "      --before figures/feature_audit_week3/kp/feature_audit_summary.csv \\\n"
            "      --after  figures/feature_audit_week3/kp/feature_audit_summary.csv \\\n"
            "      --output /tmp/test_timing_compare.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--before", required=True, metavar="CSV",
        help="Path to the 'before' feature_audit_summary.csv.",
    )
    p.add_argument(
        "--after", required=True, metavar="CSV",
        help="Path to the 'after' feature_audit_summary.csv.",
    )
    p.add_argument(
        "--vars", default=",".join(DEFAULT_VARS), metavar="VAR1,VAR2,...",
        help=(
            f"Comma-separated list of variable names to compare. "
            f"Default: {','.join(DEFAULT_VARS)}"
        ),
    )
    p.add_argument(
        "--output", required=True, metavar="CSV",
        help="Destination path for the comparison CSV.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print join diagnostics (row counts, dropped rows).",
    )
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    variables = [v.strip() for v in args.vars.split(",") if v.strip()]
    if not variables:
        sys.exit("ERROR: --vars produced an empty variable list.")
    compare_audits(
        before_path=args.before,
        after_path=args.after,
        variables=variables,
        output_path=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
