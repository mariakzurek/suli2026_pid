"""
audit_species.py — Species-specific MC vs data feature audit driver.

Wraps compare_mc_data.run_feature_audit with SULI-2026-specific species cuts
and MC truth-match modes.  The generic engine stays generic; this script
encodes the project workflow: species pid cut, truth-match mode, and a
provenance README written into each output directory.

Species map:  kp→321 (K+), pip→211 (π+), p→2212 (p),
              em→11 (e-), pim→-211 (π-), kn→-321 (K-)

Truth modes (--truth-mode):
  matched (default) — (pid==SPEC) & (mc_matching_pid!=-9999)
      Use for ML feature drift: audits the EB-labeled sample the classifier
      will see, including mis-IDs.  Correct choice for Task 3a.
  pure              — (pid==SPEC) & (mc_matching_pid==SPEC)
      Truth-pure tracks only.  For detector-response physics studies, not
      ML feature audits.
  off               — pid==SPEC only.
      No truth-match cut.  Use when truth matching is suspect or for
      apples-to-apples with data without extra MC quality requirements.

Data selection is always pid==SPEC (data has no mc_matching_pid column).

Canonical invocation:
    python scripts/audit_species.py \\
        --mc   /volatile/clas12/<user>/SULI/mc_pid_training_full.root \\
        --data /volatile/clas12/<user>/SULI/data_pid_training.root \\
        --species kp --vars all_audit kinematics \\
        --outdir figures/feature_audit/kp
"""

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

# Support `python scripts/audit_species.py` from the repo root.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from compare_mc_data import (   # noqa: E402
    _load_file, _resolve_variables, run_feature_audit,
    print_summary_table, SENTINEL_LOW,
)


SPECIES_MAP = {
    "kp"  : {"pid": 321,   "label": "K+"},
    "pip" : {"pid": 211,   "label": "pi+"},
    "p"   : {"pid": 2212,  "label": "p"},
    "em"  : {"pid": 11,    "label": "e-"},
    "pim" : {"pid": -211,  "label": "pi-"},
    "kn"  : {"pid": -321,  "label": "K-"},
}
HIT_FRAC_THRESHOLD = 0.05   # flag |Δhit| > this


def _filter_mc(df, pid, truth_mode):
    """Apply species + truth-match filter to the MC DataFrame."""
    base = df["pid"] == pid
    if truth_mode == "matched":
        mask = base & (df["mc_matching_pid"] != SENTINEL_LOW)
    elif truth_mode == "pure":
        mask = base & (df["mc_matching_pid"] == pid)
    elif truth_mode == "off":
        mask = base
    else:
        raise ValueError(f"Unknown truth_mode: {truth_mode!r}")
    return df[mask].reset_index(drop=True)


def _filter_data(df, pid):
    """Apply species filter to the data DataFrame."""
    return df[df["pid"] == pid].reset_index(drop=True)


def _write_audit_readme(outdir, pid, label, truth_mode, mc_path, data_path, variables):
    """Write a provenance README (≤20 lines) into outdir."""
    now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mc_sel = {
        "matched": f"(pid == {pid}) & (mc_matching_pid != {SENTINEL_LOW})",
        "pure":    f"(pid == {pid}) & (mc_matching_pid == {pid})",
        "off":     f"pid == {pid}",
    }[truth_mode]
    content = (
        f"# Feature audit — species {label} (pid={pid})\n\n"
        f"Generated: {now}\n\n"
        f"## Selections applied\n"
        f"  MC   : {mc_sel}\n"
        f"  Data : pid == {pid}\n"
        f"  Truth mode: {truth_mode}\n\n"
        f"## Input files\n"
        f"  MC   : {mc_path}\n"
        f"  Data : {data_path}\n\n"
        f"## Variables audited\n"
        f"  {', '.join(variables)}\n\n"
        f"## Column meanings\n"
        f"  See scripts/README.md (compare_mc_data.py Output section) for CSV column definitions.\n"
        f"  See notes/cooper_10week_plan.md Task 3a for the full audit workflow.\n"
    )
    readme_path = os.path.join(outdir, "README.md")
    with open(readme_path, "w") as fh:
        fh.write(content)
    print(f"  Provenance README written to: {readme_path}")


def _print_hit_fraction_alerts(summary):
    """List (variable, cell) pairs where |hit_frac_delta| > HIT_FRAC_THRESHOLD."""
    if "hit_frac_delta" not in summary.columns:
        return
    flagged = summary[summary["hit_frac_delta"].abs() > HIT_FRAC_THRESHOLD]
    if flagged.empty:
        print(f"  No hit-fraction mismatches above {HIT_FRAC_THRESHOLD} threshold.")
        return
    for _, row in flagged.iterrows():
        delta = row["hit_frac_delta"]
        sign  = "+" if delta >= 0 else ""
        print(
            f"  ⚑ hit-fraction mismatch: {row['variable']} "
            f"in p=[{row['p_lo']},{row['p_hi']}] theta=[{row['theta_lo']},{row['theta_hi']}]: "
            f"MC={row['hit_frac_mc_cell']:.3f} Data={row['hit_frac_data_cell']:.3f} "
            f"(Δ={sign}{delta:.3f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Species-specific MC vs data feature audit driver.\n"
            "Wraps compare_mc_data.run_feature_audit with SULI-specific selections."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mc",   required=True,
                   help="MC ROOT file path. Supports 'file.root:TreeName' syntax.")
    p.add_argument("--data", required=True,
                   help="Data ROOT file path, same syntax.")
    p.add_argument("--species", required=True, choices=list(SPECIES_MAP),
                   help="Species alias: kp | pip | p | em | pim | kn")
    p.add_argument("--vars", nargs="+", default=["all_audit", "kinematics"],
                   help="Variable names or group aliases. Default: all_audit kinematics")
    p.add_argument("--outdir", default=None,
                   help="Output directory. Default: figures/feature_audit/<species>")
    p.add_argument("--truth-mode", default="matched",
                   choices=["matched", "pure", "off"],
                   help="MC truth-match mode. Default: matched")
    p.add_argument("--bins", type=int, default=50,
                   help="Histogram bin count. Default 50.")
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip histogram normalisation.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Load at most N rows per file.")
    p.add_argument("--label-mc",   default=None,
                   help="MC label in plots. Default: 'MC <label>'")
    p.add_argument("--label-data", default=None,
                   help="Data label in plots. Default: 'Data <label>'")
    return p


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

    spec    = SPECIES_MAP[args.species]
    pid     = spec["pid"]
    label   = spec["label"]
    outdir  = args.outdir or os.path.join("figures", "feature_audit", args.species)
    lmc     = args.label_mc   or f"MC {label}"
    ldata   = args.label_data or f"Data {label}"

    mc_sel_str = {
        "matched": f"(pid=={pid}) & (mc_matching_pid!={SENTINEL_LOW})",
        "pure":    f"(pid=={pid}) & (mc_matching_pid=={pid})",
        "off":     f"pid=={pid}",
    }[args.truth_mode]
    data_sel_str = f"pid=={pid}"

    variables = _resolve_variables(args.vars)

    # ── Preamble ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"audit_species.py  —  {label} (pid={pid})")
    print(f"{'='*65}")
    print(f"  Species     : {args.species} → {label} (pid={pid})")
    print(f"  Truth mode  : {args.truth_mode}")
    print(f"  MC selection: {mc_sel_str}")
    print(f"  Data select : {data_sel_str}")
    print(f"  Variables   : {variables}")
    print(f"  Output dir  : {outdir}")
    print(f"  Bins        : {args.bins}   Normalize: {not args.no_normalize}")

    # ── Load files ─────────────────────────────────────────────────────────────
    print(f"\nLoading MC:   {args.mc}")
    df_mc_raw   = _load_file(args.mc,   max_rows=args.max_rows)
    print(f"Loading Data: {args.data}")
    df_data_raw = _load_file(args.data, max_rows=args.max_rows)

    # ── Filter by species (+ truth match for MC) ───────────────────────────────
    df_mc   = _filter_mc(df_mc_raw,   pid, args.truth_mode)
    df_data = _filter_data(df_data_raw, pid)

    if len(df_mc) == 0:
        sys.exit(
            f"\nERROR: MC DataFrame is empty after species+truth-match filter "
            f"({mc_sel_str}).  Check that the MC file contains pid={pid} events "
            f"and that mc_matching_pid is present for truth modes 'matched'/'pure'."
        )
    if len(df_data) == 0:
        sys.exit(
            f"\nERROR: Data DataFrame is empty after species filter "
            f"({data_sel_str}).  Check that the data file contains pid={pid} events."
        )

    print(f"\n  MC after species + truth-match selection : {len(df_mc):,}")
    print(f"  Data after species selection             : {len(df_data):,}")

    # ── Run audit ─────────────────────────────────────────────────────────────
    os.makedirs(outdir, exist_ok=True)
    print(f"\nRunning feature audit …\n")
    summary = run_feature_audit(
        df_mc, df_data,
        variables=variables,
        output_dir=outdir,
        bins=args.bins,
        normalize=not args.no_normalize,
        label_mc=lmc,
        label_data=ldata,
    )

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(outdir, "feature_audit_summary.csv")
    summary.to_csv(csv_path, index=False, float_format="%.6g")
    print(f"\nSummary CSV saved to: {csv_path}")

    # ── Write provenance README ────────────────────────────────────────────────
    _write_audit_readme(outdir, pid, label, args.truth_mode,
                        args.mc, args.data, variables)

    # ── Per-variable summary table ─────────────────────────────────────────────
    print_summary_table(summary, variables)

    # ── Hit-fraction alert section ─────────────────────────────────────────────
    print(f"\nHit-fraction mismatches (|Δ| > {HIT_FRAC_THRESHOLD}):")
    _print_hit_fraction_alerts(summary)

    # ── Next steps ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Next steps:")
    print(f"  1. Review {csv_path}")
    print("     Fill the 'decision_notes' column: one entry per row,")
    print("     capturing any visual-cross-check observation or override.")
    print("  2. Write per-variable narrative for CANDIDATE/DROP variables")
    print("     into Section 5 of the Week 1-2 report.")
    print(f"{'─'*65}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Self-test (no CLI args, no file I/O)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        sys.exit(0)

    print("Running audit_species.py self-test …\n")

    rng = np.random.default_rng(seed=99)
    N = 5000

    # Build synthetic DataFrames with multiple pid values so the species cut
    # actually removes events.  mc_matching_pid is present on MC only.
    pids_mc   = rng.choice([321, 211, 2212, -9999], size=N,
                            p=[0.40, 0.25, 0.20, 0.15])
    match_pid = pids_mc.copy()
    # For about 20% of kp tracks, corrupt the mc_matching_pid (simulate unmatched)
    kp_idx = np.where(pids_mc == 321)[0]
    corrupt = rng.choice(kp_idx, size=max(1, len(kp_idx) // 5), replace=False)
    match_pid[corrupt] = -9999
    # For 'pure' test: ~10% of remaining kp have mc_matching_pid != 321
    kp_good = np.array([i for i in kp_idx if i not in set(corrupt)])
    mixed   = rng.choice(kp_good, size=max(1, len(kp_good) // 10), replace=False)
    match_pid[mixed] = 211  # wrong pid

    df_mc_synth = pd.DataFrame({
        "pid"             : pids_mc,
        "mc_matching_pid" : match_pid,
        "p"               : rng.uniform(1.0, 5.0, N),
        "theta"           : rng.uniform(5.0, 35.0, N),
        "beta"            : rng.uniform(0.0, 1.0, N),
        "chi2pid"         : rng.normal(0, 1, N),
    })

    pids_data = rng.choice([321, 211, 2212], size=N, p=[0.50, 0.30, 0.20])
    df_data_synth = pd.DataFrame({
        "pid"    : pids_data,
        "p"      : rng.uniform(1.0, 5.0, N),
        "theta"  : rng.uniform(5.0, 35.0, N),
        "beta"   : rng.uniform(0.0, 1.0, N),
        "chi2pid": rng.normal(0, 1, N),
    })

    pid_kp = SPECIES_MAP["kp"]["pid"]  # 321

    # ── 1. Truth mode 'off': only pid==321 cut ─────────────────────────────────
    df_mc_off = _filter_mc(df_mc_synth, pid_kp, "off")
    expected_off = int((pids_mc == 321).sum())
    assert len(df_mc_off) == expected_off, \
        f"FAIL off: expected {expected_off}, got {len(df_mc_off)}"
    print(f"  truth_mode='off':     MC rows = {len(df_mc_off)} (expected {expected_off}) ✓")

    # ── 2. Truth mode 'matched': pid==321 AND mc_matching_pid != -9999 ─────────
    df_mc_mat = _filter_mc(df_mc_synth, pid_kp, "matched")
    expected_mat = int(((pids_mc == 321) & (match_pid != -9999)).sum())
    assert len(df_mc_mat) == expected_mat, \
        f"FAIL matched: expected {expected_mat}, got {len(df_mc_mat)}"
    assert len(df_mc_mat) < len(df_mc_off), \
        "FAIL matched should be a strict subset of off"
    print(f"  truth_mode='matched': MC rows = {len(df_mc_mat)} (expected {expected_mat}) ✓")

    # ── 3. Truth mode 'pure': pid==321 AND mc_matching_pid==321 ────────────────
    df_mc_pur = _filter_mc(df_mc_synth, pid_kp, "pure")
    expected_pur = int(((pids_mc == 321) & (match_pid == 321)).sum())
    assert len(df_mc_pur) == expected_pur, \
        f"FAIL pure: expected {expected_pur}, got {len(df_mc_pur)}"
    assert len(df_mc_pur) <= len(df_mc_mat), \
        "FAIL pure should be a subset of matched"
    print(f"  truth_mode='pure':    MC rows = {len(df_mc_pur)} (expected {expected_pur}) ✓")

    # ── 4. Data filter: pid==321 only ─────────────────────────────────────────
    df_data_flt = _filter_data(df_data_synth, pid_kp)
    expected_data = int((pids_data == 321).sum())
    assert len(df_data_flt) == expected_data, \
        f"FAIL data filter: expected {expected_data}, got {len(df_data_flt)}"
    print(f"  data filter:          Data rows = {len(df_data_flt)} (expected {expected_data}) ✓")

    # ── 5. Subset ordering: pure ⊆ matched ⊆ off ──────────────────────────────
    assert len(df_mc_pur) <= len(df_mc_mat) <= len(df_mc_off), \
        "FAIL subset ordering: pure ⊆ matched ⊆ off violated"
    print("  Subset ordering pure ⊆ matched ⊆ off ✓")

    print("\naudit_species.py self-test passed.")
