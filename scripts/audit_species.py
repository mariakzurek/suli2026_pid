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

Default selections:
  A vertex-z cut of -8 < vz < 2 cm is applied by default to both MC and data
  after the species/truth-match filter; this window matches the standard SULI
  target-window definition.  Disable with --no-vz-cut.

  No event-level kinematic cuts (Q², W, y, Mx) are applied by default.
  The audit default is uncut because the trained classifier will see the
  uncut per-track sample at training time; KEEP/CANDIDATE/DROP decisions
  must reflect that.  Use --sidis-cuts or individual --q2-cut / --w-cut /
  --y-cut / --mx-cut flags for diagnostic comparison runs.

Column pruning:
  The audit loads only the ROOT branches it needs (audit variables + species
  selector + cut columns).  On production-scale ROOT files this is much faster
  than loading every branch.  Use ``--load-all-cols`` to override and load
  everything (e.g. for debugging or for ad-hoc exploration in the same Python
  session).

Canonical invocations:

  # Primary audit (no event-level cuts; matches what the classifier sees):
  python scripts/audit_species.py \\
      --mc   /volatile/clas12/<user>/SULI/mc_pid_training_full.root \\
      --data /volatile/clas12/<user>/SULI/data_pid_training.root \\
      --species kp --vars all_audit kinematics \\
      --outdir figures/feature_audit/kp

  # Diagnostic SIDIS-cut audit (Q² > 2, W > 2, y < 0.75, Mx_eKX > 1.6):
  python scripts/audit_species.py \\
      --mc   /volatile/clas12/<user>/SULI/mc_pid_training_full.root \\
      --data /volatile/clas12/<user>/SULI/data_pid_training.root \\
      --species kp --vars all_audit kinematics \\
      --sidis-cuts
  # (Output auto-suffixed to figures/feature_audit/kp_sidis/)
"""

import argparse
import datetime
import os
import shutil
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

# Species aliases for which a SIDIS Mx cut is meaningful (positive hadrons only).
_MX_SPECIES = {"kp": "Mx_eKX", "pip": "Mx_epiX", "p": "Mx_epX"}

# SIDIS defaults for --sidis-cuts convenience flag.
_SIDIS_Q2  = (2.0, float("inf"))
_SIDIS_W   = (2.0, float("inf"))
_SIDIS_Y   = (0.0, 0.75)
_SIDIS_MX_DEFAULTS = {"kp": 1.6, "pip": 1.5, "p": 1.0}


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


def _apply_vz_cut(df_mc, df_data, vz_min, vz_max):
    """Apply a vertex-z window cut to both MC and data DataFrames.

    Parameters
    ----------
    df_mc, df_data : pd.DataFrame
        Species- (and truth-match-) filtered DataFrames.
    vz_min, vz_max : float
        Exclusive bounds (cm): keep rows where vz_min < vz < vz_max.

    Returns
    -------
    (df_mc_cut, df_data_cut) : tuple of pd.DataFrame

    Raises
    ------
    SystemExit
        If the ``vz`` column is absent from either DataFrame.
    """
    missing = []
    if "vz" not in df_mc.columns:
        missing.append("MC")
    if "vz" not in df_data.columns:
        missing.append("Data")
    if missing:
        sys.exit(
            f"\nERROR: 'vz' column missing from: {', '.join(missing)} DataFrame(s).\n"
            f"  Use --no-vz-cut to skip the vertex-z cut and proceed without it."
        )
    mc_mask   = (df_mc["vz"]   > vz_min) & (df_mc["vz"]   < vz_max)
    data_mask = (df_data["vz"] > vz_min) & (df_data["vz"] < vz_max)
    return df_mc[mc_mask].reset_index(drop=True), df_data[data_mask].reset_index(drop=True)


def _apply_q2_cut(df_mc, df_data, q2_min, q2_max):
    """Apply Q² cut to both DataFrames.

    Parameters
    ----------
    df_mc, df_data : pd.DataFrame
        Species- (and truth-match-) filtered DataFrames (vz cut already applied).
    q2_min, q2_max : float
        Exclusive bounds (GeV²): keep rows where q2_min < Q2 < q2_max.
        Use float('inf') as q2_max for a one-sided lower cut.

    Returns
    -------
    (df_mc_cut, df_data_cut) : tuple of pd.DataFrame

    Raises
    ------
    SystemExit
        If the ``Q2`` column is absent from either DataFrame.
    """
    missing = []
    if "Q2" not in df_mc.columns:
        missing.append("MC")
    if "Q2" not in df_data.columns:
        missing.append("Data")
    if missing:
        sys.exit(
            f"\nERROR: 'Q2' column missing from: {', '.join(missing)} DataFrame(s).\n"
            f"  Use --no-q2-cut to skip the Q² cut and proceed without it."
        )
    mc_mask   = (df_mc["Q2"]   > q2_min) & (df_mc["Q2"]   < q2_max)
    data_mask = (df_data["Q2"] > q2_min) & (df_data["Q2"] < q2_max)
    return df_mc[mc_mask].reset_index(drop=True), df_data[data_mask].reset_index(drop=True)


def _apply_w_cut(df_mc, df_data, w_min, w_max):
    """Apply W cut to both DataFrames.

    Parameters
    ----------
    df_mc, df_data : pd.DataFrame
        Species- (and truth-match-) filtered DataFrames.
    w_min, w_max : float
        Exclusive bounds (GeV): keep rows where w_min < W < w_max.
        Use float('inf') as w_max for a one-sided lower cut.

    Returns
    -------
    (df_mc_cut, df_data_cut) : tuple of pd.DataFrame

    Raises
    ------
    SystemExit
        If the ``W`` column is absent from either DataFrame.
    """
    missing = []
    if "W" not in df_mc.columns:
        missing.append("MC")
    if "W" not in df_data.columns:
        missing.append("Data")
    if missing:
        sys.exit(
            f"\nERROR: 'W' column missing from: {', '.join(missing)} DataFrame(s).\n"
            f"  Use --no-w-cut to skip the W cut and proceed without it."
        )
    mc_mask   = (df_mc["W"]   > w_min) & (df_mc["W"]   < w_max)
    data_mask = (df_data["W"] > w_min) & (df_data["W"] < w_max)
    return df_mc[mc_mask].reset_index(drop=True), df_data[data_mask].reset_index(drop=True)


def _apply_y_cut(df_mc, df_data, y_min, y_max):
    """Apply inelasticity y cut to both DataFrames.

    Parameters
    ----------
    df_mc, df_data : pd.DataFrame
        Species- (and truth-match-) filtered DataFrames.
    y_min, y_max : float
        Exclusive bounds (dimensionless): keep rows where y_min < y < y_max.

    Returns
    -------
    (df_mc_cut, df_data_cut) : tuple of pd.DataFrame

    Raises
    ------
    SystemExit
        If the ``y`` column is absent from either DataFrame.
    """
    missing = []
    if "y" not in df_mc.columns:
        missing.append("MC")
    if "y" not in df_data.columns:
        missing.append("Data")
    if missing:
        sys.exit(
            f"\nERROR: 'y' column missing from: {', '.join(missing)} DataFrame(s).\n"
            f"  Use --no-y-cut to skip the y cut and proceed without it."
        )
    mc_mask   = (df_mc["y"]   > y_min) & (df_mc["y"]   < y_max)
    data_mask = (df_data["y"] > y_min) & (df_data["y"] < y_max)
    return df_mc[mc_mask].reset_index(drop=True), df_data[data_mask].reset_index(drop=True)


def _apply_mx_cut(df_mc, df_data, mx_min, mx_max, species_alias):
    """Apply species-appropriate missing-mass cut to both DataFrames.

    The column used depends on the species (hadron-mass hypothesis):

      kp  → Mx_eKX   (K+ hypothesis)
      pip → Mx_epiX  (π+ hypothesis)
      p   → Mx_epX   (proton hypothesis)

    For species aliases em, pim, kn the missing-mass cut is not defined
    (no SIDIS-relevant exclusive channel); call sites must guard against
    these aliases and use ``--no-mx-cut`` or skip ``--sidis-cuts`` Mx
    application rather than calling this function.

    Parameters
    ----------
    df_mc, df_data : pd.DataFrame
        Species- (and truth-match-) filtered DataFrames.
    mx_min, mx_max : float
        Exclusive bounds (GeV): keep rows where mx_min < Mx_col < mx_max.
        Use float('inf') as mx_max for a one-sided lower cut.
    species_alias : str
        One of the keys in SPECIES_MAP.

    Returns
    -------
    (df_mc_cut, df_data_cut) : tuple of pd.DataFrame

    Raises
    ------
    ValueError
        If species_alias is not one of {kp, pip, p}.
    SystemExit
        If the appropriate Mx column is absent from either DataFrame.
        This happens with old ROOT files produced before the Mx columns
        were added to the groovy in 2026-06.
    """
    if species_alias not in _MX_SPECIES:
        raise ValueError(
            f"Mx cut undefined for species: {species_alias!r}.  "
            f"The missing-mass cut is only defined for kp (Mx_eKX), "
            f"pip (Mx_epiX), and p (Mx_epX).  For species em/pim/kn, "
            f"pass --no-mx-cut or omit --mx-cut / --sidis-cuts."
        )
    col = _MX_SPECIES[species_alias]
    missing = []
    if col not in df_mc.columns:
        missing.append("MC")
    if col not in df_data.columns:
        missing.append("Data")
    if missing:
        sys.exit(
            f"\nERROR: '{col}' column missing from: {', '.join(missing)} DataFrame(s).\n"
            f"  The Mx columns were added to the groovy in 2026-06; re-process your\n"
            f"  ROOT files with the current groovy, or pass --no-mx-cut to skip this filter."
        )
    mc_mask   = (df_mc[col]   > mx_min) & (df_mc[col]   < mx_max)
    data_mask = (df_data[col] > mx_min) & (df_data[col] < mx_max)
    return df_mc[mc_mask].reset_index(drop=True), df_data[data_mask].reset_index(drop=True)


def _write_audit_readme(outdir, pid, label, truth_mode, mc_path, data_path, variables,
                        vz_cut_line="Vertex-z cut: disabled",
                        extra_cut_lines=None):
    """Write a provenance README (≤30 lines) into outdir."""
    now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mc_sel = {
        "matched": f"(pid == {pid}) & (mc_matching_pid != {SENTINEL_LOW})",
        "pure":    f"(pid == {pid}) & (mc_matching_pid == {pid})",
        "off":     f"pid == {pid}",
    }[truth_mode]
    extra_block = ""
    if extra_cut_lines:
        extra_block = "".join(f"  {line}\n" for line in extra_cut_lines)
    content = (
        f"# Feature audit — species {label} (pid={pid})\n\n"
        f"Generated: {now}\n\n"
        f"## Selections applied\n"
        f"  MC   : {mc_sel}\n"
        f"  Data : pid == {pid}\n"
        f"  Truth mode: {truth_mode}\n"
        f"  {vz_cut_line}\n"
        f"{extra_block}"
        f"\n## Input files\n"
        f"  MC   : {mc_path}\n"
        f"  Data : {data_path}\n\n"
        f"## Variables audited\n"
        f"  {', '.join(variables)}\n\n"
        f"## Column meanings\n"
        f"  See figures/feature_audit/COLUMNS.md for the per-column glossary.\n"
        f"  See scripts/README.md (compare_mc_data.py Output section) for full metric definitions.\n"
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
    # ── Vertex-z cut ────────────────────────────────────────────────────────
    p.add_argument("--vz-cut", nargs=2, type=float, metavar=("MIN", "MAX"),
                   default=[-8.0, 2.0],
                   help="Vertex-z cut window in cm (exclusive bounds). Default: -8 2")
    p.add_argument("--no-vz-cut", action="store_true",
                   help="Disable the vertex-z cut entirely.")
    # ── Event-level kinematic cuts ───────────────────────────────────────────
    p.add_argument("--q2-cut", nargs=2, type=float, metavar=("MIN", "MAX"),
                   default=None,
                   help="Q² cut (GeV²), exclusive bounds. Only applied if flag is passed "
                        "or --sidis-cuts is set.")
    p.add_argument("--no-q2-cut", action="store_true",
                   help="Suppress Q² cut even if --sidis-cuts is set.")
    p.add_argument("--w-cut", nargs=2, type=float, metavar=("MIN", "MAX"),
                   default=None,
                   help="W cut (GeV), exclusive bounds. Only applied if flag is passed "
                        "or --sidis-cuts is set.")
    p.add_argument("--no-w-cut", action="store_true",
                   help="Suppress W cut even if --sidis-cuts is set.")
    p.add_argument("--y-cut", nargs=2, type=float, metavar=("MIN", "MAX"),
                   default=None,
                   help="Inelasticity y cut (dimensionless), exclusive bounds. Only applied "
                        "if flag is passed or --sidis-cuts is set.")
    p.add_argument("--no-y-cut", action="store_true",
                   help="Suppress y cut even if --sidis-cuts is set.")
    p.add_argument("--mx-cut", nargs=2, type=float, metavar=("MIN", "MAX"),
                   default=None,
                   help="Species-appropriate missing-mass cut (GeV). Column selected "
                        "by species: kp→Mx_eKX, pip→Mx_epiX, p→Mx_epX. Only applied "
                        "if flag is passed or --sidis-cuts is set.")
    p.add_argument("--no-mx-cut", action="store_true",
                   help="Suppress Mx cut even if --sidis-cuts is set.")
    # ── SIDIS convenience bundle ─────────────────────────────────────────────
    p.add_argument("--sidis-cuts", action="store_true",
                   help=(
                       "Enable all SIDIS event-level cuts with canonical defaults: "
                       "Q² > 2 GeV², W > 2 GeV, y < 0.75, and species-appropriate "
                       "Mx lower bound (kp→1.6, pip→1.5, p→1.0 GeV). "
                       "Individual --no-X flags suppress specific cuts. "
                       "Species em/pim/kn have no Mx cut even with --sidis-cuts. "
                       "Output directory is auto-suffixed with '_sidis' unless "
                       "--outdir is explicitly provided."
                   ))
    # ── Column pruning ───────────────────────────────────────────────────────
    p.add_argument("--load-all-cols", action="store_true",
                   help=("Load every branch from the ROOT file.  The default "
                         "loads only the columns the audit needs (audit "
                         "variables + species selector + cut columns), which "
                         "is much faster on production-scale files.  Use this "
                         "for debugging or when you want to inspect columns "
                         "beyond the audit set."))
    return p


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

    spec    = SPECIES_MAP[args.species]
    pid     = spec["pid"]
    label   = spec["label"]

    # ── Resolve output directory ───────────────────────────────────────────────
    user_set_outdir = args.outdir is not None
    if user_set_outdir:
        outdir = args.outdir
    else:
        base = os.path.join("figures", "feature_audit", args.species)
        if args.sidis_cuts:
            outdir = base + "_sidis"
        else:
            outdir = base

    lmc     = args.label_mc   or f"MC {label}"
    ldata   = args.label_data or f"Data {label}"

    mc_sel_str = {
        "matched": f"(pid=={pid}) & (mc_matching_pid!={SENTINEL_LOW})",
        "pure":    f"(pid=={pid}) & (mc_matching_pid=={pid})",
        "off":     f"pid=={pid}",
    }[args.truth_mode]
    data_sel_str = f"pid=={pid}"

    variables = _resolve_variables(args.vars)

    # ── Resolve vz-cut settings ────────────────────────────────────────────────
    vz_cut_enabled = not args.no_vz_cut
    vz_min, vz_max = args.vz_cut   # floats from argparse nargs=2
    if args.no_vz_cut and args.vz_cut != [-8.0, 2.0]:
        # User supplied both --no-vz-cut and --vz-cut; --no-vz-cut wins.
        print("WARNING: both --no-vz-cut and --vz-cut were supplied; "
              "--no-vz-cut wins and the vz cut is disabled.")

    if vz_cut_enabled:
        vz_cut_line = f"Vertex-z cut: {vz_min} < vz < {vz_max} cm"
        vz_preamble = f"  Vertex-z cut: {vz_min} < vz < {vz_max} cm"
    else:
        vz_cut_line = "Vertex-z cut: disabled"
        vz_preamble = "  Vertex-z cut: disabled"

    # ── Resolve event-level kinematic cut settings ─────────────────────────────
    # Priority: --no-X always wins. Otherwise --X takes effect if supplied.
    # Otherwise --sidis-cuts enables a default. Otherwise the cut is off.

    sidis_defaults = {
        "q2": _SIDIS_Q2,
        "w":  _SIDIS_W,
        "y":  _SIDIS_Y,
    }

    def _resolve_cut(name, user_val, no_flag, default_pair):
        """Return (enabled, min, max) for a named cut.

        Parameters
        ----------
        name : str        — human name for warnings
        user_val : list or None  — parsed value of --X-cut (None if not passed)
        no_flag : bool    — True if --no-X-cut was passed
        default_pair : tuple or None — (min, max) to use when --sidis-cuts active

        Returns (enabled, lo, hi) where enabled is bool and lo/hi are floats.
        """
        if no_flag:
            if user_val is not None:
                print(f"WARNING: both --{name}-cut and --no-{name}-cut were supplied; "
                      f"--no-{name}-cut wins and the {name} cut is disabled.")
            return False, None, None
        if user_val is not None:
            return True, float(user_val[0]), float(user_val[1])
        if args.sidis_cuts and default_pair is not None:
            return True, float(default_pair[0]), float(default_pair[1])
        return False, None, None

    q2_on, q2_lo, q2_hi = _resolve_cut(
        "q2", args.q2_cut, args.no_q2_cut, sidis_defaults["q2"])
    w_on,  w_lo,  w_hi  = _resolve_cut(
        "w",  args.w_cut,  args.no_w_cut,  sidis_defaults["w"])
    y_on,  y_lo,  y_hi  = _resolve_cut(
        "y",  args.y_cut,  args.no_y_cut,  sidis_defaults["y"])

    # Mx cut: species-aware; em/pim/kn have no SIDIS Mx default.
    mx_on  = False
    mx_lo  = None
    mx_hi  = None
    if args.no_mx_cut:
        if args.mx_cut is not None:
            print("WARNING: both --mx-cut and --no-mx-cut were supplied; "
                  "--no-mx-cut wins and the Mx cut is disabled.")
        mx_on = False
    elif args.mx_cut is not None:
        if args.species not in _MX_SPECIES:
            sys.exit(
                f"\nERROR: --mx-cut is not defined for species '{args.species}'.  "
                f"The missing-mass cut is only defined for kp, pip, and p."
            )
        mx_on = True
        mx_lo = float(args.mx_cut[0])
        mx_hi = float(args.mx_cut[1])
    elif args.sidis_cuts and args.species in _SIDIS_MX_DEFAULTS:
        mx_on = True
        mx_lo = float(_SIDIS_MX_DEFAULTS[args.species])
        mx_hi = float("inf")
    # else: mx_on stays False (em/pim/kn with --sidis-cuts, or no flag at all)

    # ── Preamble ───────────────────────────────────────────────────────────────
    sidis_header = "  SIDIS cuts: enabled (--sidis-cuts)\n" if args.sidis_cuts else ""

    def _fmt_cut(name, on, lo, hi, no_flag, unit=""):
        if no_flag:
            return f"  {name} cut: disabled (--no-{name.lower().replace(' ', '-')}-cut)"
        if on:
            hi_str = "inf" if hi == float("inf") else str(hi)
            return f"  {name} cut: {lo} < {name} < {hi_str}{unit}"
        return f"  {name} cut: not applied"

    q2_preamble  = _fmt_cut("Q2", q2_on, q2_lo, q2_hi, args.no_q2_cut, " GeV²")
    w_preamble   = _fmt_cut("W",  w_on,  w_lo,  w_hi,  args.no_w_cut,  " GeV")
    y_preamble   = _fmt_cut("y",  y_on,  y_lo,  y_hi,  args.no_y_cut)
    if args.no_mx_cut:
        mx_preamble = "  Mx cut: disabled (--no-mx-cut)"
    elif mx_on:
        mx_col   = _MX_SPECIES.get(args.species, "N/A")
        hi_str   = "inf" if mx_hi == float("inf") else str(mx_hi)
        mx_preamble = f"  Mx cut: {mx_lo} < {mx_col} < {hi_str} GeV"
    else:
        mx_preamble = "  Mx cut: not applied"

    # ── Compute needed column sets (column pruning) ────────────────────────────
    # Loading the full ntuple (all 57 MC columns, all 54 data columns) is slow
    # on production-scale ROOT files (~425–575 MB).  We only need: audit
    # variables + species selector + (p, theta) for cell slicing + columns
    # used by the enabled cuts.  Pass --load-all-cols to disable this and
    # load everything (matches the pre-pruning default behaviour).

    # Columns always required by the audit machinery regardless of flags
    required_cols = {"pid", "p", "theta"}

    # Truth-mode columns: mc_matching_pid is only present in MC
    if args.truth_mode in ("matched", "pure"):
        mc_only_required = {"mc_matching_pid"}
    else:
        mc_only_required = set()

    # Columns required by enabled cuts
    if vz_cut_enabled:
        required_cols.add("vz")
    if q2_on:
        required_cols.add("Q2")
    if w_on:
        required_cols.add("W")
    if y_on:
        required_cols.add("y")
    if mx_on:
        required_cols.add(_MX_SPECIES[args.species])  # e.g. "Mx_eKX"

    if getattr(args, "load_all_cols", False):
        mc_cols   = None
        data_cols = None
    else:
        mc_cols   = sorted(set(variables) | required_cols | mc_only_required)
        data_cols = sorted(set(variables) | required_cols)

    # ── Preamble ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"audit_species.py  —  {label} (pid={pid})")
    print(f"{'='*65}")
    print(f"  Species     : {args.species} → {label} (pid={pid})")
    print(f"  Truth mode  : {args.truth_mode}")
    print(f"  MC selection: {mc_sel_str}")
    print(f"  Data select : {data_sel_str}")
    print(vz_preamble)
    if args.sidis_cuts:
        print(f"  SIDIS cuts  : enabled (--sidis-cuts)")
    print(q2_preamble)
    print(w_preamble)
    print(y_preamble)
    print(mx_preamble)
    print(f"  Variables   : {variables}")
    print(f"  Output dir  : {outdir}")
    print(f"  Bins        : {args.bins}   Normalize: {not args.no_normalize}")
    if mc_cols is None:
        print(f"  Columns      : all (--load-all-cols set)")
    else:
        n_audit = len(variables)
        n_sel   = len(mc_cols) - len([c for c in mc_cols if c in variables])
        print(f"  Columns      : {n_audit} audit + {n_sel} selection/cut "
              f"({len(mc_cols)} MC cols requested; --load-all-cols to override)")

    # ── Load files ─────────────────────────────────────────────────────────────
    print(f"\nLoading MC:   {args.mc}")
    df_mc_raw   = _load_file(args.mc,   max_rows=args.max_rows, branches=mc_cols)
    print(f"Loading Data: {args.data}")
    df_data_raw = _load_file(args.data, max_rows=args.max_rows, branches=data_cols)

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

    # ── Apply vz cut ───────────────────────────────────────────────────────────
    if vz_cut_enabled:
        df_mc, df_data = _apply_vz_cut(df_mc, df_data, vz_min, vz_max)
        print(f"\n  MC after species + truth-match + vz cut : {len(df_mc):,}")
        print(f"  Data after species + vz cut             : {len(df_data):,}")
    else:
        print(f"\n  MC after species + truth-match selection : {len(df_mc):,}")
        print(f"  Data after species selection             : {len(df_data):,}")

    # ── Apply event-level kinematic cuts (in order: Q², W, y, Mx) ─────────────
    if q2_on:
        df_mc, df_data = _apply_q2_cut(df_mc, df_data, q2_lo, q2_hi)
        hi_str = "inf" if q2_hi == float("inf") else str(q2_hi)
        print(f"  MC after Q² cut ({q2_lo} < Q2 < {hi_str})   : {len(df_mc):,}")
        print(f"  Data after Q² cut                       : {len(df_data):,}")

    if w_on:
        df_mc, df_data = _apply_w_cut(df_mc, df_data, w_lo, w_hi)
        hi_str = "inf" if w_hi == float("inf") else str(w_hi)
        print(f"  MC after W cut ({w_lo} < W < {hi_str})     : {len(df_mc):,}")
        print(f"  Data after W cut                        : {len(df_data):,}")

    if y_on:
        df_mc, df_data = _apply_y_cut(df_mc, df_data, y_lo, y_hi)
        print(f"  MC after y cut ({y_lo} < y < {y_hi})      : {len(df_mc):,}")
        print(f"  Data after y cut                        : {len(df_data):,}")

    if mx_on:
        df_mc, df_data = _apply_mx_cut(df_mc, df_data, mx_lo, mx_hi, args.species)
        mx_col  = _MX_SPECIES[args.species]
        hi_str  = "inf" if mx_hi == float("inf") else str(mx_hi)
        print(f"  MC after Mx cut ({mx_lo} < {mx_col} < {hi_str}): {len(df_mc):,}")
        print(f"  Data after Mx cut                       : {len(df_data):,}")

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
    extra_cut_lines = []
    if args.sidis_cuts:
        extra_cut_lines.append("SIDIS cuts: enabled (--sidis-cuts)")
    extra_cut_lines.append(q2_preamble.strip())
    extra_cut_lines.append(w_preamble.strip())
    extra_cut_lines.append(y_preamble.strip())
    extra_cut_lines.append(mx_preamble.strip())

    _write_audit_readme(outdir, pid, label, args.truth_mode,
                        args.mc, args.data, variables,
                        vz_cut_line=vz_cut_line,
                        extra_cut_lines=extra_cut_lines)

    # ── Copy COLUMNS.md into the species output directory ──────────────────────
    columns_src = os.path.join("figures", "feature_audit", "COLUMNS.md")
    if os.path.isfile(columns_src):
        shutil.copy(columns_src, os.path.join(outdir, "COLUMNS.md"))
        print(f"  COLUMNS.md copied to: {os.path.join(outdir, 'COLUMNS.md')}")

    # ── Per-variable summary table ─────────────────────────────────────────────
    print_summary_table(summary, variables)

    # ── Cuts-applied summary line ──────────────────────────────────────────────
    def _fmt_range(on, lo, hi):
        if not on:
            return "off"
        hi_str = "inf" if hi == float("inf") else str(hi)
        return f"[{lo}, {hi_str}]"

    if vz_cut_enabled:
        vz_range = f"[{vz_min}, {vz_max}]"
    else:
        vz_range = "off"

    mx_col_name = _MX_SPECIES.get(args.species, "Mx")
    print(f"\nCuts applied: "
          f"Q2 {_fmt_range(q2_on, q2_lo, q2_hi)}, "
          f"W {_fmt_range(w_on, w_lo, w_hi)}, "
          f"y {_fmt_range(y_on, y_lo, y_hi)}, "
          f"{mx_col_name} {_fmt_range(mx_on, mx_lo, mx_hi)}, "
          f"vz {vz_range}")

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
        "chi2pid"         : rng.normal(-10, 10, N),
        "vz"              : rng.uniform(-15.0, 10.0, N),
        "Q2"              : rng.uniform(0.5, 8.0, N),
        "W"               : rng.uniform(1.5, 4.5, N),
        "y"               : rng.uniform(0.05, 0.95, N),
        "Mx_eKX"          : rng.uniform(0.8, 3.5, N),
        "Mx_epiX"         : rng.uniform(0.8, 3.5, N),
        "Mx_epX"          : rng.uniform(0.8, 3.5, N),
    })

    pids_data = rng.choice([321, 211, 2212], size=N, p=[0.50, 0.30, 0.20])
    df_data_synth = pd.DataFrame({
        "pid"    : pids_data,
        "p"      : rng.uniform(1.0, 5.0, N),
        "theta"  : rng.uniform(5.0, 35.0, N),
        "beta"   : rng.uniform(0.0, 1.0, N),
        "chi2pid": rng.normal(-10, 10, N),
        "vz"     : rng.uniform(-15.0, 10.0, N),
        "Q2"     : rng.uniform(0.5, 8.0, N),
        "W"      : rng.uniform(1.5, 4.5, N),
        "y"      : rng.uniform(0.05, 0.95, N),
        "Mx_eKX" : rng.uniform(0.8, 3.5, N),
        "Mx_epiX": rng.uniform(0.8, 3.5, N),
        "Mx_epX" : rng.uniform(0.8, 3.5, N),
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

    # ── 6. vz cut: rows outside -8 < vz < 2 are removed ──────────────────────
    # Use the species-filtered (matched) and data DataFrames as inputs.
    df_mc_flt   = _filter_mc(df_mc_synth, pid_kp, "matched")
    df_data_flt2 = _filter_data(df_data_synth, pid_kp)
    vz_min_t, vz_max_t = -8.0, 2.0
    mc_cut, data_cut = _apply_vz_cut(df_mc_flt, df_data_flt2, vz_min_t, vz_max_t)
    expected_mc_cut   = int(((df_mc_flt["vz"]    > vz_min_t) & (df_mc_flt["vz"]    < vz_max_t)).sum())
    expected_data_cut = int(((df_data_flt2["vz"] > vz_min_t) & (df_data_flt2["vz"] < vz_max_t)).sum())
    assert len(mc_cut)   == expected_mc_cut, \
        f"FAIL vz cut MC: expected {expected_mc_cut}, got {len(mc_cut)}"
    assert len(data_cut) == expected_data_cut, \
        f"FAIL vz cut Data: expected {expected_data_cut}, got {len(data_cut)}"
    assert len(mc_cut)   < len(df_mc_flt),    "FAIL vz cut should remove MC rows"
    assert len(data_cut) < len(df_data_flt2), "FAIL vz cut should remove data rows"
    print(f"  vz cut [-8,2] MC:   {len(df_mc_flt)} → {len(mc_cut)} rows ✓")
    print(f"  vz cut [-8,2] Data: {len(df_data_flt2)} → {len(data_cut)} rows ✓")

    # ── 7. vz cut disabled: all rows survive ──────────────────────────────────
    # Simulate --no-vz-cut by not calling _apply_vz_cut.
    # We verify this by passing a window that covers the full vz range; the
    # internal helper is called with the same inputs and result must be identical.
    mc_full, data_full = _apply_vz_cut(df_mc_flt, df_data_flt2, -1000.0, 1000.0)
    assert len(mc_full)   == len(df_mc_flt),    "FAIL wide vz window should keep all MC rows"
    assert len(data_full) == len(df_data_flt2), "FAIL wide vz window should keep all data rows"
    print("  vz cut disabled (wide window): all rows survive ✓")

    # ── 8. Missing vz column triggers sys.exit ─────────────────────────────────
    df_no_vz_mc   = df_mc_flt.drop(columns=["vz"])
    df_no_vz_data = df_data_flt2.drop(columns=["vz"])
    try:
        _apply_vz_cut(df_no_vz_mc, df_data_flt2, -8.0, 2.0)
        assert False, "FAIL should have raised SystemExit for missing vz in MC"
    except SystemExit as e:
        assert "MC" in str(e), f"FAIL error message should name MC: {e}"
        print("  Missing vz in MC → SystemExit with 'MC' in message ✓")
    try:
        _apply_vz_cut(df_mc_flt, df_no_vz_data, -8.0, 2.0)
        assert False, "FAIL should have raised SystemExit for missing vz in Data"
    except SystemExit as e:
        assert "Data" in str(e), f"FAIL error message should name Data: {e}"
        print("  Missing vz in Data → SystemExit with 'Data' in message ✓")
    try:
        _apply_vz_cut(df_no_vz_mc, df_no_vz_data, -8.0, 2.0)
        assert False, "FAIL should have raised SystemExit for missing vz in both"
    except SystemExit as e:
        msg = str(e)
        assert "MC" in msg and "Data" in msg, \
            f"FAIL error message should name both MC and Data: {e}"
        print("  Missing vz in both → SystemExit naming both ✓")

    # ── 9. Q² cut ─────────────────────────────────────────────────────────────
    q2_lo_t, q2_hi_t = 2.0, 6.0
    mc_q2, data_q2 = _apply_q2_cut(df_mc_flt, df_data_flt2, q2_lo_t, q2_hi_t)
    exp_mc_q2   = int(((df_mc_flt["Q2"]    > q2_lo_t) & (df_mc_flt["Q2"]    < q2_hi_t)).sum())
    exp_data_q2 = int(((df_data_flt2["Q2"] > q2_lo_t) & (df_data_flt2["Q2"] < q2_hi_t)).sum())
    assert len(mc_q2)   == exp_mc_q2,   f"FAIL Q2 cut MC: expected {exp_mc_q2}, got {len(mc_q2)}"
    assert len(data_q2) == exp_data_q2, f"FAIL Q2 cut Data: expected {exp_data_q2}, got {len(data_q2)}"
    assert len(mc_q2)   < len(df_mc_flt),    "FAIL Q2 cut should remove MC rows"
    assert len(data_q2) < len(df_data_flt2), "FAIL Q2 cut should remove Data rows"
    print(f"  Q2 cut [{q2_lo_t},{q2_hi_t}] MC:   {len(df_mc_flt)} → {len(mc_q2)} rows ✓")
    print(f"  Q2 cut [{q2_lo_t},{q2_hi_t}] Data: {len(df_data_flt2)} → {len(data_q2)} rows ✓")

    # Missing Q2 column triggers sys.exit
    df_no_q2_mc = df_mc_flt.drop(columns=["Q2"])
    try:
        _apply_q2_cut(df_no_q2_mc, df_data_flt2, 2.0, float("inf"))
        assert False, "FAIL should have raised SystemExit for missing Q2 in MC"
    except SystemExit as e:
        assert "MC" in str(e), f"FAIL Q2 error message should name MC: {e}"
        assert "--no-q2-cut" in str(e), f"FAIL Q2 error message should mention --no-q2-cut: {e}"
        print("  Missing Q2 in MC → SystemExit with 'MC' and '--no-q2-cut' in message ✓")

    # ── 10. W cut ─────────────────────────────────────────────────────────────
    w_lo_t, w_hi_t = 2.0, 4.0
    mc_w, data_w = _apply_w_cut(df_mc_flt, df_data_flt2, w_lo_t, w_hi_t)
    exp_mc_w   = int(((df_mc_flt["W"]    > w_lo_t) & (df_mc_flt["W"]    < w_hi_t)).sum())
    exp_data_w = int(((df_data_flt2["W"] > w_lo_t) & (df_data_flt2["W"] < w_hi_t)).sum())
    assert len(mc_w)   == exp_mc_w,   f"FAIL W cut MC: expected {exp_mc_w}, got {len(mc_w)}"
    assert len(data_w) == exp_data_w, f"FAIL W cut Data: expected {exp_data_w}, got {len(data_w)}"
    assert len(mc_w)   < len(df_mc_flt),    "FAIL W cut should remove MC rows"
    assert len(data_w) < len(df_data_flt2), "FAIL W cut should remove Data rows"
    print(f"  W cut [{w_lo_t},{w_hi_t}] MC:   {len(df_mc_flt)} → {len(mc_w)} rows ✓")
    print(f"  W cut [{w_lo_t},{w_hi_t}] Data: {len(df_data_flt2)} → {len(data_w)} rows ✓")

    # Missing W column triggers sys.exit
    df_no_w_mc = df_mc_flt.drop(columns=["W"])
    try:
        _apply_w_cut(df_no_w_mc, df_data_flt2, 2.0, float("inf"))
        assert False, "FAIL should have raised SystemExit for missing W in MC"
    except SystemExit as e:
        assert "MC" in str(e), f"FAIL W error message should name MC: {e}"
        assert "--no-w-cut" in str(e), f"FAIL W error message should mention --no-w-cut: {e}"
        print("  Missing W in MC → SystemExit with 'MC' and '--no-w-cut' in message ✓")

    # ── 11. y cut ─────────────────────────────────────────────────────────────
    y_lo_t, y_hi_t = 0.0, 0.75
    mc_y, data_y = _apply_y_cut(df_mc_flt, df_data_flt2, y_lo_t, y_hi_t)
    exp_mc_y   = int(((df_mc_flt["y"]    > y_lo_t) & (df_mc_flt["y"]    < y_hi_t)).sum())
    exp_data_y = int(((df_data_flt2["y"] > y_lo_t) & (df_data_flt2["y"] < y_hi_t)).sum())
    assert len(mc_y)   == exp_mc_y,   f"FAIL y cut MC: expected {exp_mc_y}, got {len(mc_y)}"
    assert len(data_y) == exp_data_y, f"FAIL y cut Data: expected {exp_data_y}, got {len(data_y)}"
    assert len(mc_y)   < len(df_mc_flt),    "FAIL y cut should remove MC rows"
    assert len(data_y) < len(df_data_flt2), "FAIL y cut should remove Data rows"
    print(f"  y cut [{y_lo_t},{y_hi_t}] MC:   {len(df_mc_flt)} → {len(mc_y)} rows ✓")
    print(f"  y cut [{y_lo_t},{y_hi_t}] Data: {len(df_data_flt2)} → {len(data_y)} rows ✓")

    # Missing y column triggers sys.exit
    df_no_y_mc = df_mc_flt.drop(columns=["y"])
    try:
        _apply_y_cut(df_no_y_mc, df_data_flt2, 0.0, 0.75)
        assert False, "FAIL should have raised SystemExit for missing y in MC"
    except SystemExit as e:
        assert "MC" in str(e), f"FAIL y error message should name MC: {e}"
        assert "--no-y-cut" in str(e), f"FAIL y error message should mention --no-y-cut: {e}"
        print("  Missing y in MC → SystemExit with 'MC' and '--no-y-cut' in message ✓")

    # ── 12. Mx cut — kp: uses Mx_eKX column ───────────────────────────────────
    mx_lo_t, mx_hi_t = 1.6, float("inf")
    mc_mx, data_mx = _apply_mx_cut(df_mc_flt, df_data_flt2, mx_lo_t, mx_hi_t, "kp")
    exp_mc_mx   = int((df_mc_flt["Mx_eKX"]    > mx_lo_t).sum())   # hi is inf
    exp_data_mx = int((df_data_flt2["Mx_eKX"] > mx_lo_t).sum())
    assert len(mc_mx)   == exp_mc_mx,   f"FAIL Mx(kp) cut MC: expected {exp_mc_mx}, got {len(mc_mx)}"
    assert len(data_mx) == exp_data_mx, f"FAIL Mx(kp) cut Data: expected {exp_data_mx}, got {len(data_mx)}"
    assert len(mc_mx)   < len(df_mc_flt),    "FAIL Mx cut should remove MC rows"
    assert len(data_mx) < len(df_data_flt2), "FAIL Mx cut should remove Data rows"
    print(f"  Mx(kp) cut [1.6,inf] MC(Mx_eKX):  {len(df_mc_flt)} → {len(mc_mx)} rows ✓")
    print(f"  Mx(kp) cut [1.6,inf] Data(Mx_eKX):{len(df_data_flt2)} → {len(data_mx)} rows ✓")

    # ── 13. Mx cut — pip: uses Mx_epiX column ─────────────────────────────────
    mc_mx_pip, data_mx_pip = _apply_mx_cut(df_mc_flt, df_data_flt2, 1.5, float("inf"), "pip")
    exp_mc_pip   = int((df_mc_flt["Mx_epiX"]    > 1.5).sum())
    exp_data_pip = int((df_data_flt2["Mx_epiX"] > 1.5).sum())
    assert len(mc_mx_pip)   == exp_mc_pip,   f"FAIL Mx(pip) cut MC: expected {exp_mc_pip}"
    assert len(data_mx_pip) == exp_data_pip, f"FAIL Mx(pip) cut Data: expected {exp_data_pip}"
    print(f"  Mx(pip) cut [1.5,inf] uses Mx_epiX column ✓")

    # ── 14. Mx cut — em: raises ValueError ────────────────────────────────────
    try:
        _apply_mx_cut(df_mc_flt, df_data_flt2, 0.0, float("inf"), "em")
        assert False, "FAIL should have raised ValueError for species em"
    except ValueError as e:
        assert "em" in str(e), f"FAIL ValueError should name species: {e}"
        print("  _apply_mx_cut('em') → ValueError (no Mx defined for electrons) ✓")

    # ── 15. Mx cut — missing column triggers sys.exit with groovy message ──────
    df_no_mx_mc = df_mc_flt.drop(columns=["Mx_eKX"])
    try:
        _apply_mx_cut(df_no_mx_mc, df_data_flt2, 1.6, float("inf"), "kp")
        assert False, "FAIL should have raised SystemExit for missing Mx_eKX in MC"
    except SystemExit as e:
        msg = str(e)
        assert "MC" in msg,          f"FAIL Mx error should name MC: {e}"
        assert "groovy" in msg,      f"FAIL Mx error should mention groovy: {e}"
        assert "2026-06" in msg,     f"FAIL Mx error should mention 2026-06: {e}"
        assert "--no-mx-cut" in msg, f"FAIL Mx error should mention --no-mx-cut: {e}"
        print("  Missing Mx_eKX in MC → SystemExit with groovy-provenance message ✓")

    # ── 16. --sidis-cuts resolution logic ─────────────────────────────────────
    # Simulate what main() does for species kp with --sidis-cuts and no overrides.
    # Expected: Q²=2-inf, W=2-inf, y=0-0.75, Mx=1.6-inf.
    class _FakeArgs:
        sidis_cuts = True
        q2_cut = None;  no_q2_cut = False
        w_cut  = None;  no_w_cut  = False
        y_cut  = None;  no_y_cut  = False
        mx_cut = None;  no_mx_cut = False
        species = "kp"

    fa = _FakeArgs()

    def _resolve_cut_test(name, user_val, no_flag, default_pair):
        if no_flag:
            return False, None, None
        if user_val is not None:
            return True, float(user_val[0]), float(user_val[1])
        if fa.sidis_cuts and default_pair is not None:
            return True, float(default_pair[0]), float(default_pair[1])
        return False, None, None

    q2_on_t, q2_lo_t2, q2_hi_t2 = _resolve_cut_test("q2", fa.q2_cut, fa.no_q2_cut, _SIDIS_Q2)
    w_on_t,  w_lo_t2,  w_hi_t2  = _resolve_cut_test("w",  fa.w_cut,  fa.no_w_cut,  _SIDIS_W)
    y_on_t,  y_lo_t2,  y_hi_t2  = _resolve_cut_test("y",  fa.y_cut,  fa.no_y_cut,  _SIDIS_Y)

    assert q2_on_t and q2_lo_t2 == 2.0 and q2_hi_t2 == float("inf"), \
        f"FAIL sidis kp Q2 resolution: {q2_on_t}, {q2_lo_t2}, {q2_hi_t2}"
    assert w_on_t  and w_lo_t2  == 2.0 and w_hi_t2  == float("inf"), \
        f"FAIL sidis kp W resolution: {w_on_t}, {w_lo_t2}, {w_hi_t2}"
    assert y_on_t  and y_lo_t2  == 0.0 and y_hi_t2  == 0.75, \
        f"FAIL sidis kp y resolution: {y_on_t}, {y_lo_t2}, {y_hi_t2}"
    # Mx for kp
    mx_on_t = (not fa.no_mx_cut) and (fa.mx_cut is None) and fa.sidis_cuts and (fa.species in _SIDIS_MX_DEFAULTS)
    mx_lo_t2 = float(_SIDIS_MX_DEFAULTS["kp"]) if mx_on_t else None
    mx_hi_t2 = float("inf") if mx_on_t else None
    assert mx_on_t and mx_lo_t2 == 1.6 and mx_hi_t2 == float("inf"), \
        f"FAIL sidis kp Mx resolution: {mx_on_t}, {mx_lo_t2}, {mx_hi_t2}"
    print("  --sidis-cuts kp: Q2=[2,inf], W=[2,inf], y=[0,0.75], Mx=[1.6,inf] ✓")

    # Simulate for species em with --sidis-cuts: no Mx cut.
    fa_em = _FakeArgs()
    fa_em.species = "em"
    mx_on_em = (not fa_em.no_mx_cut) and (fa_em.mx_cut is None) and \
               fa_em.sidis_cuts and (fa_em.species in _SIDIS_MX_DEFAULTS)
    assert not mx_on_em, "FAIL sidis em should have no Mx cut"
    print("  --sidis-cuts em: no Mx cut (em not in _SIDIS_MX_DEFAULTS) ✓")

    print("\naudit_species.py self-test passed.")
