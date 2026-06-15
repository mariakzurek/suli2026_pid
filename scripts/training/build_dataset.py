"""
build_dataset.py — Convert MC ROOT files → train/val/test parquet trio + manifest.

WHAT IT DOES
------------
Reads the three file-level split lists produced by Cooper's Phase-0 task
(slurm/train_files.txt, slurm/val_files.txt, slurm/test_files.txt), loads
the corresponding MC ROOT files, applies EB-K+ selection and momentum cap,
writes three parquet files (train.parquet, val.parquet, test.parquet) and a
JSON manifest.

Dataset contract (fixed column order; downstream selects by name):

  Columns always present:
    p         float32   track momentum (GeV/c)
    theta     float32   polar angle (deg)
    phi       float32   azimuthal angle (deg)
    vz        float32   vertex z (cm)
    sector    int32     CLAS12 sector (1–6)
    chi2pid   float32   required by evaluate.py baseline cut

  Feature columns (from --features-file, in file order):
    <name>    float32   NaN where ntuple sentinel -9999

  Metadata columns always present:
    pid               int32    always 321 (EB K+)
    mc_matching_pid   int32    -9999 if unmatched (preserved as-is)

  Label column:
    label   int8   (train/val)   1 if mc_matching_pid==321, 0 if ==211
            Int8   (test)        nullable — NA for protons/unmatched/other

Selection:
  All splits: pid == 321 (EB K+), p < p_max
  train / val: mc_matching_pid in {211, 321}  → non-nullable int8 label
  test:        all EB-K+ rows                 → nullable Int8 label

WHEN TO USE
-----------
Run once before training, after the Week-3 audit KEEP list is finalised.
Re-run with --overwrite whenever --features-file or --p-max changes.
Use --max-files N for smoke tests; production runs use all files.

PITFALLS
--------
* --features-file must be non-empty (pointing at the audit KEEP list); the
  script errors rather than building an empty feature set.
* Missing files > 5% of the split cause a hard failure unless
  --allow-missing-files is set.  Cooper cross-checks overlaps with:
    diff <(sort -u slurm/train_files.txt) \\
         <(ls /volatile/clas12/$USER/SULI/mc_v01/ | sed 's/.root$//' | sort -u)
* ROOT loading uses uproot with library="np" and filter_name= (the numpy
  fast-path from compare_mc_data._load_file).  Do not switch to library="pd"
  or expressions= — measured ~30x slower on production-scale ROOT files.
* Sentinel value -9999 is replaced with NaN in feature columns only; it is
  preserved as-is in pid and mc_matching_pid.

Usage:
  python scripts/training/build_dataset.py \\
      --mc-dir /volatile/clas12/$USER/SULI/mc_v01 \\
      --split-dir slurm \\
      --outdir /volatile/clas12/$USER/SULI/datasets/v01 \\
      --features-file scripts/training/feature_list.txt \\
      --p-max 3.0 \\
      --overwrite

Smoke test (no ROOT files needed if you supply --max-files 0):
  python scripts/training/build_dataset.py --help
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SENTINEL = -9999

# Columns always loaded regardless of features-file.
# These match the MC ntuple branch names (processing_mc_pid_training.groovy).
_ALWAYS_COLS = ["p", "theta", "phi", "vz", "sector", "chi2pid",
                "pid", "mc_matching_pid"]

# EB K+ PID code
PID_KPLUS  = 321
PID_PIPLUS = 211


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_features_file(path: pathlib.Path) -> List[str]:
    """
    Parse the audit KEEP list (one feature name per line, # comments stripped).

    WHAT IT DOES
    ------------
    Reads the features file, strips blank lines and comment lines, and returns
    a list of branch names.  Errors if the file is missing or results in an
    empty list (the script must not silently train on zero features).

    PITFALLS
    --------
    Branch names must match the MC ntuple exactly (case-sensitive).  A typo
    here causes KeyError in _load_root_file, not a clean error message.
    """
    lines = []
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    if not lines:
        print(
            f"ERROR: --features-file is empty or contains only comments: {path}\n"
            f"       Fill it with the audit KEEP list from:\n"
            f"         figures/feature_audit/kp/feature_audit_summary.csv\n"
            f"       One branch name per line, # for comments.",
            file=sys.stderr,
        )
        sys.exit(1)
    return lines


def _sha256_file(path: pathlib.Path) -> str:
    """Return hex SHA-256 of a file (for manifest provenance)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha(repo_root: pathlib.Path) -> str:
    """Return the short git SHA of HEAD, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _parse_split_file(path: pathlib.Path) -> List[str]:
    """
    Read a split file (one stem per line, no .root suffix).

    Returns a list of stems.  Empty lines and lines starting with # are
    skipped.
    """
    stems = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            stems.append(s)
    return stems


def _load_root_file(root_path: pathlib.Path, branch_names: List[str]) -> pd.DataFrame:
    """
    Load requested branches from a ROOT file using the numpy fast-path.

    WHAT IT DOES
    ------------
    Opens the ROOT file with uproot, finds the first TTree, and reads the
    requested branches using library="np" and filter_name= (not expressions=).
    This is the same fast-path used by compare_mc_data._load_file and is
    ~30x faster than library="pd" on production-scale multi-column reads.

    WHEN TO USE
    -----------
    Whenever you need a DataFrame from a CLAS12 ntuple ROOT file.  The result
    is a plain pandas DataFrame; downstream code can filter and manipulate it
    without uproot being in scope.

    PITFALLS
    --------
    * filter_name= matches branch names by exact string; branch names absent
      from the TTree are silently dropped.  Check the returned column list if
      you suspect a name mismatch.
    * library="np" returns a dict of 1-D numpy arrays; we construct a DataFrame
      from that dict explicitly.
    * Jagged (variable-length) arrays (e.g. cluster arrays) are not supported
      by this simple loader; all branches here are scalar per-track.
    """
    import uproot

    with uproot.open(str(root_path)) as f:
        # Find the first TTree (the ntuple has a single tree per file).
        tree_key = None
        for key in f.keys():
            obj = f[key]
            if hasattr(obj, "keys"):   # TTree has a .keys() method
                tree_key = key
                break
        if tree_key is None:
            raise RuntimeError(f"No TTree found in {root_path}")

        tree = f[tree_key]
        # filter_name= is the documented fast path (not expressions=).
        arrays = tree.arrays(
            filter_name=branch_names,
            library="np",
        )

    # Build DataFrame from numpy arrays — avoids uproot's pandas overhead.
    df = pd.DataFrame({name: arrays[name] for name in arrays})
    return df


def _apply_selection_and_label(
    df: pd.DataFrame,
    split_name: str,
    p_max: float,
) -> pd.DataFrame:
    """
    Apply EB-K+ selection, momentum cut, and binary label assignment.

    WHAT IT DOES
    ------------
    For all splits: keeps pid==321 rows with p < p_max.
    For train/val: additionally requires mc_matching_pid in {211, 321} and
      assigns a non-nullable int8 label (1=K, 0=π).
    For test: keeps all EB-K+ rows and assigns a nullable Int8 label
      (1=K, 0=π, NA for protons/unmatched/other).

    PITFALLS
    --------
    * The test set intentionally includes protons so evaluate.py can compute
      C^{p→K}.  Do not filter protons out of the test split.
    * train/val must NOT include the test-only nullable rows — the calibration
      and training code expects clean binary labels with no NA.
    """
    # EB-K+ selection
    df = df[df["pid"] == PID_KPLUS].copy()

    # Momentum cap
    df = df[df["p"] < p_max]

    if split_name in ("train", "val"):
        # Binary-labeled rows only
        mask_binary = df["mc_matching_pid"].isin([PID_PIPLUS, PID_KPLUS])
        df = df[mask_binary].copy()
        df["label"] = (df["mc_matching_pid"] == PID_KPLUS).astype(np.int8)
    else:
        # Test: nullable Int8 label
        label = pd.array(
            np.where(
                df["mc_matching_pid"] == PID_KPLUS, 1,
                np.where(df["mc_matching_pid"] == PID_PIPLUS, 0, pd.NA)
            ),
            dtype="Int8",
        )
        df["label"] = label

    return df


def _truth_breakdown(df: pd.DataFrame, split_name: str) -> dict:
    """
    Return a dict with truth-pid counts for the manifest.

    Counts rows by mc_matching_pid value.  Sentinel -9999 is reported as
    the 'unmatched' key.  The 'labeled_K' and 'labeled_pi' keys are the
    two classes used in training.
    """
    vc = df["mc_matching_pid"].value_counts().to_dict()
    breakdown = {
        "labeled_K":    int(vc.get(PID_KPLUS, 0)),
        "labeled_pi":   int(vc.get(PID_PIPLUS, 0)),
        "proton":       int(vc.get(2212, 0)),
        "unmatched":    int(vc.get(SENTINEL, 0)),
        "other":        int(sum(v for k, v in vc.items()
                                if k not in (PID_KPLUS, PID_PIPLUS, 2212, SENTINEL))),
    }
    return breakdown


def _enforce_schema(df: pd.DataFrame, feature_list: List[str],
                    split_name: str) -> pd.DataFrame:
    """
    Enforce fixed column order and dtypes.

    Fixed schema (see module docstring):
      p, theta, phi, vz, sector, chi2pid, <features...>,
      pid, mc_matching_pid, label

    Feature columns: float32, NaN where sentinel -9999.
    pid, mc_matching_pid: int32 (sentinel preserved).
    label: int8 (train/val) or Int8 nullable (test).
    """
    # Replace sentinel with NaN in feature columns (not in pid/mc_matching_pid).
    for col in feature_list:
        if col in df.columns:
            df[col] = df[col].replace(SENTINEL, np.nan).astype("float32")
        else:
            # Column missing in this file — fill with NaN.
            df[col] = np.nan
            df[col] = df[col].astype("float32")

    # Kinematic columns — float32
    for col in ("p", "theta", "phi", "vz", "chi2pid"):
        df[col] = df[col].astype("float32")

    # Sector — int32
    df["sector"] = df["sector"].astype("int32")

    # Metadata — int32 (sentinel preserved)
    df["pid"] = df["pid"].astype("int32")
    df["mc_matching_pid"] = df["mc_matching_pid"].astype("int32")

    # Fixed column order
    cols = ["p", "theta", "phi", "vz", "sector", "chi2pid"] + feature_list + \
           ["pid", "mc_matching_pid", "label"]
    return df[[col for col in cols if col in df.columns]]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    mc_dir: pathlib.Path,
    split_lists: Dict[str, List[str]],
    feature_list: List[str],
    outdir: pathlib.Path,
    p_max: float,
    max_files: Optional[int] = None,
    allow_missing_files: bool = False,
    overwrite: bool = False,
) -> Dict[str, pathlib.Path]:
    """
    Build the train/val/test parquet trio and write a JSON manifest.

    WHAT IT DOES
    ------------
    Iterates the three split stem lists (train, val, test), resolves each stem
    to a ROOT file in mc_dir, loads the requested branches with the numpy
    fast-path, applies EB-K+ selection and labeling, and writes parquet files
    (snappy compression) plus a manifest.json summarising provenance.

    Parameters
    ----------
    mc_dir : path to the directory containing .root files
    split_lists : dict with keys 'train', 'val', 'test' mapping to lists of
        file stems (no .root suffix)
    feature_list : list of branch names to load (from --features-file)
    outdir : directory to write parquet files and manifest into
    p_max : momentum cap (GeV/c); rows with p >= p_max are excluded
    max_files : if set, load at most this many files per split (smoke test)
    allow_missing_files : if True, proceed even when > 5% of stems are missing;
        if False (default), raise RuntimeError
    overwrite : if True, overwrite existing outputs; if False, skip splits
        that already have parquet files

    Returns
    -------
    dict with keys 'train', 'val', 'test', 'manifest' mapping to Path objects

    PITFALLS
    --------
    * feature_list must not include the always-present columns (p, theta, phi,
      vz, sector, chi2pid, pid, mc_matching_pid) — duplicates are silently
      de-duped but it's cleaner to keep features-file focused on ML features.
    * The manifest records the features_file SHA-256 at call time; if the file
      changes after building, rebuild to keep manifest in sync.
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mc_dir = pathlib.Path(mc_dir)

    # Branches to load: always-present + features (de-duped, preserving order).
    seen = set()
    all_branches = []
    for col in _ALWAYS_COLS + feature_list:
        if col not in seen:
            seen.add(col)
            all_branches.append(col)

    # Trim feature_list of any duplicates with ALWAYS_COLS.
    feature_list_clean = [f for f in feature_list if f not in set(_ALWAYS_COLS)]

    split_names = ("train", "val", "test")
    results: Dict[str, pathlib.Path] = {}

    # Flat manifest accumulator dicts — no nesting wrapper.
    m_n_rows:            Dict[str, int]        = {}
    m_truth_breakdown:   Dict[str, dict]       = {}
    m_source_file_stems: Dict[str, List[str]]  = {}
    m_missing_file_stems:Dict[str, List[str]]  = {}
    m_missing_fraction:  Dict[str, float]      = {}

    for split in split_names:
        out_path = outdir / f"{split}.parquet"
        results[split] = out_path

        if out_path.exists() and not overwrite:
            print(f"  [{split}] Already exists, skipping (--overwrite to replace): {out_path}")
            # Try to populate manifest info from existing file.
            try:
                df_existing = pd.read_parquet(out_path, columns=["mc_matching_pid"])
                m_n_rows[split]             = len(df_existing)
                m_truth_breakdown[split]    = _truth_breakdown(df_existing, split)
                m_source_file_stems[split]  = split_lists[split]
                m_missing_file_stems[split] = []
                m_missing_fraction[split]   = 0.0
            except Exception:
                m_n_rows[split]             = -1
                m_truth_breakdown[split]    = {}
                m_source_file_stems[split]  = split_lists[split]
                m_missing_file_stems[split] = []
                m_missing_fraction[split]   = 0.0
            continue

        stems = split_lists[split]
        if max_files is not None and max_files < len(stems):
            stems = stems[:max_files]
            print(f"  [{split}] --max-files {max_files}: capped to {len(stems)} stems")

        dfs = []
        missing_stems = []

        for stem in stems:
            root_path = mc_dir / f"{stem}"    #editted by CB
            if not root_path.exists():
                missing_stems.append(stem)
                print(f"  [{split}] WARNING: file not found: {root_path}", file=sys.stderr)
                continue

            print(f"  [{split}] Loading {root_path.name} ...", end=" ", flush=True)
            try:
                df_raw = _load_root_file(root_path, all_branches)
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)
                missing_stems.append(stem)
                continue

            df_sel = _apply_selection_and_label(df_raw, split, p_max)
            dfs.append(df_sel)
            print(f"{len(df_sel):,} rows after selection")

        n_expected = len(split_lists[split]) if max_files is None \
                     else min(max_files, len(split_lists[split]))
        missing_fraction = len(missing_stems) / max(n_expected, 1)

        if missing_fraction > 0.05 and not allow_missing_files:
            raise RuntimeError(
                f"[{split}] {len(missing_stems)}/{n_expected} files missing "
                f"({100*missing_fraction:.1f}% > 5%). "
                f"Pass --allow-missing-files to proceed anyway, or fix the split file.\n"
                f"Missing stems: {missing_stems[:10]}{'...' if len(missing_stems)>10 else ''}"
            )

        if not dfs:
            raise RuntimeError(
                f"[{split}] No data loaded — all {n_expected} files were missing or failed."
            )

        df_split = pd.concat(dfs, ignore_index=True)
        df_split = _enforce_schema(df_split, feature_list_clean, split)

        # Write parquet with snappy compression.
        df_split.to_parquet(str(out_path), compression="snappy", index=False)
        print(f"  [{split}] Wrote {len(df_split):,} rows → {out_path}")

        m_n_rows[split]             = len(df_split)
        m_truth_breakdown[split]    = _truth_breakdown(df_split, split)
        m_source_file_stems[split]  = stems
        m_missing_file_stems[split] = missing_stems
        m_missing_fraction[split]   = round(missing_fraction, 4)

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest_path = outdir / "manifest.json"
    results["manifest"] = manifest_path

    # Collect repo-root from script location for git SHA.
    _this_file = pathlib.Path(__file__).resolve()
    # Layout: <repo_root>/suli2026_pid/scripts/training/build_dataset.py
    repo_root = _this_file.parent.parent.parent.parent

    # SHA-256 of the features file: resolve from the default path relative to
    # this script's location.  The kwarg was removed from the public API; we
    # locate the file by convention (scripts/training/feature_list.txt) and
    # also accept it via the module-level default path constant.
    _features_file_default = pathlib.Path(__file__).parent / "feature_list.txt"
    features_sha = (
        _sha256_file(_features_file_default)
        if _features_file_default.exists()
        else None
    )

    # Flat manifest — no "splits" nesting wrapper.
    manifest = {
        "feature_list":        feature_list_clean,
        "p_max":               p_max,
        "n_rows":              m_n_rows,
        "truth_breakdown":     m_truth_breakdown,
        "source_file_stems":   m_source_file_stems,
        "missing_file_stems":  m_missing_file_stems,
        "missing_fraction":    m_missing_fraction,
        "mc_dir":              str(mc_dir),
        "features_file_sha256": features_sha,
        "build_timestamp":     datetime.datetime.utcnow().isoformat() + "Z",
        "git_sha":             _git_sha(repo_root),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest written → {manifest_path}")

    return results


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
        "--mc-dir",
        default="/volatile/clas12/${USER}/SULI/mc_v01",
        help="Directory containing the MC ROOT files (default: %(default)s)",
    )
    p.add_argument(
        "--split-dir",
        default="slurm",
        help="Directory containing train_files.txt, val_files.txt, test_files.txt "
             "(default: %(default)s)",
    )
    p.add_argument(
        "--outdir",
        default="datasets/v01",
        help="Output directory for parquet files and manifest (default: %(default)s)",
    )
    p.add_argument(
        "--features-file",
        default="scripts/training/feature_list.txt",
        help="Path to the audit KEEP list; one branch name per line, # comments "
             "(default: %(default)s)",
    )
    p.add_argument(
        "--p-max",
        type=float,
        required=True,
        help="Momentum cap (GeV/c). Cooper's call based on beta-vs-p separation plots. "
             "Expectation: ~3.0 GeV. Recorded in manifest.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help="Load at most N ROOT files per split (smoke test flag).",
    )
    p.add_argument(
        "--allow-missing-files",
        action="store_true",
        default=False,
        help="Proceed even if >5%% of expected ROOT files are missing. "
             "Default: error on >5%% missing.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output parquet files. "
             "Default: skip splits that already exist.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # Expand $USER in mc_dir (useful when called from slurm env)
    mc_dir = pathlib.Path(os.path.expandvars(args.mc_dir))
    split_dir = pathlib.Path(args.split_dir)
    outdir = pathlib.Path(args.outdir)
    features_file = pathlib.Path(args.features_file)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not features_file.exists():
        print(
            f"ERROR: --features-file not found: {features_file}\n"
            f"       Run the Week-3 feature audit first and fill in the KEEP list.\n"
            f"       Reference: figures/feature_audit/kp/feature_audit_summary.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    feature_list = _parse_features_file(features_file)
    print(f"Features loaded ({len(feature_list)}): {feature_list}")

    # Load split files
    split_lists: Dict[str, List[str]] = {}
    for split_name in ("train", "val", "test"):
        split_path = split_dir / f"{split_name}_files.txt"
        if not split_path.exists():
            print(
                f"ERROR: Split file not found: {split_path}\n"
                f"       Run the Phase-0 splitter (see notes/week4_training_examples_plan.md §4)",
                file=sys.stderr,
            )
            sys.exit(1)
        stems = _parse_split_file(split_path)
        split_lists[split_name] = stems
        print(f"  {split_name}: {len(stems)} stems from {split_path}")

    print(f"MC directory: {mc_dir}")
    print(f"p_max: {args.p_max} GeV/c")
    print(f"Output: {outdir}")

    results = build_dataset(
        mc_dir=mc_dir,
        split_lists=split_lists,
        feature_list=feature_list,
        outdir=outdir,
        p_max=args.p_max,
        max_files=args.max_files,
        allow_missing_files=args.allow_missing_files,
        overwrite=args.overwrite,
    )

    print("\nDone.")
    print(f"  train  : {results['train']}")
    print(f"  val    : {results['val']}")
    print(f"  test   : {results['test']}")
    print(f"  manifest: {results['manifest']}")


if __name__ == "__main__":
    main()
