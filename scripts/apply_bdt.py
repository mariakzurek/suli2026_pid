"""
apply_bdt.py — Apply a trained BDT to a ROOT file and write bdt_score (+ bdt_pass).

WHAT IT DOES
------------
Loads a trained BDT wrapper produced by scripts/training/train_bdt.py, reads all
branches from an input ROOT file (data or MC), computes per-track BDT scores, and
writes a new ROOT file containing every original branch plus the new BDT branch(es).

The feature list is authoritative from the model.  Do not pass a feature-list flag;
the wrapper dict embeds exactly what the model was trained on.

Output branch(es):
  bdt_score   float32   calibrated P(kaon) from predict_proba[:, 1].  Always written.
  bdt_pass    bool      score > threshold.  Written only in threshold mode (see below).

COPY-AND-APPEND SEMANTICS
--------------------------
uproot cannot append branches in-place to an existing ROOT file.  Instead, this
script reads the entire input tree into memory (or in batches if the tree is large),
computes scores, then writes a fresh ROOT file with all original branches plus the
new ones.  Writes to a temporary file first and atomically renames to the final
output path on success, so a killed job never leaves a half-written file.

THRESHOLD MODES
---------------
Three modes, mutually exclusive:

  (none)                   Write bdt_score only.

  --threshold FLOAT        Write bdt_score and bdt_pass = (score > threshold).
                           Strict greater-than; a track with score == threshold
                           does NOT pass.  Threshold must be in [0, 1].

  --threshold-csv PATH     Per-(p, theta)-bin threshold lookup.  CSV must have
                           columns: p_low, p_high, theta_low, theta_high, t_optimal
                           (matching the per_bin_sweep.csv convention from evaluate.py).
                           For each track, the bin whose [p_low, p_high) x
                           [theta_low, theta_high) contains that track's (p, theta)
                           is looked up; bdt_pass = (score > t_optimal) for that bin.
                           Tracks that fall outside every bin get bdt_pass = False.
                           Out-of-bin assignment is deliberate and conservative: if
                           the model was not calibrated/evaluated in a kinematic region,
                           do not silently accept tracks from it.

OUT-OF-BIN BEHAVIOR (CSV MODE)
--------------------------------
Tracks whose (p, theta) falls outside all bins in the CSV get bdt_pass = False.
This is intentional and conservative.  Inspect bdt_score directly for such tracks
if a different policy is needed.  The out-of-bin count is printed to stdout at the
end of the run.

STRICT > COMPARISON
--------------------
Both threshold modes use strict greater-than (score > threshold), not >=.
This matches the convention in evaluate.py (_bin_metrics_at_threshold) and ensures
a track at exactly the threshold boundary is rejected.

WHEN TO USE
-----------
Run this after train_bdt.py and evaluate.py have produced a final per-bin threshold
CSV.  Apply to both the MC test set (validation that output scores match evaluate.py
reference) and to data ROOT files in /volatile/clas12/$USER/SULI/data_v01/.

For large directories of ROOT files, use the slurm array wrapper:
  ./slurm/submit_apply_bdt.sh --input-dir /volatile/clas12/$USER/SULI/data_v01/ \\
      --model /work/clas12/$USER/SULI/models/tier1_v01/model.joblib \\
      --output-dir /volatile/clas12/$USER/SULI/scored_data_v01/ \\
      --threshold-csv /volatile/clas12/$USER/SULI/eval/v01/per_bin_sweep.csv

PITFALLS
--------
* The model wrapper dict {"model": ..., "features": [...]} is the only supported
  format.  Old bare-estimator joblib files are NOT supported here (unlike evaluate.py
  which has a manifest fallback).  Rebuild the model with the current train_bdt.py.
* If any feature column is missing from the input ROOT tree, the script fails
  loudly naming the missing column(s).  No imputation, no silent dropping.
* bdt_score is stored as float32 to keep output ROOT files manageable.  The
  underlying predict_proba returns float64 internally; the cast happens at write time.
* bdt_pass is stored as numpy bool_ and written as a boolean branch.  uproot 5.x
  supports bool branches natively; the branch reads back as dtype('bool').
* The input tree is read into memory.  For very large files (> a few GB), use the
  --batch-size flag to process in chunks; the output file is still a single file.
* The output file path must differ from the input path — the script refuses to
  overwrite the source.

Usage:
  python scripts/apply_bdt.py \\
      --input  /volatile/clas12/$USER/SULI/data_v01/run_001.root \\
      --model  /work/clas12/$USER/SULI/models/tier1_v01/model.joblib \\
      --output /volatile/clas12/$USER/SULI/scored_data_v01/run_001.root

  # With a single global threshold:
  python scripts/apply_bdt.py \\
      --input  run_001.root --model model.joblib --output run_001_scored.root \\
      --threshold 0.72

  # With per-bin thresholds from evaluate.py:
  python scripts/apply_bdt.py \\
      --input  run_001.root --model model.joblib --output run_001_scored.root \\
      --threshold-csv /volatile/.../eval/v01/per_bin_sweep.csv
"""

import argparse
import os
import pathlib
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import uproot


# ──────────────────────────────────────────────────────────────────────────────
# Default tree name
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TREE = "PhysicsEvents"

# Column names for the per-bin threshold lookup (p and theta in the input tree).
# Confirmed against the 60-col MC / 57-col data schema in compare_mc_data.py
# (cols 10-11: p, theta).
P_COLUMN     = "p"
THETA_COLUMN = "theta"


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_model(model_path: pathlib.Path):
    """
    Load the model wrapper dict from a joblib file.

    WHAT IT DOES
    ------------
    Loads the wrapper dict {"model": calibrated_clf, "features": [...]} produced
    by train_bdt.py.  Validates that the dict has the expected keys and that the
    feature list is non-empty.  This is the only supported format; the manifest
    fallback present in evaluate.py is deliberately omitted here because apply_bdt.py
    is new tooling and should only use the current, correct format.

    PITFALLS
    --------
    * Old bare-estimator joblib files (pre-week4-tier-flexible) will raise a clear
      error rather than silently failing downstream.
    * The model object is a CalibratedClassifierCV; call predict_proba on it directly.
    """
    raw = joblib.load(str(model_path))
    if not (isinstance(raw, dict) and "model" in raw and "features" in raw):
        print(
            f"ERROR: {model_path} does not contain the expected wrapper dict.\n"
            f"  Expected: dict with keys 'model' and 'features'.\n"
            f"  Got: {type(raw)}\n\n"
            f"  This script requires the new model format produced by train_bdt.py\n"
            f"  (week4-tier-flexible or later).  Rebuild the model:\n"
            f"    python scripts/training/train_bdt.py --features-file <tier> ...",
            file=sys.stderr,
        )
        sys.exit(1)

    model    = raw["model"]
    features = raw["features"]

    if not features:
        print(
            f"ERROR: model wrapper at {model_path} has an empty feature list.",
            file=sys.stderr,
        )
        sys.exit(1)

    return model, features


# ──────────────────────────────────────────────────────────────────────────────
# Threshold-CSV loading and per-track lookup
# ──────────────────────────────────────────────────────────────────────────────

def _load_threshold_csv(csv_path: pathlib.Path) -> pd.DataFrame:
    """
    Load per-bin threshold CSV and validate required columns.

    WHAT IT DOES
    ------------
    Reads a CSV with at least the five columns:
      p_low, p_high, theta_low, theta_high, t_optimal
    as produced by a threshold-selection step operating on per_bin_sweep.csv
    from evaluate.py.  Each row defines one (p, theta) bin and its optimal
    BDT score threshold.

    PITFALLS
    --------
    * p ranges are [p_low, p_high) — left-inclusive, right-exclusive.
    * theta ranges are [theta_low, theta_high) — same convention.
    * t_optimal must be a float in [0, 1]; not validated here but upstream.
    """
    required = {"p_low", "p_high", "theta_low", "theta_high", "best_threshold"}
    df = pd.read_csv(str(csv_path))
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(
            f"ERROR: --threshold-csv {csv_path} is missing required columns:\n"
            f"  {sorted(missing_cols)}\n\n"
            f"  Required columns: {sorted(required)}\n"
            f"  Found columns:    {sorted(df.columns.tolist())}\n\n"
            f"  The threshold CSV must have columns p_low, p_high, theta_low,\n"
            f"  theta_high, t_optimal.  Produce it from evaluate.py's\n"
            f"  per_bin_sweep.csv by selecting the optimal threshold per bin.",
            file=sys.stderr,
        )
        sys.exit(1)
    return df[["p_low", "p_high", "theta_low", "theta_high", "best_threshold"]].copy()


def _apply_threshold_csv(
    scores: np.ndarray,
    p_arr: np.ndarray,
    theta_arr: np.ndarray,
    threshold_df: pd.DataFrame,
) -> tuple:
    """
    Per-track bin lookup: assign bdt_pass based on per-(p,theta)-bin t_optimal.

    WHAT IT DOES
    ------------
    For each track, finds the threshold-CSV row whose [p_low, p_high) x
    [theta_low, theta_high) bin contains that track's (p, theta).  Compares
    bdt_score strictly against t_optimal (score > t_optimal → pass).

    Tracks outside every bin get bdt_pass = False.  The number of out-of-bin
    tracks is returned as the second element of the tuple.

    WHEN TO USE
    -----------
    Called only when --threshold-csv is provided.  For a single global threshold,
    use the simple score > threshold comparison instead.

    PITFALLS
    --------
    * Bin matching iterates over CSV rows (typically ≤ ~20 bins), with numpy
      boolean masking used within each iteration to select matching tracks.
      The per-bin loop is fast in practice; typical bin counts are small.
    * If multiple CSV rows cover the same (p, theta) point (overlapping bins),
      the first matching row is used.  Ensure the CSV has non-overlapping bins.
    * Out-of-bin tracks: bdt_pass = False.  This is conservative and intentional.
    * Bin edges follow [p_low, p_high) x [theta_low, theta_high) — strict-less
      on the upper edge, as in evaluate.py.
    """
    n = len(scores)
    bdt_pass = np.zeros(n, dtype=np.bool_)
    assigned = np.zeros(n, dtype=np.bool_)

    p_low     = threshold_df["p_low"].to_numpy()
    p_high    = threshold_df["p_high"].to_numpy()
    t_low     = threshold_df["theta_low"].to_numpy()
    t_high    = threshold_df["theta_high"].to_numpy()
    t_optimal = threshold_df["best_threshold"].to_numpy()

    for bi in range(len(threshold_df)):
        in_bin = (
            (p_arr     >= p_low[bi])  & (p_arr     < p_high[bi]) &
            (theta_arr >= t_low[bi])  & (theta_arr < t_high[bi]) &
            (~assigned)
        )
        bdt_pass[in_bin] = scores[in_bin] > t_optimal[bi]
        assigned[in_bin] = True

    n_out_of_bin = int((~assigned).sum())
    return bdt_pass, n_out_of_bin


# ──────────────────────────────────────────────────────────────────────────────
# Core apply logic
# ──────────────────────────────────────────────────────────────────────────────

def apply_bdt(
    input_path: pathlib.Path,
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    tree_name: str = DEFAULT_TREE,
    threshold: float = None,
    threshold_csv: pathlib.Path = None,
    batch_size: int = None,
    _preloaded_model=None,
    _preloaded_features=None,
) -> None:
    """
    Apply BDT to one ROOT file and write augmented output.

    WHAT IT DOES
    ------------
    1. Loads the model wrapper dict and extracts feature list + calibrated model.
       (If _preloaded_model/_preloaded_features are supplied by main(), the load
       is skipped — avoids loading the file twice when the CLI prints a banner.)
    2. Opens the input ROOT tree with uproot and reads all branches.
    3. Validates all required feature columns are present; fails fast if any missing.
    4. Calls predict_proba on the feature matrix to get bdt_score (float32).
    5. Optionally computes bdt_pass (bool) in either global or per-bin threshold mode.
    6. Writes a new ROOT file with all original branches + new branch(es).
       Uses a temp file + atomic rename to avoid half-written outputs on failure.

    Parameters
    ----------
    input_path    : source ROOT file
    model_path    : model.joblib wrapper dict from train_bdt.py
    output_path   : destination ROOT file (must differ from input_path)
    tree_name     : ROOT tree name (default: PhysicsEvents)
    threshold     : float in [0,1]; enables global bdt_pass (mutually exclusive
                    with threshold_csv)
    threshold_csv : path to per-bin CSV; enables per-bin bdt_pass (mutually
                    exclusive with threshold)
    batch_size    : if not None, process this many events at a time (reduces
                    peak RAM for very large files; single output file still written)
    _preloaded_model, _preloaded_features : internal use by main() to avoid
                    loading model.joblib a second time after the banner load.
    """
    # ── Sanity: output must not overwrite input ────────────────────────────────
    if input_path.resolve() == output_path.resolve():
        print(
            "ERROR: --output must differ from --input.\n"
            "       This script cannot overwrite the source ROOT file in-place.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load model (or reuse pre-loaded result from main()) ───────────────────
    if _preloaded_model is not None and _preloaded_features is not None:
        model, features = _preloaded_model, _preloaded_features
        print(f"Model: {model_path} (pre-loaded, {len(features)} features)")
    else:
        print(f"Loading model: {model_path}")
        model, features = _load_model(model_path)
    print(f"  Features ({len(features)}): {features}")

    # ── Load threshold CSV if needed ──────────────────────────────────────────
    threshold_df = None
    if threshold_csv is not None:
        print(f"Loading threshold CSV: {threshold_csv}")
        threshold_df = _load_threshold_csv(threshold_csv)
        print(f"  {len(threshold_df)} bin(s) loaded.")
        threshold_mode = "csv"
    elif threshold is not None:
        threshold_mode = "global"
        print(f"  Global threshold: {threshold} (strict greater-than)")
    else:
        threshold_mode = "none"

    # ── Open input ROOT file ───────────────────────────────────────────────────
    print(f"Opening input: {input_path} (tree: {tree_name})")
    with uproot.open(str(input_path)) as f:
        if tree_name not in f:
            available = [k.split(";")[0] for k in f.keys()]
            print(
                f"ERROR: Tree '{tree_name}' not found in {input_path}.\n"
                f"       Available objects: {available}\n"
                f"       Use --tree to specify a different tree name.",
                file=sys.stderr,
            )
            sys.exit(1)
        tree = f[tree_name]
        n_entries = tree.num_entries
        print(f"  Tree entries: {n_entries:,}")

        # ── Validate feature columns ────────────────────────────────────────
        tree_keys = set(b.name for b in tree.branches)
        missing_features = [feat for feat in features if feat not in tree_keys]
        if missing_features:
            print(
                f"ERROR: The following features required by the model are not\n"
                f"       present in tree '{tree_name}' of {input_path}:\n"
                f"  Missing: {missing_features}\n\n"
                f"  Model path: {model_path}\n"
                f"  Model features: {features}\n\n"
                f"  Check that the input ROOT file was produced with the same\n"
                f"  column schema as the training data (compare_mc_data.py column map).",
                file=sys.stderr,
            )
            sys.exit(1)

        # ── Validate p/theta columns for CSV mode ──────────────────────────
        if threshold_mode == "csv":
            for col in (P_COLUMN, THETA_COLUMN):
                if col not in tree_keys:
                    print(
                        f"ERROR: Column '{col}' required for --threshold-csv lookup\n"
                        f"       is not present in tree '{tree_name}' of {input_path}.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        # ── Read all branches ───────────────────────────────────────────────
        all_branch_names = [b.name for b in tree.branches]

        print(f"  Reading {len(all_branch_names)} branches ...")

        if batch_size is None or n_entries <= batch_size:
            # Single in-memory read (typical case).
            arrays = tree.arrays(library="np")
            _process_and_write(
                arrays=arrays,
                all_branch_names=all_branch_names,
                features=features,
                model=model,
                threshold_mode=threshold_mode,
                threshold=threshold,
                threshold_df=threshold_df,
                output_path=output_path,
                tree_name=tree_name,
                n_entries=n_entries,
            )
        else:
            # Batched read — reduces peak RAM for large files.
            _process_batched(
                tree=tree,
                all_branch_names=all_branch_names,
                features=features,
                model=model,
                threshold_mode=threshold_mode,
                threshold=threshold,
                threshold_df=threshold_df,
                output_path=output_path,
                tree_name=tree_name,
                n_entries=n_entries,
                batch_size=batch_size,
            )


def _process_and_write(
    arrays,
    all_branch_names,
    features,
    model,
    threshold_mode,
    threshold,
    threshold_df,
    output_path,
    tree_name,
    n_entries,
):
    """
    Score all events and write output.  Called when the full tree fits in RAM.

    WHAT IT DOES
    ------------
    Builds the feature matrix from the numpy arrays dict, runs predict_proba,
    optionally computes bdt_pass, then calls _write_output_root with all data.

    PITFALLS
    --------
    * Feature matrix is cast to float32 before scoring (matches training convention
      in train_bdt.py and evaluate.py).
    * bdt_score output is stored as float32; the cast from float64 predict_proba
      output happens here.
    """
    print("  Computing BDT scores ...")
    X = np.column_stack([arrays[f].astype(np.float32) for f in features])
    scores_f64 = model.predict_proba(X)[:, 1]
    scores     = scores_f64.astype(np.float32)
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    bdt_pass = None
    n_out_of_bin = 0

    if threshold_mode == "global":
        bdt_pass = (scores > threshold).astype(np.bool_)
        n_pass = int(bdt_pass.sum())
        print(f"  bdt_pass (global threshold={threshold}): {n_pass:,} / {n_entries:,} tracks pass")

    elif threshold_mode == "csv":
        p_arr     = arrays[P_COLUMN].astype(np.float64)
        theta_arr = arrays[THETA_COLUMN].astype(np.float64)
        bdt_pass_bool, n_out_of_bin = _apply_threshold_csv(
            scores_f64, p_arr, theta_arr, threshold_df
        )
        bdt_pass = bdt_pass_bool  # already np.bool_ from _apply_threshold_csv
        n_pass = int(bdt_pass.sum())
        print(f"  bdt_pass (per-bin CSV): {n_pass:,} / {n_entries:,} tracks pass")
        if n_out_of_bin > 0:
            print(f"  WARNING: {n_out_of_bin:,} tracks had (p,theta) outside all CSV bins → bdt_pass=False")

    _write_output_root(
        arrays=arrays,
        all_branch_names=all_branch_names,
        scores=scores,
        bdt_pass=bdt_pass,
        output_path=output_path,
        tree_name=tree_name,
    )

    if n_out_of_bin > 0:
        print(f"  Out-of-bin tracks (bdt_pass forced False): {n_out_of_bin:,}")


def _process_batched(
    tree,
    all_branch_names,
    features,
    model,
    threshold_mode,
    threshold,
    threshold_df,
    output_path,
    tree_name,
    n_entries,
    batch_size,
):
    """
    Batched scoring + writing for large trees.  Streams batches through the model
    and accumulates all results before writing a single output file.

    WHAT IT DOES
    ------------
    Reads the input tree in chunks of batch_size events, scores each chunk, and
    concatenates all arrays in memory before the final write.  This reduces peak
    RAM compared to a single read when the model scoring memory footprint is small
    relative to the raw array data.

    PITFALLS
    --------
    * All branch data is still concatenated in memory at write time.  If memory is
      truly limited, consider splitting the output across files (not supported here
      — single-output-file semantics are required by the design).
    """
    all_scores    = []
    all_bdt_pass  = []
    all_arrays    = {name: [] for name in all_branch_names}
    n_out_of_bin_total = 0

    for start in range(0, n_entries, batch_size):
        stop = min(start + batch_size, n_entries)
        print(f"  Batch [{start}:{stop}] ...")
        batch = tree.arrays(library="np", entry_start=start, entry_stop=stop)

        X = np.column_stack([batch[f].astype(np.float32) for f in features])
        scores_f64 = model.predict_proba(X)[:, 1]
        scores_batch = scores_f64.astype(np.float32)
        all_scores.append(scores_batch)

        for name in all_branch_names:
            all_arrays[name].append(batch[name])

        if threshold_mode == "global":
            bdt_pass_batch = (scores_batch > threshold).astype(np.bool_)
            all_bdt_pass.append(bdt_pass_batch)

        elif threshold_mode == "csv":
            p_arr     = batch[P_COLUMN].astype(np.float64)
            theta_arr = batch[THETA_COLUMN].astype(np.float64)
            bdt_pass_bool, n_oob = _apply_threshold_csv(
                scores_f64, p_arr, theta_arr, threshold_df
            )
            all_bdt_pass.append(bdt_pass_bool)  # already np.bool_ from _apply_threshold_csv
            n_out_of_bin_total += n_oob

    # Concatenate all batches.
    scores = np.concatenate(all_scores)
    bdt_pass = np.concatenate(all_bdt_pass) if all_bdt_pass else None
    arrays = {name: np.concatenate(chunks) for name, chunks in all_arrays.items()}

    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    if threshold_mode != "none":
        n_pass = int(bdt_pass.sum())
        print(f"  bdt_pass: {n_pass:,} / {n_entries:,} tracks pass")
    if n_out_of_bin_total > 0:
        print(f"  WARNING: {n_out_of_bin_total:,} tracks outside all CSV bins → bdt_pass=False")

    _write_output_root(
        arrays=arrays,
        all_branch_names=all_branch_names,
        scores=scores,
        bdt_pass=bdt_pass,
        output_path=output_path,
        tree_name=tree_name,
    )

    if n_out_of_bin_total > 0:
        print(f"  Out-of-bin tracks (bdt_pass forced False): {n_out_of_bin_total:,}")


def _write_output_root(
    arrays,
    all_branch_names,
    scores,
    bdt_pass,
    output_path: pathlib.Path,
    tree_name: str,
):
    """
    Write all branches plus new BDT branch(es) to a ROOT file.

    WHAT IT DOES
    ------------
    Writes a new ROOT file using uproot.recreate, copying all original branches
    from the input tree and appending bdt_score (float32) and optionally bdt_pass
    (bool).  Uses a temporary file + atomic rename so that if the write fails
    mid-way, the output path is never left in a partial state.

    PITFALLS
    --------
    * uproot cannot append to an existing ROOT file — this is a full copy-and-write.
    * bdt_pass is stored as numpy bool_ (dtype('bool')).  uproot 5.x supports
      boolean branches natively; the branch reads back as dtype('bool') via uproot
      and as Bool_t / 'O' via PyROOT.
    * The atomic rename (os.replace) is POSIX-atomic on the same filesystem.
      Ensure the output directory and the tempfile directory are on the same
      filesystem (both under the output parent directory).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory (same filesystem as output).
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(output_path.parent),
        suffix=".tmp.root",
        prefix=f".{output_path.name}.",
    )
    os.close(tmp_fd)

    try:
        print(f"  Writing output → {tmp_path} (will rename to {output_path})")
        branch_dict = {}
        for name in all_branch_names:
            branch_dict[name] = arrays[name]

        branch_dict["bdt_score"] = scores

        if bdt_pass is not None:
            branch_dict["bdt_pass"] = bdt_pass

        with uproot.recreate(tmp_path) as out_f:
            out_f[tree_name] = branch_dict

        # Atomic rename.
        os.replace(tmp_path, str(output_path))
        print(f"  Output written: {output_path}")

    except Exception:
        # Clean up the temp file on failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Apply a trained BDT to a ROOT file and write bdt_score (+ bdt_pass).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Input ROOT file (data or MC ntuple).",
    )
    p.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Path to model.joblib wrapper dict produced by train_bdt.py.  "
             "Must contain {'model': ..., 'features': [...]}.",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output ROOT file (must differ from --input).  "
             "All original branches are preserved; bdt_score (and bdt_pass "
             "if --threshold or --threshold-csv is given) are appended.",
    )
    p.add_argument(
        "--tree",
        default=DEFAULT_TREE,
        metavar="NAME",
        help=f"ROOT tree name (default: {DEFAULT_TREE}).",
    )

    # Threshold mode — mutually exclusive.
    thresh_group = p.add_mutually_exclusive_group()
    thresh_group.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Global BDT score threshold in [0,1].  If given, also writes "
             "bdt_pass = (bdt_score > threshold) (strict greater-than).  "
             "Mutually exclusive with --threshold-csv.",
    )
    thresh_group.add_argument(
        "--threshold-csv",
        default=None,
        metavar="PATH",
        help="Path to per-(p,theta)-bin threshold CSV with columns: "
             "p_low, p_high, theta_low, theta_high, t_optimal.  "
             "Tracks outside all bins get bdt_pass=False.  "
             "Mutually exclusive with --threshold.",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="If set, process this many events at a time to reduce peak RAM.  "
             "Default: read the entire tree in one pass.  "
             "Output is always a single ROOT file.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.  "
             "Default: refuse and exit if the output path already exists.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    input_path  = pathlib.Path(args.input)
    model_path  = pathlib.Path(args.model)
    output_path = pathlib.Path(args.output)

    # ── Preflight checks ──────────────────────────────────────────────────────
    if not input_path.exists():
        print(f"ERROR: Input ROOT file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if output_path.exists() and not args.overwrite:
        print(
            f"ERROR: Output already exists: {output_path}\n"
            f"       Pass --overwrite to replace it.",
            file=sys.stderr,
        )
        sys.exit(1)

    threshold_csv = pathlib.Path(args.threshold_csv) if args.threshold_csv else None
    if threshold_csv is not None and not threshold_csv.exists():
        print(f"ERROR: --threshold-csv not found: {threshold_csv}", file=sys.stderr)
        sys.exit(1)

    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        print(
            f"ERROR: --threshold {args.threshold} is out of range.  "
            f"Must be in [0, 1].",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load model once — used for both the banner and scoring ───────────────
    # _load_model validates the wrapper dict format and fails with a clear error
    # if model_path is an old bare-estimator file.  Loading here (not in a raw
    # joblib.load) ensures the correct error path fires before the banner prints.
    model, features = _load_model(model_path)

    # ── Determine threshold mode for banner ──────────────────────────────────
    if args.threshold is not None:
        threshold_mode_label = f"global threshold={args.threshold}"
    elif threshold_csv is not None:
        threshold_mode_label = f"per-bin CSV={threshold_csv}"
    else:
        threshold_mode_label = "none (bdt_score only)"

    # ── Start banner ─────────────────────────────────────────────────────────
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("apply_bdt.py — BDT application start")
    print(f"  input          : {input_path}")
    print(f"  model          : {model_path}")
    print(f"  output         : {output_path}")
    print(f"  tree           : {args.tree}")
    print(f"  feature count  : {len(features)}")
    print(f"  threshold mode : {threshold_mode_label}")
    if args.batch_size:
        print(f"  batch size     : {args.batch_size:,}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── Apply ─────────────────────────────────────────────────────────────────
    # Pass the pre-loaded model so apply_bdt() does not load it a second time.
    apply_bdt(
        input_path=input_path,
        model_path=model_path,
        output_path=output_path,
        tree_name=args.tree,
        threshold=args.threshold,
        threshold_csv=threshold_csv,
        batch_size=args.batch_size,
        _preloaded_model=model,
        _preloaded_features=features,
    )

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("apply_bdt.py DONE")
    print(f"  output: {output_path}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
