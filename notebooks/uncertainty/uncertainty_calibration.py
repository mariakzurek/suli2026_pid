#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Calibration sensitivity check
#
# Compares per-bin kaon contamination C^{pi->K} using the
# model's Platt-calibrated score vs. the uncalibrated (base
# estimator) score, applying the SAME per-bin BDT thresholds
# already optimized on the calibrated score (no re-optimization
# for the uncalibrated case -- that's the point of the check).
#
# If delta_c is smaller than the MC statistical uncertainty in
# every bin, calibration has a negligible effect and this
# source can be closed out.
#
# IMPORTANT: the attribute path to the uncalibrated base
# estimator:
#     model["model"].calibrated_classifiers_[0].base_estimator
# is specific to scikit-learn 1.5.x's CalibratedClassifierCV
# with cv='prefit'. VERIFY THIS INTERACTIVELY -- run the
# "verify attribute chain" block below by itself first and
# confirm it prints a real estimator with predict_proba, before
# trusting the bulk loop. In newer sklearn this attribute was
# renamed `estimator`; the code below tries both.
# ============================================================

import sys
sys.path.append("../../scripts/")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import common_functions as au

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

model_path = "/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib"
csvLocation = "../../figures/full_range/optimized_thresholdsV4.csv"

# ASSUMPTION: bins used to build optimized_thresholdsV4.csv.
# Not given in the snippet you sent -- guessed to match the
# "full range" V4 threshold set (0-35 theta, 2.5-5 p) used
# elsewhere in this analysis. EDIT if wrong -- everything below
# only makes sense if these match how the CSV was built.
tStart, tEnd, tBinNum = 0, 35, 5
pStart, pEnd, pBinNum = 2.5, 5, 10

tBinEdges = np.linspace(tStart, tEnd, tBinNum + 1)
pBinEdges = np.linspace(pStart, pEnd, pBinNum + 1)

outCSV = "calibration_sensitivity.csv"

# ------------------------------------------------------------
# Load model, data, thresholds
# ------------------------------------------------------------

df_val = pd.read_parquet("~/ML_Files/dataset_v03/val.parquet")
mod, mod_df = au.load_model_and_data(model_path, df_val)

df_test = pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/test.parquet")
df_test = df_test[df_test["mc_matching_pid"] != -9999]

feature_names = au.get_feature_names(model_path)
print(feature_names)

# Track which column apply_model_to_df adds, rather than assuming
# a name -- avoids silently guessing wrong if it's "score" vs
# "bdt_score" vs something else.
cols_before = set(df_test.columns)
df_test = au.apply_model_to_df(mod, df_test, feature_names)
new_cols = set(df_test.columns) - cols_before
print(f"apply_model_to_df added columns: {sorted(new_cols)}")

if "score" in df_test.columns:
    SCORE_COL = "score"
elif "bdt_score" in df_test.columns:
    SCORE_COL = "bdt_score"
elif len(new_cols) == 1:
    SCORE_COL = next(iter(new_cols))
else:
    raise RuntimeError(
        "Could not determine which column holds the calibrated score. "
        f"apply_model_to_df added: {sorted(new_cols)}. Set SCORE_COL manually."
    )

print(f"Using '{SCORE_COL}' as the calibrated score column")

if not Path(csvLocation).exists():
    raise FileNotFoundError(
        f"{csvLocation} not found -- this script expects thresholds "
        "already optimized on the CALIBRATED score, it does not "
        "re-run optimization."
    )

print(f"Loading optimized thresholds from {csvLocation}")
results_df = pd.read_csv(csvLocation)

# ------------------------------------------------------------
# Normalize threshold CSV column names
# ------------------------------------------------------------
# apply_optimized_bdt_cut expects specific column names
# (confirmed from the traceback: at least p_low/p_high). CSVs
# in this project have used different naming conventions in
# different places (p_lo/p_hi, etc.), so rename whatever is
# actually in results_df to match, rather than assuming the
# CSV was already built with the right names.

print("results_df columns (before rename):", results_df.columns.tolist())

RENAME_CANDIDATES = {
    "p_low":  ["p_low", "p_lo", "pLow", "p_min"],
    "p_high": ["p_high", "p_hi", "pHigh", "p_max"],
    "theta_low":  ["theta_low", "thetaLow", "t_low", "t_lo", "tLow", "theta_lo"],
    "theta_high": ["theta_high", "thetaHigh", "t_high", "t_hi", "tHigh", "theta_hi"],
}

rename_map = {}
for canonical, candidates in RENAME_CANDIDATES.items():
    if canonical in results_df.columns:
        continue  # already correct
    for cand in candidates:
        if cand in results_df.columns:
            rename_map[cand] = canonical
            break

if rename_map:
    print(f"Renaming columns: {rename_map}")
    results_df = results_df.rename(columns=rename_map)

print("results_df columns (after rename):", results_df.columns.tolist())

required = {"p_low", "p_high", "theta_low", "theta_high"}
missing = required - set(results_df.columns)
if missing:
    raise KeyError(
        f"results_df is missing required column(s) {missing} even after "
        f"renaming. Available columns: {results_df.columns.tolist()}. "
        "Add the actual column name to RENAME_CANDIDATES above and rerun -- "
        "I can't see your CSV so I can't guess every possible name."
    )

# ------------------------------------------------------------
# Verify the attribute chain to the uncalibrated base estimator
# ------------------------------------------------------------
# RUN THIS BLOCK BY ITSELF FIRST. Confirm base_estimator is a
# real fitted estimator with predict_proba before trusting the
# bulk loop below.

raw_model = joblib.load(model_path)
calibrated_clf = raw_model["model"]

try:
    base_estimator = calibrated_clf.calibrated_classifiers_[0].base_estimator
except AttributeError:
    # sklearn >= ~1.4 in some builds renamed this to `estimator`
    base_estimator = calibrated_clf.calibrated_classifiers_[0].estimator

print("Base estimator:", type(base_estimator))
assert hasattr(base_estimator, "predict_proba"), (
    "Resolved object has no predict_proba -- the attribute chain is "
    "wrong for this sklearn version. Inspect "
    "calibrated_clf.calibrated_classifiers_[0].__dict__.keys() and "
    "fix the attribute name before running in bulk."
)

# ------------------------------------------------------------
# Compute uncalibrated score for df_test
# ------------------------------------------------------------
# NOTE: this assumes df_test[feature_names], in this column
# order, is exactly what apply_model_to_df feeds the model
# internally (no additional scaling/transform). If
# apply_model_to_df does extra preprocessing, replicate that
# here too, or the uncalibrated score won't be apples-to-apples.

X_test = df_test[feature_names]

uncal_proba = base_estimator.predict_proba(X_test)[:, 1]  # kaon-class prob
df_test["score_uncalibrated"] = uncal_proba

# ------------------------------------------------------------
# Per-bin contamination: calibrated vs uncalibrated
# ------------------------------------------------------------

thetaBins = au.makeBins(df_test, "theta", binEdges=tBinEdges)

rows = []

for i, thetaBin in enumerate(thetaBins):
    theta_lo, theta_hi = tBinEdges[i], tBinEdges[i + 1]

    pBins = au.makeBins(thetaBin, "p", binEdges=pBinEdges)

    for j, pbin in enumerate(pBins):
        p_lo, p_hi = pBinEdges[j], pBinEdges[j + 1]

        # Calibrated selection -- uses the score column as-is
        calMask = au.apply_optimized_bdt_cut(pbin, threshold_df=results_df)
        cal_selected = pbin[calMask]
        c_cal, err_cal = au.compute_contamination(cal_selected)

        # Uncalibrated selection -- SAME per-bin thresholds, just
        # applied to the uncalibrated score instead. Swap the
        # score column so apply_optimized_bdt reads the
        # uncalibrated values without changing its logic.
        pbin_uncal = pbin.copy()
        pbin_uncal[SCORE_COL] = pbin_uncal["score_uncalibrated"]
        uncalMask = au.apply_optimized_bdt_cut(pbin_uncal, threshold_df=results_df)
        uncal_selected = pbin[uncalMask]
        c_uncal, err_uncal = au.compute_contamination(uncal_selected)

        delta_c = abs(c_uncal - c_cal)

        rows.append({
            "theta_lo": theta_lo,
            "theta_hi": theta_hi,
            "p_lo": p_lo,
            "p_hi": p_hi,
            "c_calibrated": c_cal,
            "c_uncalibrated": c_uncal,
            "delta_c": delta_c,
        })

out_df = pd.DataFrame(rows)
out_df.to_csv(outCSV, index=False)

print(f"Wrote {len(out_df)} rows to {outCSV}")
print(out_df)