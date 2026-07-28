#!/usr/bin/env python
# coding: utf-8

# ============================================================
# RICH-truth kaon contamination (BDT + EB PID), binned in (theta, p)
#
# For each (theta, p) bin, selects kaon candidates using BOTH
# the existing bdt_pass flag AND EB PID (pid == 321), then
# computes contamination against RICH truth (rich_best_PID),
# at two rich_RQ quality thresholds: the "current" value and
# that value +0.1. Writes a CSV with one row per bin.
#
# Mirrors the working DATA-contamination block: candidates are
# pid==321 & bdt_pass==True, contamination = fraction of those
# candidates whose rich_best_PID != 321, with binomial-style
# error r * sqrt(1/numerator + 1/denominator).
#
# No MC truth and no chi2pid selection are used here -- RICH
# is the only truth reference.
# ============================================================

import sys
sys.path.append("../../scripts/")

import numpy as np
import pandas as pd
import uproot

import common_functions as au

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

RICH_RQ_INITIAL = 0.2          # "current" rich_RQ threshold
RICH_RQ_SHIFT   = 0.1          # amount added for the shifted version
RICH_RQ_SHIFTED = RICH_RQ_INITIAL + RICH_RQ_SHIFT

THETA_LO, THETA_HI, THETA_NBINS = 5, 15, 3
P_LO, P_HI, P_NBINS = 2.5, 5, 10

outCSV = "rich_contamination_binned.csv"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

cols = [
    "pid",
    "p",
    "theta",
    "beta",
    "chi2pid",
    "rich_RQ",
    "vz",
    "bdt_pass",
    "rich_best_PID",
    "rich_best_ntot",
    "bdt_score"
]

kinematics = [
    "Mx_eKX",
    "Mx_epiX",
    "Mx_epX",
    "Q2",
    "W",
    "y"
]

cols.extend(kinematics)

df = uproot.open(
    "~/ML_Files/scored_data_v01/nSidis_005046.root:PhysicsEvents"
).arrays(cols, library="pd")

df = au.apply_Sidis_Cuts(df)

# ------------------------------------------------------------
# Bin edges
# ------------------------------------------------------------

thetaEdges = np.linspace(THETA_LO, THETA_HI, THETA_NBINS + 1)
pEdges = np.linspace(P_LO, P_HI, P_NBINS + 1)


def calc_contamination(candidates):
    """
    candidates: rows already selected as kaon candidates
    (pid == 321 & bdt_pass == True). Returns the fraction with
    rich_best_PID != 321.
    """

    numerator_df = candidates[candidates["rich_best_PID"] != 321]

    numerator = len(numerator_df)
    denominator = len(candidates)

    if denominator == 0:
        return 0

    return numerator / denominator


def contamination_for_threshold(rich_rq_threshold):
    """
    Returns a 2D list [theta_bin_index][p_bin_index] of
    contamination values, restricted to events passing the
    given rich_RQ threshold.
    """

    df_rich = df[
        (df["rich_best_ntot"] > 3)
        & (df["rich_RQ"] > rich_rq_threshold)
        & (df["rich_best_PID"] != -9999)
    ]

    thetaBins = au.makeBins(df_rich, "theta", binEdges=thetaEdges)

    results = []

    for thetaBin in thetaBins:
        pBins = au.makeBins(thetaBin, "p", binEdges=pEdges)

        rowResults = []
        for pBin in pBins:
            candidates = pBin[
                (pBin["pid"] == 321)
                & (pBin["bdt_pass"] == True)
            ]
            r = calc_contamination(candidates)
            rowResults.append(r)

        results.append(rowResults)

    return results


# ------------------------------------------------------------
# Compute contamination at both thresholds
# ------------------------------------------------------------

initial_results = contamination_for_threshold(RICH_RQ_INITIAL)
shifted_results = contamination_for_threshold(RICH_RQ_SHIFTED)

# ------------------------------------------------------------
# Assemble output rows
# ------------------------------------------------------------

rows = []

for i in range(THETA_NBINS):
    theta_lo, theta_hi = thetaEdges[i], thetaEdges[i + 1]

    for j in range(P_NBINS):
        p_lo, p_hi = pEdges[j], pEdges[j + 1]

        cont_init = initial_results[i][j]
        cont_shift = shifted_results[i][j]
        delta_c = abs(cont_shift - cont_init)

        rows.append({
            "theta_lo": theta_lo,
            "theta_hi": theta_hi,
            "p_lo": p_lo,
            "p_hi": p_hi,
            "contamination_initial": cont_init,
            "contamination_shifted": cont_shift,
            "delta_c": delta_c,
        })

out_df = pd.DataFrame(rows)
out_df.to_csv(outCSV, index=False)

print(f"Wrote {len(out_df)} rows to {outCSV}")
print(out_df)