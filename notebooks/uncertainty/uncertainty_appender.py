#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib.backends.backend_pdf import PdfPages
import argparse
import pathlib
import importlib

import joblib
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import awkward as ak
import json
import math
from scipy.integrate import quad
from scipy.optimize import curve_fit
import uproot
import sys
sys.path.append("../../scripts/")

from pathlib import Path
import common_functions as au
from baseline_chi2pid import passes_kplus_chi2pid_cut


# In[2]:


def apply_shift(df, threshold_df=None, CSVPath=None, shift=0):
    """
    Apply optimized BDT thresholds to a dataframe with an optional threshold shift.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing 'p', 'theta', and 'score'.

    threshold_df : pandas.DataFrame, optional
        DataFrame containing optimized thresholds.

    CSVPath : str or Path, optional
        Path to a CSV containing optimized thresholds. Used if
        threshold_df is not provided.

    shift : float, optional
        Amount to shift each threshold by. Positive values make the cut
        tighter; negative values make the cut looser.

    Returns
    -------
    numpy.ndarray
        Boolean mask indicating which events pass the optimized cut.
    """

    # Load thresholds from CSV if needed
    if threshold_df is None:
        if CSVPath is None:
            raise ValueError("Either threshold_df or CSVPath must be provided.")

        threshold_df = pd.read_csv(Path(CSVPath))

    p = df["p"].to_numpy()
    theta = df["theta"].to_numpy()
    score = df["score"].to_numpy()

    pass_bdt = np.zeros(len(df), dtype=bool)

    for _, row in threshold_df.iterrows():

        if np.isnan(row["best_threshold"]):
            continue

        # Apply shift to threshold
        threshold = row["best_threshold"] + shift

        bin_mask = (
            (p >= row["p_low"]) & (p < row["p_high"]) &
            (theta >= row["theta_low"]) & (theta < row["theta_high"])
        )

        pass_bdt |= bin_mask & (score > threshold)

    return pass_bdt


# In[3]:


import pandas as pd
import math

df = pd.read_csv("../../figures/full_range/evaluate_outputs/per_bin_sweep.csv")

# Add event number for future reference
df["event_number"] = range(len(df))

# Lists to hold new values
stat_unc_values = []
denominator_values = []

# Loop over each row
for _, row in df.iterrows():

    # Read values
    C_p = row["C_p"]
    C_pi = row["C_pi"]
    n_accepted = row["n_accepted"]

    # Denominator used for C_pi:
    # N(K->K) + N(pi->K) = n_accepted * (1 - C_p)
    denominator = n_accepted * (1 - C_p)
    if not np.isnan(denominator):
        denominator = round(denominator)

    # Save denominator
    denominator_values.append(denominator)

    # Calculate statistical uncertainty
    if denominator > 0 and not math.isnan(C_pi):
        result = math.sqrt(C_pi * (1 - C_pi) / denominator)
    else:
        result = 9999

    stat_unc_values.append(result)

# Append new columns
df["C_pi_denominator"] = denominator_values
df["stat_unc"] = stat_unc_values

# Save updated CSV
df.to_csv(
    "../../figures/full_range/evaluate_outputs/per_bin_sweep_uncertianties.csv",
    index=False
)


# In[ ]:


df_val = pd.read_parquet(
        "~/ML_Files/dataset_v03/val.parquet"
    )

model_path = "/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib"

mod, mod_df = au.load_model_and_data(model_path, df_val)

df_test=pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/test.parquet")
df_test=df_test[df_test["mc_matching_pid"]!=-9999]
feature_names=au.get_feature_names("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib")
print(feature_names)
df_test = au.apply_model_to_df(mod, df_test, feature_names)


# In[ ]:




# In[ ]:


csvLocation = "../../figures/full_range/optimized_thresholdsV4.csv"
csvFile = csvLocation

if Path(csvLocation).exists():
    print(f"Loading optimized thresholds from {csvLocation}")
    results_df = pd.read_csv(csvLocation)
else:
    print("Optimized threshold file not found. Running optimization...")
    results_df = au.optimizeFOM(
        mod_df,
        tBinEdges,
        pBinEdges,
        outputCSV=csvLocation,
        deviation=0.03
    )


# ---------------------------------------------------------
# Rename threshold columns
# ---------------------------------------------------------

threshold_df = results_df.rename(
    columns={
        "pLow": "p_low",
        "pHigh": "p_high",
        "thetaLow": "theta_low",
        "thetaHigh": "theta_high",
    }
)[
    [
        "p_low",
        "p_high",
        "theta_low",
        "theta_high",
        "best_threshold",
    ]
].copy()


# ---------------------------------------------------------
# Compute sensitivity per p-theta bin
# ---------------------------------------------------------

rows = []

for _, row in threshold_df.iterrows():

    p_low = row["p_low"]
    p_high = row["p_high"]
    theta_low = row["theta_low"]
    theta_high = row["theta_high"]

    # Select the same p-theta bin used for optimization
    bin_mask = (
        (df_test["p"] >= p_low) &
        (df_test["p"] < p_high) &
        (df_test["theta"] >= theta_low) &
        (df_test["theta"] < theta_high)
    )

    df_bin = df_test[bin_mask].copy()

    if len(df_bin) == 0:
        continue


    # -----------------------------------------------------
    # Nominal contamination
    # -----------------------------------------------------

    # Only use this threshold row
    this_threshold = threshold_df[
        (threshold_df["p_low"] == p_low) &
        (threshold_df["p_high"] == p_high) &
        (threshold_df["theta_low"] == theta_low) &
        (threshold_df["theta_high"] == theta_high)
    ]

    base_mask = au.apply_optimized_bdt_cut(
        df_bin,
        threshold_df=this_threshold
    )

    C_base, _ = au.compute_contamination(
        df_bin[base_mask]
    )


    # -----------------------------------------------------
    # +0.05 threshold shift
    # -----------------------------------------------------

    plus_mask = apply_shift(
        df_bin,
        threshold_df=this_threshold,
        shift=0.05
    )

    C_plus, _ = au.compute_contamination(
        df_bin[plus_mask]
    )


    # -----------------------------------------------------
    # -0.05 threshold shift
    # -----------------------------------------------------

    minus_mask = apply_shift(
        df_bin,
        threshold_df=this_threshold,
        shift=-0.05
    )

    C_minus, _ = au.compute_contamination(
        df_bin[minus_mask]
    )


    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    rows.append({
        "p_low": p_low,
        "p_high": p_high,
        "theta_low": theta_low,
        "theta_high": theta_high,
        "best_threshold": row["best_threshold"],

        "C_base": C_base,
        "C_plus": C_plus,
        "C_minus": C_minus,

        "delta_c_plus": abs(C_plus - C_base),
        "delta_c_minus": abs(C_minus - C_base),
        "delta_c_total": abs(C_plus - C_minus),

        "n_events": len(df_bin),
    })


# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------

sensitivity_df = pd.DataFrame(rows)

output = "../../figures/full_range/threshold_sensitivity.csv"

sensitivity_df.to_csv(
    output,
    index=False
)

print(f"Saved: {output}")

