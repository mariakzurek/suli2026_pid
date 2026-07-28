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


# ----------------------------
# Find ROOT files
# ----------------------------

data_dir = "/work/clas12/CooperBe/MLStuff/scored_data_v02/"

files = sorted(
    glob.glob(data_dir + "*.root")
)


if len(files) == 0:
    raise FileNotFoundError(
        "No ROOT files found"
    )


# take requested number of files

N_FILES=2
files_to_use = files[:N_FILES]


print(f"Found {len(files)} ROOT files")
print(f"Using {len(files_to_use)} ROOT files")


# ----------------------------
# Load files
# ----------------------------

dfs = []

for f in files_to_use:

    print("Opening:", f)

    df_temp = uproot.open(
        f + ":PhysicsEvents"
    ).arrays(
        cols
    )

    # ----------------------------
    # Convert to pandas if needed
    # ----------------------------

    if isinstance(df_temp, ak.Array):

        df_temp = ak.to_dataframe(
            df_temp
        )

        # uproot/awkward can create a multi-index
        # for nested structures, remove it
        df_temp = df_temp.reset_index(drop=True)


    elif not isinstance(df_temp, pd.DataFrame):

        raise TypeError(
            f"Unknown data type: {type(df_temp)}"
        )


    dfs.append(df_temp)


# combine all files
df = pd.concat(
    dfs,
    ignore_index=True
)

cols.append("mc_matching_pid")
df_mc = uproot.open("~/ML_Files/MC_scored/pid_training_v2.root:PhysicsEvents").arrays(cols, library="pd")


# ---------------------------------------------------------
# Load or create p-theta weighting map
# ---------------------------------------------------------

old_csv = "../../figures/feature_audit_week3/kp/feature_audit_summary.csv"
csvOut = "weighting.csv"


if Path(csvOut).exists():

    print(f"Loading existing weighting map: {csvOut}")

    weighting_df = pd.read_csv(csvOut)


else:

    print("Weighting map not found. Creating from data/MC...")


    # -----------------------------------------------------
    # Load old CSV only for bin definitions
    # -----------------------------------------------------

    old_df = pd.read_csv(old_csv)

    weighting_df = old_df[
        [
            "p_lo",
            "p_hi",
            "theta_lo",
            "theta_hi",
        ]
    ].drop_duplicates().copy()


    n_data_list = []
    n_mc_list = []
    weight_list = []


    # -----------------------------------------------------
    # Compute data/MC ratio per bin
    # -----------------------------------------------------

    for _, row in weighting_df.iterrows():

        p_lo = row["p_lo"]
        p_hi = row["p_hi"]

        theta_lo = row["theta_lo"]
        theta_hi = row["theta_hi"]


        # Data selection
        data_mask = (
            (df["p"] >= p_lo) &
            (df["p"] < p_hi) &
            (df["theta"] >= theta_lo) &
            (df["theta"] < theta_hi)
        )

        df_data_bin = df[data_mask]


        # MC selection
        mc_mask = (
            (df_mc["p"] >= p_lo) &
            (df_mc["p"] < p_hi) &
            (df_mc["theta"] >= theta_lo) &
            (df_mc["theta"] < theta_hi)
        )

        df_mc_bin = df_mc[mc_mask]


        # Use len() explicitly
        n_data = len(df_data_bin)
        n_mc = len(df_mc_bin)


        n_data_list.append(n_data)
        n_mc_list.append(n_mc)


        if n_mc > 0:
            weight_list.append(n_data / n_mc)
        else:
            weight_list.append(0.0)


    weighting_df["n_data"] = n_data_list
    weighting_df["n_mc"] = n_mc_list
    weighting_df["weight"] = weight_list


    # Save so it does not need to be recalculated
    weighting_df.to_csv(
        csvOut,
        index=False
    )

    print(f"Saved new weighting map: {csvOut}")



###########################################################################################################


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

################################################################################################################

# ---------------------------------------------------------
# Weighted contamination
# ---------------------------------------------------------

def compute_weighted_contamination(df):

    weight = df["weight"].to_numpy()

    # MC truth labels
    is_pi = df["mc_matching_pid"].to_numpy() == 211
    is_k  = df["mc_matching_pid"].to_numpy() == 321
    is_p  = df["mc_matching_pid"].to_numpy() == 2212

    # Weighted yields
    pi_to_k = weight[is_pi].sum()
    k_to_k  = weight[is_k].sum()
    p_to_k  = weight[is_p].sum()

    denom = k_to_k + pi_to_k + p_to_k

    if denom > 0:
        return pi_to_k / denom
    else:
        return np.nan



# ---------------------------------------------------------
# Compute contamination comparison
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

rows = []

for _, row in weighting_df.iterrows():

    p_lo = row["p_lo"]
    p_hi = row["p_hi"]
    theta_lo = row["theta_lo"]
    theta_hi = row["theta_hi"]


    # ---------------------------------------------------------
    # p-theta subset
    # ---------------------------------------------------------

    bin_mask = (
        (df_test["p"] >= p_lo) &
        (df_test["p"] < p_hi) &
        (df_test["theta"] >= theta_lo) &
        (df_test["theta"] < theta_hi)
    )

    df_bin = df_test[bin_mask].copy()


    if len(df_bin) == 0:
        continue


    # ---------------------------------------------------------
    # Apply optimized BDT threshold
    # ---------------------------------------------------------

    bdt_mask = au.apply_optimized_bdt_cut(
        df_bin,
        threshold_df=threshold_df
    )


    # Keep all truth species that pass the BDT
    # (K, pi, and p are needed for contamination)
    df_pass = df_bin[bdt_mask].copy()


    if len(df_pass) == 0:
        continue


    # ---------------------------------------------------------
    # Unweighted contamination
    # ---------------------------------------------------------

    C_unweighted, _ = au.compute_contamination(
        df_pass
    )


    # ---------------------------------------------------------
    # Weighted contamination
    # ---------------------------------------------------------

    # Same p-theta bin, so every event gets the same weight
    df_pass["weight"] = row["weight"]

    C_weighted = compute_weighted_contamination(
        df_pass
    )


    # ---------------------------------------------------------
    # Percent difference
    # ---------------------------------------------------------

    if C_unweighted != 0 and not np.isnan(C_unweighted):

        percent_diff = (
            abs(C_weighted - C_unweighted)
            / C_unweighted
            * 100
        )

    else:

        percent_diff = np.nan


    rows.append({

        "p_lo": p_lo,
        "p_hi": p_hi,
        "theta_lo": theta_lo,
        "theta_hi": theta_hi,

        "weight": row["weight"],

        "n_events": len(df_bin),
        "n_pass": len(df_pass),

        "C_unweighted": C_unweighted,
        "C_weighted": C_weighted,

        "percent_difference": percent_diff,
        "C_difference": abs(C_unweighted-C_weighted)
    })


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

contamination_df = pd.DataFrame(rows)

outFile = "weighted_contamination_comparison.csv"

contamination_df.to_csv(
    outFile,
    index=False
)

print(f"Saved {outFile}")





