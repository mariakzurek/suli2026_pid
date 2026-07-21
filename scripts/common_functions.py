from matplotlib.backends.backend_pdf import PdfPages
import argparse
import pathlib


import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import json
import math


import sys
from pathlib import Path
from baseline_chi2pid import passes_kplus_chi2pid_cut



##############################################################################################################
#
#   READ ME
#
#   This script does not do anything on it's own, it is a list of functions that are useful in analysis
#   to use these functions do "from functions.py import [name of desired function]" When performing you're imports
# 
#   WHAT THIS PROGRAM CONTAINS
#         -functions to compute purity, contamination, efficiency, and Mis-ID
#         -functions to automatically create bins on data
#         -function that will apply basic SIDIS analysis cuts
#         -function that assist plotting
#         -function tat Load BDT ML model and adds score to df
#         -functions that apply threshold optimizations on feature of merit, additionally a function that applies this cut
#         -functions that can get information on a model
#         -functions used to apply a binary BDT cut that matches the efficiency of the baseline method (chi2pid)
#
#
##############################################################################################################


def compute_contamination(df, pid=None):
    """
    Computes contamination among reconstructed K+ candidates.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    pid : int or None, optional
        If None, computes total contamination
        (all non-321 truth particles reconstructed as K+).
        Otherwise computes the contamination contribution
        from the specified MC PID.

    Returns
    -------
    r : float
        Contamination fraction.
    rErr : float
        Statistical uncertainty.
    """

    temp = df[df["pid"] == 321]

    if pid is None:
        # Total contamination
        a = (temp["mc_matching_pid"] != 321).sum()
    else:
        # Contribution from one particle species
        a = (temp["mc_matching_pid"] == pid).sum()

    b = len(temp)

    r = 0.0
    rErr = 99.0

    if b != 0:
        r = a / b

    if a != 0:
        rErr = r * math.sqrt((1 / a) + (1 / b))

    return r, rErr

def compute_purity(df):
    #Computes the purity of the sample
    temp=df[df["pid"]==321]
    a= (temp["mc_matching_pid"]==321).sum()
    b= temp["pid"].sum()
    r=0
    rErr=99
    if b!=0:
        r=a/b
    if a!=0:
        rErr = r*math.sqrt((1/a)+(1/b))
    return r, rErr

def compute_efficiency(df, cut, raw=False):
    #Will compute efficiency, raw gives the raw, uncut efficiency (PID efficiency)
    #while cut a cut you want to test the efficiency of (must be masked cut)
    if not raw:
        temp = df[df["pid"]==321]
    else:
        temp=df
    up = temp[cut]
    a= (up["mc_matching_pid"]==321).sum()
    b= (temp["mc_matching_pid"]==321).sum()
    r=0
    rErr=99
    if b!=0:
        r=a/b
    if a!=0:
        rErr = r*math.sqrt((1/a)+(1/b))
    return r, rErr

def compute_Mis_ID(df, cut, raw):
    #Will compute Mis-ID, raw gives the raw, uncut Mis-ID (PID Mis-ID)
    #while cut a cut you want to test the Mis-ID of (must be masked cut)
    if not raw:
        temp = df[df["pid"]==321].copy()
        up = temp[~cut].copy()
    else:
        temp=df
        up= temp[(temp["pid"]!=321)&(temp["mc_matching_pid"]==321)]
    
    a= (up["mc_matching_pid"]==321).sum()
    b= (temp["mc_matching_pid"]==321).sum()
    r=0
    rErr=99
    if b!=0:
        r=a/b
    if a!=0:
        rErr = r*math.sqrt((1/a)+(1/b))
    return r, rErr
    
def makeBins(df, variable, binEdges=None, start=None, end=None, binNum=None):
    #This function will return a list of dataframes that are binned on variable(string) you can give an array of bin edges or the start,end,bin number to generate these 
    #(DO NOT TRY TO DO BOTH AT THE SAME TIME)
    bins =[]
    if binEdges is None:
        step=(end-start)/binNum
        for i in range(binNum):
            binCut=df[(df[variable]>=(start+(i*step)))&(df[variable]<(start+(i+1)*step))]
            bins.append(binCut)
    else:
        for i in range(len(binEdges)-1):
            binCut=df[(df[variable]>=binEdges[i])&(df[variable]<=binEdges[i+1])]
            bins.append(binCut)
    return bins

def makeBinEdges(start,end,bins):
    #auto generates bin edges for uniform binning with a start,end, and bin number
    binEdges = np.linspace(start, end, bins + 1)
    return binEdges


def apply_Sidis_Cuts(df):
    #Applies all Basic SIDIS Cuts used in analysis
    baseline=df[
    (df["Q2"]>2)&
    (df["W"]>2)&
    ((df["y"]>0)&(df["y"]<0.75))
    ]
    baseline=baseline[
        ((baseline["pid"]==321)&(baseline["Mx_eKX"]>1.6))|
        ((baseline["pid"]==211)&(baseline["Mx_epiX"]>1.5))|
        ((baseline["pid"]==2212)&(baseline["Mx_epX"]>1))]
    return baseline



##########################################################################################




def add_plot(vals, errs, ax, label, center):
    #returns ax with the plot of some values errors ect, useful for when wanting to make a comparison of two plots
    ax.errorbar(
        center,
        vals,
        yerr=errs,
        marker="o",
        capsize=3,
        label=label,
        linestyle="none"
    )
    return ax

    




import joblib
import json
import pathlib
import numpy as np




##################################################################################################################################

#These functions handel loading and using the functionality of the Binary BDT Classifier

def load_model_and_data(model_path, df):
    #Loads a model using a path and dataframe. It will return the model obejct and the input dataframe with the scores column
    
    # -------------------------------------------------
    # 1. LOAD MODEL
    # -------------------------------------------------
    print(f"Loading model: {model_path}")
    model_obj = joblib.load(str(model_path))

    if isinstance(model_obj, dict) and "model" in model_obj:
        model = model_obj["model"]
        feature_names = model_obj["features"]
        print(f"Loaded wrapped model with {len(feature_names)} features")

    else:
        model = model_obj

        manifest_path = pathlib.Path(model_path).resolve().parents[0] / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json at {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        feature_names = manifest.get("feature_list") or manifest.get("columns")

        if feature_names is None:
            raise ValueError("manifest.json missing feature_list/columns")

        print(f"Loaded legacy model with {len(feature_names)} features")

    # -------------------------------------------------
    # 2. SCORE MODEL
    # -------------------------------------------------
    X = df[feature_names].to_numpy(dtype=np.float32)

    scores = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["score"] = scores

    return model, df


def get_feature_names(model_path):
    """
    Extract feature names from a saved model or manifest.
    Does NOT load or use any dataframe.
    """

    print(f"Reading model metadata: {model_path}")

    model_obj = joblib.load(str(model_path))

    # -------------------------------------------------
    # Wrapped model case
    # -------------------------------------------------
    if isinstance(model_obj, dict) and "model" in model_obj:
        feature_names = model_obj["features"]
        print(f"Wrapped model: {len(feature_names)} features")

    # -------------------------------------------------
    # Legacy model + manifest case
    # -------------------------------------------------
    else:
        manifest_path = pathlib.Path(model_path).resolve().parents[0] / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json at {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        feature_names = manifest.get("feature_list") or manifest.get("columns")

        if feature_names is None:
            raise ValueError("manifest.json missing feature_list/columns")

        print(f"Legacy model: {len(feature_names)} features")

    return feature_names

def apply_model_to_df(model, df, feature_names):
    #adds the scores column to a dataframe.

    X = df.loc[:, feature_names].values.astype(np.float32, copy=False)

    scores = model.predict_proba(X)[:, 1]

    df["score"] = scores
    return df


def optimizeFOM(model_df, tBinEdges, pBinEdges, outputCSV=None, deviation=0):
    #This program will find the FOM optimization of the threshold, you can optionally tell it to save as a CSV, but it will also return things as a dataframe, the deviation means that it will take the highest BDT threshold that is within that deviation so deviation 0.01 tells it to take the highest threshold such that FOM> 99% of Max FOM, useful for if the FOM versus BDT threshold plateus.
    
    if not (0 <= deviation < 1):
        raise ValueError("deviation must satisfy 0 <= deviation < 1.")

    thresholds = np.linspace(0.0, 0.95, 100)
    results = []

    thetaBins = makeBins(
        df=model_df,
        variable="theta",
        binEdges=tBinEdges
    )

    for i in range(len(thetaBins)):

        pBins = makeBins(
            df=thetaBins[i],
            variable="p",
            binEdges=pBinEdges
        )

        for j in range(len(pBins)):

            df_bin = pBins[j]

            if len(df_bin) == 0:
                continue

            mc = df_bin["mc_matching_pid"].to_numpy()
            scores = df_bin["score"].to_numpy()

            is_K = (mc == 321)
            is_pi = (mc == 211)

            if np.sum(is_K) == 0:
                continue

            fom_values = []

            # ---------------------------------------------
            # Calculate FOM for every threshold
            # ---------------------------------------------
            for t in thresholds:

                accepted = scores > t

                N_K = np.sum(accepted & is_K)
                N_pi = np.sum(accepted & is_pi)

                denom = np.sqrt(N_K + N_pi)

                if denom == 0:
                    fom = 0.0
                else:
                    fom = N_K / denom

                fom_values.append(fom)

            fom_values = np.array(fom_values)

            # ---------------------------------------------
            # Choose optimal threshold
            # ---------------------------------------------
            max_fom = np.max(fom_values)

            if deviation == 0:
                # Original behavior: first threshold with max FOM
                best_t = thresholds[np.argmax(fom_values)]
            else:
                tolerance = 1 - deviation

                allowed_thresholds = thresholds[
                    fom_values >= tolerance * max_fom
                ]

                if len(allowed_thresholds) > 0:
                    # Highest threshold within allowed deviation
                    best_t = allowed_thresholds[-1]
                else:
                    best_t = np.nan

            results.append({
                "thetaLow": tBinEdges[i],
                "thetaHigh": tBinEdges[i + 1],
                "pLow": pBinEdges[j],
                "pHigh": pBinEdges[j + 1],
                "best_threshold": best_t,
                "best_fom": max_fom
            })

    results_df = pd.DataFrame(results)

    if outputCSV is not None:
        from pathlib import Path
        output_path = Path(outputCSV)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
    return results_df


def apply_optimized_bdt_cut(df, threshold_df=None, CSVPath=None):
    """
    Apply optimized BDT thresholds to a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing 'p', 'theta', and 'score'.

    threshold_df : pandas.DataFrame, optional
        DataFrame containing optimized thresholds.

    CSVPath : str or Path, optional
        Path to a CSV containing optimized thresholds. Used if
        threshold_df is not provided.

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

        bin_mask = (
            (p >= row["pLow"]) & (p < row["pHigh"]) &
            (theta >= row["thetaLow"]) & (theta < row["thetaHigh"])
        )

        pass_bdt |= bin_mask & (score > row["best_threshold"])

    return pass_bdt




#################################################################################################3
#THESE ARE FOR MATCHING BASELINE EFFICIENCIES


def computeEffChiPID(df, pBinEdges): #Standalone function for doing the efficiency using the baseline chi2pid cut
    import numpy as np

    # -------------------------------------------------
    # keep only valid MC-matched tracks
    # -------------------------------------------------
    df = df[df["mc_matching_pid"] != -9999].copy()

    eff = []

    for i in range(len(pBinEdges) - 1):

        p_lo = pBinEdges[i]
        p_hi = pBinEdges[i + 1]

        binp = df[
            (df["p"] >= p_lo) &
            (df["p"] < p_hi)
        ]

        if len(binp) == 0:
            eff.append(np.nan)
            continue

        # -------------------------------------------------
        # TRUE K+ (DENOMINATOR)
        # -------------------------------------------------
        true_k = (binp["mc_matching_pid"] == 321)
        n_true_k = np.sum(true_k)

        if n_true_k == 0:
            eff.append(np.nan)
            continue

        # -------------------------------------------------
        # χ²PID CUT
        # -------------------------------------------------
        chi2_mask = passes_kplus_chi2pid_cut(
            binp["chi2pid"].to_numpy(),
            binp["p"].to_numpy()
        )

        # -------------------------------------------------
        # NUMERATOR:
        # true K+ that pass χ²PID and are reconstructed as K+
        # -------------------------------------------------
        reconstructed_k = (binp["pid"] == 321)

        passed = chi2_mask & reconstructed_k.to_numpy() & true_k.to_numpy()

        eff.append(np.sum(passed) / n_true_k)

    return np.array(eff, dtype=float)

def MatchEfficiency(df, pBinEdges):    #This is a BDT cut that Matches th efficiency of the chi2pid baseline method
    import numpy as np

    # Remove unmatched particles
    df = df[df["mc_matching_pid"] != -9999].copy()

    # Target χ²PID efficiencies
    effs = computeEffChiPID(df, pBinEdges)

    thresholds_out = []
    pass_bdt = np.zeros(len(df), dtype=bool)

    scores = df["score"].to_numpy()
    p = df["p"].to_numpy()

    for i in range(len(pBinEdges) - 1):

        if np.isnan(effs[i]):
            thresholds_out.append(np.nan)
            continue

        p_lo = pBinEdges[i]
        p_hi = pBinEdges[i + 1]

        bin_mask = (p >= p_lo) & (p < p_hi)

        if np.sum(bin_mask) == 0:
            thresholds_out.append(np.nan)
            continue

        binp = df.loc[bin_mask]

        true_k = (binp["mc_matching_pid"] == 321).to_numpy()
        n_true_k = np.sum(true_k)

        if n_true_k == 0:
            thresholds_out.append(np.nan)
            continue

        bin_scores = binp["score"].to_numpy()
        target_eff = effs[i]

        thresholds = np.linspace(0.01, 0.99, 200)

        best_t = np.nan
        best_diff = np.inf

        for t in thresholds:

            accepted = bin_scores > t
            eff = np.sum(accepted & true_k) / n_true_k

            diff = abs(eff - target_eff)

            if diff < best_diff:
                best_diff = diff
                best_t = t

        thresholds_out.append(best_t)

        # Apply threshold to all events in this momentum bin
        if not np.isnan(best_t):
            pass_bdt[bin_mask] = scores[bin_mask] > best_t

    return np.array(thresholds_out), pass_bdt


