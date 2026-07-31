from matplotlib.backends.backend_pdf import PdfPages
import argparse
import pathlib
import importlib
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd
import json
import uproot
import math

sys.path.append("../scripts/")
import common_functions as au


# -------------------------------------------------
# OPTIONAL: reload if developing
# import importlib
# importlib.reload(au)
# -------------------------------------------------


import numpy as np

def apply_bdt_cut(df, threshold_df):

    p = df["p"].to_numpy()
    theta = df["theta"].to_numpy()
    score = df["score"].to_numpy()

    pass_bdt = np.zeros(len(df), dtype=bool)

    for _, row in threshold_df.iterrows():

        if np.isnan(row["best_threshold"]):
            continue

        bin_mask = (
            (p >= row["p_low"]) & (p < row["p_high"]) &
            (theta >= row["theta_low"]) & (theta < row["theta_high"])
        )

        pass_bdt |= bin_mask & (score > row["best_threshold"])

    return pass_bdt


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def main():

    # -----------------------------
    # Load data + model
    # -----------------------------
    df_val = pd.read_parquet(
        "/volatile/clas12/cooperb/SULI/dataset_v03/val.parquet"
    )

    model_path = "/work/clas12/CooperBe/MLStuff/tier2All/model_v01/model.joblib"

    mod, mod_df = au.load_model_and_data(model_path, df_val)

    # -----------------------------
    # Binning setup
    # -----------------------------
    tStart = 5
    tEnd = 35
    tBinNum = 5

    pStart = 0.5
    pEnd = 5
    pBinNum = 10

    tBinEdges = np.linspace(tStart, tEnd, tBinNum + 1)
    pBinEdges = np.linspace(pStart, pEnd, pBinNum + 1)

    # -----------------------------
    # Threshold scan
    # -----------------------------
    thresholds = np.linspace(0.0, 0.95, 100)
    results = []

    thetaBins = au.makeBins(
        df=mod_df,
        variable="theta",
        binEdges=tBinEdges
    )

    for i in range(len(thetaBins)):

        pBins = au.makeBins(
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

            best_t = np.nan
            best_fom = -np.inf

            for t in thresholds:
                accepted = scores > t

                N_K = np.sum(accepted & is_K)
                N_pi = np.sum(accepted & is_pi)

                denom = np.sqrt(N_K + N_pi)

                if denom == 0:
                    continue

                fom = N_K / denom

                if fom > best_fom:
                    best_fom = fom
                    best_t = t

            results.append({
                "theta_low": tBinEdges[i],
                "theta_high": tBinEdges[i + 1],
                "p_low": pBinEdges[j],
                "pHigh": pBinEdges[j + 1],
                "best_threshold": best_t,
                "best_fom": best_fom
            })

    # -----------------------------
    # Save results
    # -----------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "../figures/optimized_thresholdsV5.csv",
        index=False
    )

    # -----------------------------
    # Apply model + cut
    # -----------------------------
    df_test = pd.read_parquet(
        "/volatile/clas12/cooperb/SULI/dataset_v02/test.parquet"
    )

    feature_names = au.get_feature_names(model_path)

    df_test = au.apply_model_to_df(mod, df_test, feature_names)

    mask = apply_bdt_cut(df_test, results_df)

    print("created mask")

    # -----------------------------
    # Efficiency
    # -----------------------------
    vals, errs = au.compute_efficiency(df_test, cut=mask)

    # -----------------------------
    # Plot
    # -----------------------------
    p_centers = (pBinEdges[:-1] + pBinEdges[1:]) / 2

    plt.figure(figsize=(8, 6))

    for i in range(tBinNum):

        plt.errorbar(
            p_centers,
            vals[i],
            yerr=errs[i],
            marker="o",
            capsize=3,
            label=fr"$\theta \in [{tBinEdges[i]:.1f},{tBinEdges[i+1]:.1f}]$"
        )

    plt.xlabel("Momentum (GeV/c)")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Momentum (by Theta bin)")
    plt.grid(True)
    plt.legend()

    plt.savefig("../figures/efficiency_vs_p_TUNED.png", dpi=150)
    plt.close()


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    main()