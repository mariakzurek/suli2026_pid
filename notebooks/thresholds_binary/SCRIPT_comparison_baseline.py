from matplotlib.backends.backend_pdf import PdfPages
import argparse
import pathlib
import importlib

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import json
import uproot


import sys
sys.path.append("../../scripts/")

from pathlib import Path
import common_functions as au
from baseline_chi2pid import passes_kplus_chi2pid_cut

#importlib.reload(au)


df_val = pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/val.parquet")

#df_val=au.apply_Sidis_Cuts(df_val)
mod, mod_df = au.load_model_and_data("/work/clas12/CooperBe/MLStuff/tier2All/model_MLP_binary/model.joblib", df_val)
allPlots=[]


tStart = 5
tEnd = 35
tBinNum = 5

pStart = 0.5
pEnd = 5
pBinNum = 10

tBinEdges = np.linspace(tStart, tEnd, tBinNum + 1)
pBinEdges = np.linspace(pStart, pEnd, pBinNum + 1)


csvFile = "../../figures/binary_MLP/comparisons/optimized_thresholds_MLP.csv"

if Path(csvFile).exists():
    print(f"Loading optimized thresholds from {csvFile}")
    results_df = pd.read_csv(csvFile)
else:
    print("Optimized threshold file not found. Running optimization...")
    results_df = au.optimizeFOM(
        mod_df,
        tBinEdges,
        pBinEdges,
        outputCSV=csvFile,
        deviation=0.03
    )


df_test=pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/test.parquet")
df_test=df_test[df_test["mc_matching_pid"]!=-9999]
feature_names=au.get_feature_names("/work/clas12/CooperBe/MLStuff/tier2All/model_MLP_binary/model.joblib")
print(feature_names)
df_test = au.apply_model_to_df(mod, df_test, feature_names)


bdtMask=au.apply_optimized_bdt_cut(df_test, threshold_df=results_df)
chi2pidMask=au.MatchEfficiency(df_test, pBinEdges)
thetaMask = (df_test["theta"]>5)&(df_test["theta"]<=11)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# BDT mask on full dataframe
bdtMask = au.apply_optimized_bdt_cut(
    df_test,
    threshold_df=results_df
)

thetaMask = (
    (df_test["theta"] > 5) &
    (df_test["theta"] <= 11)
).to_numpy()


bdt_vals = []
bdt_errs = []

chi_vals = []
chi_errs = []


# Loop over momentum bins
for i in range(len(pBinEdges)-1):

    p_low = pBinEdges[i]
    p_high = pBinEdges[i+1]

    # First make p-bin subset
    pMask = (
        (df_test["p"].to_numpy() >= p_low) &
        (df_test["p"].to_numpy() < p_high)
    )

    df_pbin = df_test[pMask].copy()

    # Corresponding masks inside p bin
    bdt_bin_mask = bdtMask[pMask]
    theta_bin_mask = thetaMask[pMask]

    # Apply theta baseline
    df_pbin = df_pbin[theta_bin_mask]

    # Apply same theta selection to BDT mask
    bdt_bin_mask = bdt_bin_mask[theta_bin_mask]


    # ---------------------------------------------
    # Direct chi2pid cut on this p/theta subset
    # ---------------------------------------------
    chi_bin_mask = passes_kplus_chi2pid_cut(
        df_pbin["chi2pid"].to_numpy(),
        df_pbin["p"].to_numpy()
    )


    # Compute efficiencies
    bdt_eff, bdt_err = au.compute_efficiency(
        df_pbin,
        cut=bdt_bin_mask
    )

    chi_eff, chi_err = au.compute_efficiency(
        df_pbin,
        cut=chi_bin_mask
    )


    bdt_vals.append(bdt_eff)
    bdt_errs.append(bdt_err)

    chi_vals.append(chi_eff)
    chi_errs.append(chi_err)


bdt_vals = np.array(bdt_vals)
bdt_errs = np.array(bdt_errs)

chi_vals = np.array(chi_vals)
chi_errs = np.array(chi_errs)


# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8,6))

p_centers = (pBinEdges[:-1] + pBinEdges[1:]) / 2

ax.errorbar(
    p_centers,
    bdt_vals,
    yerr=bdt_errs,
    marker="o",
    capsize=3,
    label="Optimized BDT",
    linestyle="none"
)

ax.errorbar(
    p_centers,
    chi_vals,
    yerr=chi_errs,
    marker="o",
    capsize=3,
    label=r"Basline chi2pid Cut",
    linestyle="none"
)

ax.set_xlabel("Momentum (GeV/c)")
ax.set_ylabel("Efficiency")
ax.set_title(r"Efficiency Comparison ($5 < theta \leq 11$)")

ax.set_ylim(0, 1.1)
ax.set_xlim(pBinEdges[0], pBinEdges[-1])
ax.set_xticks(pBinEdges)

ax.grid(False)
ax.legend()


outpath = Path(
    "../../figures/binary_MLP/comparisons/efficiency_BDT_vs_chi2pid.png"
)

outpath.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    outpath,
    dpi=150,
    bbox_inches="tight"
)


plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# BDT mask on full dataframe
bdtMask = au.apply_optimized_bdt_cut(
    df_test,
    threshold_df=results_df
)

thetaMask = (
    (df_test["theta"] > 5) &
    (df_test["theta"] <= 11)
).to_numpy()


bdt_vals = []
bdt_errs = []

chi_vals = []
chi_errs = []


# Loop over momentum bins
for i in range(len(pBinEdges)-1):

    p_low = pBinEdges[i]
    p_high = pBinEdges[i+1]

    # First make p-bin subset
    pMask = (
        (df_test["p"].to_numpy() >= p_low) &
        (df_test["p"].to_numpy() < p_high)
    )

    df_pbin = df_test[pMask].copy()

    # Corresponding masks inside p bin
    bdt_bin_mask = bdtMask[pMask]
    theta_bin_mask = thetaMask[pMask]

    # Apply theta baseline
    df_pbin = df_pbin[theta_bin_mask]

    # Apply same theta selection to BDT mask
    bdt_bin_mask = bdt_bin_mask[theta_bin_mask]


    # ---------------------------------------------
    # Direct chi2pid cut on this p/theta subset
    # ---------------------------------------------
    chi_bin_mask = passes_kplus_chi2pid_cut(
        df_pbin["chi2pid"].to_numpy(),
        df_pbin["p"].to_numpy()
    )


    # Create selected subsets
    bdt_subset = df_pbin[bdt_bin_mask]
    chi_subset = df_pbin[chi_bin_mask]


    # Compute contamination
    bdt_cont, bdt_err = au.compute_contamination(
        bdt_subset
    )

    chi_cont, chi_err = au.compute_contamination(
        chi_subset
    )


    bdt_vals.append(bdt_cont)
    bdt_errs.append(bdt_err)

    chi_vals.append(chi_cont)
    chi_errs.append(chi_err)


bdt_vals = np.array(bdt_vals)
bdt_errs = np.array(bdt_errs)

chi_vals = np.array(chi_vals)
chi_errs = np.array(chi_errs)


# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8,6))

p_centers = (pBinEdges[:-1] + pBinEdges[1:]) / 2

ax.errorbar(
    p_centers,
    bdt_vals,
    yerr=bdt_errs,
    marker="o",
    capsize=3,
    label="Optimized BDT",
    linestyle="none"
)

ax.errorbar(
    p_centers,
    chi_vals,
    yerr=chi_errs,
    marker="o",
    capsize=3,
    label=r"baseline chi2pid cut",
    linestyle="none"
)


ax.set_xlabel("Momentum (GeV/c)")
ax.set_ylabel("Contamination")
ax.set_title(
    r"Contamination Comparison ($5 < theta \leq 11$)"
)

ax.set_ylim(0, 0.5)
ax.set_xlim(pBinEdges[0], pBinEdges[-1])
ax.set_xticks(pBinEdges)

ax.grid(False)
ax.legend()


# Save
outpath = Path(
    "../../figures/binary_MLP/comparisons/contamination_BDT_vs_chi2pid.png"
)

outpath.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    outpath,
    dpi=150,
    bbox_inches="tight"
)


plt.close(fig)