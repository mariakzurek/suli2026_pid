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


cols = ["pid", "p", "theta", "beta", "chi2pid", "rich_RQ", "vz", "bdt_pass", "rich_best_PID", "rich_RQ", "rich_best_ntot", "bdt_score"]
kinematics =["Mx_eKX","Mx_epiX","Mx_epX", "Q2", "W", "y"]

for kin in kinematics:
    cols.append(kin)
    
df = uproot.open("~/ML_Files/scored_data_v01/nSidis_005046.root:PhysicsEvents").arrays(cols, library="pd")
df=df[df["rich_best_ntot"]>3]
df=df[df["rich_RQ"]>0.2]
df=df[df["rich_best_PID"]!=-9999]
df=au.apply_Sidis_Cuts(df)

outDir="../../figures/Data_Application/contams/"


# In[3]:


import math
import numpy as np
import matplotlib.pyplot as plt

# Define momentum bins
pEdges = au.makeBinEdges(2.5, 5, 10)

# Select RICH angular range
df_rich = df[df["theta"] < 11.25]
# = df_rich[
 #   (df_rich["rich_best_PID"] == 321) |
  #  (df_rich["rich_best_PID"] == 221) |
   # (df_rich["rich_best_PID"] == 2212)
#]
df_rich = df_rich[
    (df_rich["rich_best_PID"]!=-9999)
]



vals=[]
errs=[]
pCenters = []

# Split into bins
pBins = au.makeBins(df_rich, "p", binEdges=pEdges)

for i, pbin in enumerate(pBins):

    numerator_df = pbin[
        (pbin["pid"] == 321) &
        (pbin["bdt_pass"] == True) &
        (pbin["rich_best_PID"] != 321)
    ]

    denominator_df = pbin[
        (pbin["pid"] == 321) &
        (pbin["bdt_pass"] == True)
    ]

    numerator = len(numerator_df)
    denominator = len(denominator_df)

    r = 0
    rErr = 0

    if denominator != 0:
        r = numerator / denominator

        if numerator != 0:
            rErr = r * math.sqrt((1/numerator) + (1/denominator))

    vals.append(r)
    errs.append(rErr)

    pCenters.append((pEdges[i] + pEdges[i+1]) / 2)


# Plot
plt.figure(figsize=(7,5))

plt.errorbar(
    pCenters,
    vals,
    yerr=errs,
    fmt='o',
    capsize=4,
    markersize=6
)

plt.xlabel("Momentum p (GeV)")
plt.ylabel("Kaon contamination")
plt.title(r"Optimized BDT ep->eKX contamination with RICH Truth")

# Put ticks at bin edges
plt.xticks(pEdges)

# Optional: make sure plot spans exactly the bin range
plt.xlim(pEdges[0], pEdges[-1])

plt.grid(False)

# Save figure
plt.savefig(outDir+"DATA_kaon_contamination.png",
            dpi=150,
            bbox_inches="tight")



############################################################# MC


df_val = pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/val.parquet")

#df_val=au.apply_Sidis_Cuts(df_val)
mod, mod_df = au.load_model_and_data("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib", df_val)


# In[3]:


tStart = 5
tEnd = 11.25
tBinNum =5

pStart = 2.5
pEnd = 5
pBinNum = 10

pBinEdges = np.linspace(pStart, pEnd, pBinNum + 1)
tBinEdges = np.linspace(tStart, tEnd, tBinNum + 1)


# In[4]:


csvFile = "optimized_thresholds_MC.csv"

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


#####################################################################


cols = ["pid", "mc_matching_pid", "p", "theta", "beta", "chi2pid", "rich_RQ", "rich_best_PID", "rich_best_ntot", "vz","ftof_energy_1B", "ftof_time_1B", "ftof_path_1B", "ecin_path", "ecin_energy", "ecin_time"]
kinematics =["Mx_eKX","Mx_epiX","Mx_epX", "Q2", "W", "y"]

for kin in kinematics:
    cols.append(kin)

#df_test = uproot.open("/volatile/clas12/cooperb/SULI/pid_training_v2.root:PhysicsEvents").arrays(cols, library="pd")
df_test = uproot.open("~/ML_Files/MC_dataset_eKX/pid_training_v3.root:PhysicsEvents").arrays(cols, library="pd")

#df_test=pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/test.parquet")
df_test=df_test[df_test["mc_matching_pid"]!=-9999]
feature_names=au.get_feature_names("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib")
print(feature_names)
df_test = au.apply_model_to_df(mod, df_test, feature_names)
df_test=au.apply_Sidis_Cuts(df_test)


outDir="../../figures/Data_Application/contams/"

# Define momentum bins
pEdges = au.makeBinEdges(2.5, 5, 10)

# Select RICH angular range
df_rich = df_test

#df_rich=df_rich[df_rich["theta"]<20]
df_rich=df_rich[df_rich["rich_RQ"]>0.2]
df_rich=df_rich[df_rich["rich_best_ntot"]>3]
df_rich=df_rich[df_rich["rich_best_PID"]!=-9999]
df_rich = df_rich[df_rich["theta"] < 11.25]



# Split into bins
pBins = au.makeBins(df_rich, "p", binEdges=pEdges)

# Store results
bdt_vals = []
bdt_errs = []

eb_vals = []
eb_errs = []

pCenters = []

for i, pbin in enumerate(pBins):

    # BDT-selected events
    bdtMask = au.apply_optimized_bdt_cut(
        pbin,
        threshold_df=results_df
    )
    bdt_selected = pbin[bdtMask]

    # EB-selected events
    eb_selected = pbin[
        pbin["pid"] == 321
    ]

    # Compute contamination
    bdt_r, bdt_e = au.compute_contamination(bdt_selected)
    eb_r, eb_e = au.compute_contamination(eb_selected)

    bdt_vals.append(bdt_r)
    bdt_errs.append(bdt_e)

    eb_vals.append(eb_r)
    eb_errs.append(eb_e)

    pCenters.append(
        (pEdges[i] + pEdges[i + 1]) / 2
    )

# ----------------------------
# Plot
# ----------------------------

plt.figure(figsize=(7,5))


plt.errorbar(
    pCenters,
    bdt_vals,
    yerr=bdt_errs,
    fmt='o',
    capsize=4,
    markersize=6,
    label="BDT"
)


#plt.errorbar(
 #   pCenters,
  #  eb_vals,
   # yerr=eb_errs,
    #fmt='o',
    #capsize=4,
    #markersize=6,
    #label="EB PID"
#)


#plt.errorbar(
    #pCenters,
    #chi_vals,
    #yerr=chi_errs,
    #fmt='o',
   # capsize=4,
  #  markersize=6,
 #   label=r"$chi2pid$ cut"
#)


plt.xlabel("Momentum (GeV/c)")
plt.ylabel("Kaon contamination")

plt.title(
    r"ep->eKX contamination with MC Truth"
)

plt.xticks(pEdges)

plt.xlim(
    pEdges[0],
    pEdges[-1]
)

plt.ylim(bottom=0)

plt.legend()
plt.grid(False)


plt.savefig(
    outDir+"MC_RQ_kaon_contamination_comparison.png",
    dpi=150,
    bbox_inches="tight"
)



plt.figure(figsize=(7,5))

plt.errorbar(
    pCenters,
    bdt_vals,
    yerr=bdt_errs,
    fmt='o',
    capsize=4,
    markersize=6,
    label="MC"
)

plt.errorbar(
    pCenters,
    vals,
    yerr=errs,
    fmt='o',
    capsize=4,
    markersize=6,
    label="DATA"
)

plt.xlabel("Momentum (GeV/c)")
plt.ylabel("Kaon contamination")

plt.title(
    r"ep->eKX contamination with MC Truth (RICH Quality Cuts)"
)

plt.xticks(pEdges)

plt.xlim(
    pEdges[0],
    pEdges[-1]
)

plt.ylim(bottom=0)

plt.legend()
plt.grid(False)


plt.savefig(
    outDir+"RICH_RQ_data_mc_contam_comparison.png",
    dpi=150,
    bbox_inches="tight"
)