#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df_val = pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/val.parquet")

#df_val=au.apply_Sidis_Cuts(df_val)
mod, mod_df = au.load_model_and_data("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib", df_val)


# In[3]:


tStart = 5
tEnd = 11.25
tBinNum = 5

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


# In[ ]:


import uproot

cols = ["pid", "mc_matching_pid", "p", "theta", "beta", "chi2pid", "rich_RQ", "rich_best_PID", "rich_best_ntot", "vz","ftof_energy_1B", "ftof_time_1B", "ftof_path_1B", "ecin_path", "ecin_energy", "ecin_time"]
kinematics =["Mx_eKX","Mx_epiX","Mx_epX", "Q2", "W", "y"]

for kin in kinematics:
    cols.append(kin)

df_test = uproot.open("/volatile/clas12/cooperb/SULI/pid_training_v2.root:PhysicsEvents").arrays(cols, library="pd")

#df_test=pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/test.parquet")
df_test=df_test[df_test["mc_matching_pid"]!=-9999]
feature_names=au.get_feature_names("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib")
print(feature_names)
df_test = au.apply_model_to_df(mod, df_test, feature_names)
df_test=au.apply_Sidis_Cuts(df_test)


# In[ ]:


ebMask=(df_test["pid"]==321)


# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt

outDir="../../figures/Data_Application/"

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
    outDir+"RICH_MC_kaon_contamination_comparison.png",
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# In[ ]:


# ============================================================
# Figure of Merit: N_K / sqrt(N_K + N_pi) vs p
# rich_best_PID used as truth
# ============================================================

bdt_fom = []
eb_fom = []

pCenters_fom = []

pBins = au.makeBins(df_rich, "p", binEdges=pEdges)


for i, pbin in enumerate(pBins):

     # Apply optimized BDT cut
    bdtMask = au.apply_optimized_bdt_cut(
        pbin,
        threshold_df=results_df
    )
    
    # ----------------------------
    # BDT selection
    # ----------------------------
    bdt_selected = pbin[
        bdtMask
    ]

    # True K and pi after BDT cut
    bdt_K = bdt_selected[
        bdt_selected["mc_matching_pid"] == 321
    ]

    bdt_pi = bdt_selected[
        bdt_selected["mc_matching_pid"] == 211
    ]


    # ----------------------------
    # EB PID selection
    # ----------------------------
    eb_selected = pbin[
        pbin["pid"] == 321
    ]

    # True K and pi after EB PID cut
    eb_K = eb_selected[
        eb_selected["mc_matching_pid"] == 321
    ]

    eb_pi = eb_selected[
        eb_selected["mc_matching_pid"] == 211
    ]


    # ----------------------------
    # FOM calculation
    # ----------------------------
    def calc_fom(K, pi):

        NK = len(K)
        Npi = len(pi)

        if (NK + Npi) == 0:
            return 0

        return NK / np.sqrt(NK + Npi)


    bdt_fom.append(
        calc_fom(bdt_K, bdt_pi)
    )

    eb_fom.append(
        calc_fom(eb_K, eb_pi)
    )


    pCenters_fom.append(
        (pEdges[i] + pEdges[i+1]) / 2
    )


# ============================================================
# Plot FOM
# ============================================================

plt.figure(figsize=(7,5))


plt.plot(
    pCenters_fom,
    bdt_fom,
    'o',
    markersize=6,
    label="BDT"
)


plt.plot(
    pCenters_fom,
    eb_fom,
    'o',
    markersize=6,
    label="EB PID"
)


plt.xlabel("Momentum (GeV/c)")
plt.ylabel(r"$N_K/\sqrt{N_K+N_\pi}$")

plt.title(
    r"ep$\rightarrow$eKX PID Figure of Merit"
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
    outDir+"MC_kaon_FOM_comparison.png",
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# In[ ]:




