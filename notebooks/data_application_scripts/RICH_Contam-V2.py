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


# In[2]:


tStart = 0
tEnd = 35
tBinNum = 5

pStart = 2.5
pEnd = 5
pBinNum = 10

tBinEdges = np.linspace(tStart, tEnd, tBinNum + 1)
pBinEdges = np.linspace(pStart, pEnd, pBinNum + 1)


# In[4]:


import awkward as ak
cols = ["pid", "p", "theta", "beta", "chi2pid", "rich_RQ", "vz", "bdt_pass", "rich_best_PID", "rich_RQ", "rich_best_ntot", "bdt_score"]
kinematics =["Mx_eKX","Mx_epiX","Mx_epX", "Q2", "W", "y"]

for kin in kinematics:
    cols.append(kin)
    
#df = uproot.open("~/ML_Files/epkx_data/scored_large/epkx_dataset_large.root:PhysicsEvents").arrays(cols, library="pd")
df= uproot.open("~/ML_Files/scored_data_v01/nSidis_005046.root:PhysicsEvents").arrays(cols, library="pd")
df=df[df["rich_best_ntot"]>3]
df=df[df["rich_RQ"]>0.2]
df=df[df["rich_best_PID"]!=-9999]
df=au.apply_Sidis_Cuts(df)

outDir="../../figures/Data_Application/"

csvFile = "optimized_thresholds_DATA_V4.csv"

df["score"] = df["bdt_score"]

if Path(csvFile).exists():
    print(f"Loading optimized thresholds from {csvFile}")
    results_df = pd.read_csv(csvFile)
else:
    print("Optimized threshold file not found. Running optimization...")
    results_df = au.optimizeFOM_DATA(
        df,
        tBinEdges,
        pBinEdges,
        outputCSV=csvFile,
        deviation=0.03
    )



# In[5]:


import math
import numpy as np
import matplotlib.pyplot as plt

# Define momentum bins
pEdges = au.makeBinEdges(2.5, 5, 10)

# Select RICH angular range
df_rich = df[
    (df["rich_RQ"]>0.1)&
    (df["rich_best_ntot"]>3)&
    (df["rich_best_PID"] != -9999)&
    (df["theta"]<11.25)
]

# Add chi2pid selection column
df_rich["chi2pid_pass"] = passes_kplus_chi2pid_cut(
    df_rich["chi2pid"],
    df_rich["p"]
)


# Split into bins
pBins = au.makeBins(df_rich, "p", binEdges=pEdges)

# Store results
bdt_vals = []
bdt_errs = []

eb_vals = []
eb_errs = []

pCenters = []

for i, pbin in enumerate(pBins):

    # ----------------------------
    # BDT kaon selection
    # ----------------------------
    bdtMask = au.apply_optimized_bdt_cut(
        pbin,
        threshold_df=results_df
    )
    bdt_selected = pbin[bdtMask]

    # ----------------------------
    # EB PID kaon selection
    # ----------------------------
    eb_selected = pbin[
        pbin["pid"] == 321
    ]

    # Compute contamination
    bdt_r, bdt_e = au.compute_contamination_RICH(bdt_selected)
    eb_r, eb_e = au.compute_contamination_RICH(eb_selected)

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
 #   eb_vals,
 #   yerr=eb_errs,
 #   fmt='o',
 #   capsize=4,
 #   markersize=6,
 #   label="EB PID"
#)


#plt.errorbar(
 #   pCenters,
 #   chi_vals,
 #   yerr=chi_errs,
 #   fmt='o',
 #   capsize=4,
 #   markersize=6,
 #   label=r"$chi2pid$ cut"
#)


plt.xlabel("Momentum (GeV/c)")
plt.ylabel("Kaon contamination")

plt.title(
    r"ep->eKX contamination with RICH Truth"
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
    outDir+"_DATA_kaon_contamination_comparison.png",
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# In[6]:


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
        bdt_selected["rich_best_PID"] == 321
    ]

    bdt_pi = bdt_selected[
        bdt_selected["rich_best_PID"] == 211
    ]


    # ----------------------------
    # EB PID selection
    # ----------------------------
    eb_selected = pbin[
        pbin["pid"] == 321
    ]

    # True K and pi after EB PID cut
    eb_K = eb_selected[
        eb_selected["rich_best_PID"] == 321
    ]

    eb_pi = eb_selected[
        eb_selected["rich_best_PID"] == 211
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
    outDir+"DATA_kaon_FOM_comparison.png",
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# In[8]:


import math
import numpy as np
import matplotlib.pyplot as plt

# Define momentum bins
pEdges = au.makeBinEdges(3, 5, 10)

# Select RICH angular range
df_rich = df[
    (df["theta"] < 20) &
    (df["rich_best_PID"] != -9999)
]

# Add chi2pid selection column
df_rich["chi2pid_pass"] = passes_kplus_chi2pid_cut(
    df_rich["chi2pid"],
    df_rich["p"]
)


# Split into bins
pBins = au.makeBins(df_rich, "p", binEdges=pEdges)


# Store results
bdt_vals = []
bdt_errs = []

eb_vals = []
eb_errs = []

chi_vals = []
chi_errs = []

pCenters = []


for i, pbin in enumerate(pBins):
    bdtMask=au.apply_optimized_bdt_cut(pbin, threshold_df=results_df)
    # ----------------------------
    # BDT kaon selection
    # ----------------------------
    bdt_den = pbin[
        bdtMask
    ]

    bdt_num = bdt_den[
        bdt_den["rich_best_PID"] != 321
    ]


    # ----------------------------
    # EB PID kaon selection
    # ----------------------------
    eb_den = pbin[
        pbin["bdt_pass"] == True
    ]

    eb_num = eb_den[
        eb_den["rich_best_PID"] != 321
    ]


    # ----------------------------
    # chi2pid kaon selection
    # ----------------------------
    chi_den = pbin[
        (pbin["pid"] == 321) &
        (pbin["chi2pid_pass"])
    ]

    chi_num = chi_den[
        chi_den["rich_best_PID"] != 321
    ]


    # Function for ratio + error
    def calc_contamination(num, den):

        numerator = len(num)
        denominator = len(den)

        if denominator == 0:
            return 0, 0

        ratio = numerator / denominator

        if numerator > 0:
            err = ratio * math.sqrt(
                (1/numerator) + (1/denominator)
            )
        else:
            err = 0

        return ratio, err


    bdt_r, bdt_e = calc_contamination(bdt_num, bdt_den)
    eb_r, eb_e = calc_contamination(eb_num, eb_den)
    chi_r, chi_e = calc_contamination(chi_num, chi_den)


    bdt_vals.append(bdt_r)
    bdt_errs.append(bdt_e)

    eb_vals.append(eb_r)
    eb_errs.append(eb_e)

    chi_vals.append(chi_r)
    chi_errs.append(chi_e)


    pCenters.append(
        (pEdges[i] + pEdges[i+1]) / 2
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
    label="DATA",
    color="blue"
)


plt.errorbar(
    pCenters,
    eb_vals,
    yerr=eb_errs,
    fmt='o',
    capsize=4,
    markersize=6,
    label="MC",
    color="red"
)


#plt.errorbar(
 #   pCenters,
 #   chi_vals,
 #   yerr=chi_errs,
 #   fmt='o',
 #   capsize=4,
 #   markersize=6,
 #   label=r"$chi2pid$ cut"
#)


plt.xlabel("Momentum (GeV/c)")
plt.ylabel("Kaon contamination")

plt.title(
    r"ep->eKX contamination with RICH Truth"
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
    outDir+"Validation_kaon_contamination_comparison.png",
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




