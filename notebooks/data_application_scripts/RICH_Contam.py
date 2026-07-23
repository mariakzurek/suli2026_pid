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


cols = ["pid", "p", "theta", "beta", "chi2pid", "rich_RQ", "vz", "bdt_pass", "rich_best_PID", "rich_RQ", "rich_best_ntot", "bdt_score"]
kinematics =["Mx_eKX","Mx_epiX","Mx_epX", "Q2", "W", "y"]

for kin in kinematics:
    cols.append(kin)
    
df = uproot.open("~/ML_Files/scored_data_v01/nSidis_005046.root:PhysicsEvents").arrays(cols, library="pd")
df=df[df["rich_best_ntot"]>3]
df=df[df["rich_RQ"]>0.2]
df=df[df["rich_best_PID"]!=-9999]
df=au.apply_Sidis_Cuts(df)

outDir="../../figures/Data_Application/"


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
plt.savefig(outDir+"bdt_rich_contamination.png",
            dpi=150,
            bbox_inches="tight")

plt.show()


# In[4]:


import math
import numpy as np
import matplotlib.pyplot as plt

# Define momentum bins
pEdges = au.makeBinEdges(2.5, 5, 10)





vals=[]
errs=[]
pCenters = []

# Split into bins
pBins = au.makeBins(df_rich, "p", binEdges=pEdges)

for i, pbin in enumerate(pBins):

    chiMask=au.ApplyMatchedEfficiency(pbin, "../data_application/matched_eff_thresholds.csv")

    numerator_df = pbin[
        (pbin["pid"] == 321) &
        chiMask &
        (pbin["rich_best_PID"] != 321)
    ]

    denominator_df = pbin[
        (pbin["pid"] == 321) &
        chiMask
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
plt.title(r"BDT RICH Truth contamination (matched eff)")

# Put ticks at bin edges
plt.xticks(pEdges)

# Optional: make sure plot spans exactly the bin range
plt.xlim(pEdges[0], pEdges[-1])

plt.grid(False)

# Save figure
plt.savefig(outDir+"chi2pid_rich_contamination.png",
            dpi=150,
            bbox_inches="tight")

plt.show()


# In[5]:


import uproot
import matplotlib.pyplot as plt

# Columns needed
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
    "Mx_eKX",
    "Mx_epiX",
    "Mx_epX",
    "Q2",
    "W",
    "y",
    "rich_best_ntot",
    "bdt_score"
]

# Load scored pion sample
df = uproot.open(
    "~/ML_Files/data_epiN_v02/scored/epiN_dataset.root:PhysicsEvents"
).arrays(cols, library="pd")


# -------------------------------------------------------
# Apply chi2pid efficiency cut BEFORE filtering dataframe
# -------------------------------------------------------
df["chi_pass"] = passes_kplus_chi2pid_cut(df["chi2pid"],df["p"])
    


# -------------------------------------------------------
# Apply good RICH quality cuts
# -------------------------------------------------------
df = df[
    (df["rich_best_ntot"] > 3) &
    (df["rich_RQ"] > 0.2) &
    (df["rich_best_PID"] != -9999)
]


outDir = "../../figures/Data_Application/"


# -------------------------------------------------------
# Define selections
# -------------------------------------------------------

# All true pions with good RICH
pCut_all = df[
    (df["p"] > 3) &
    (df["theta"] < 11.25) &
    (df["pid"] == 211)
]


# True pions passing BDT kaon selection
pCut_bdt = df[
    (df["p"] > 2.5) &
    (df["theta"] < 11.25) &
    (df["pid"] == 211) &
    (df["bdt_pass"])
]


# True pions classified as kaons by RICH
pCut_rich = df[
    (df["p"] > 2.5) &
    (df["theta"] < 11.25) &
    (df["pid"] == 211) &
    (df["rich_best_PID"] == 321)
]


# True pions passing chi2pid kaon selection
pCut_chi2pid = df[
    (df["p"] > 2.5) &
    (df["theta"] < 11.25) &
    (df["pid"] == 211) &
    (df["chi_pass"])
]


# Check statistics
print("All pions:", len(pCut_all))
print("BDT:", len(pCut_bdt))
print("RICH PID:", len(pCut_rich))
print("chi2pid:", len(pCut_chi2pid))


# -------------------------------------------------------
# Plot missing mass
# -------------------------------------------------------
plt.figure(figsize=(7,5))

plt.hist(
    pCut_all["Mx_epiX"],
    bins=100,
    histtype="step",
    label=r"All $\pi^+$ (good RICH)",
)


plt.hist(
    pCut_bdt["Mx_epiX"],
    bins=100,
    histtype="step",
    label=r"$\pi^+$ passing K BDT",
)


plt.hist(
    pCut_chi2pid["Mx_epiX"],
    bins=100,
    histtype="step",
    label=r"$\pi^+$ passing $chi2pid$",
    color="purple"
)


plt.hist(
    pCut_rich["Mx_epiX"],
    bins=100,
    histtype="step",
    label=r"$\pi^+$ with RICH PID = 321",
)


plt.xlabel(r"$M_x(e\pi^+X)$ [GeV]")
plt.ylabel("Counts")
plt.title(r"Neutron missing mass: $ep\rightarrow e\pi^+(n)$ ($p>3$ GeV)")



plt.legend()
plt.grid(False)

plt.savefig(
    outDir + "OLDMx_epiN_BDT_misID.png",
    dpi=150,
    bbox_inches="tight"
)
plt.xlim(0, 1)
plt.show()
plt.close()


# In[6]:


# ------------------------------------------------------- # Columns needed # ------------------------------------------------------- cols = [ "pid", "p", "theta", "beta", "chi2pid", "rich_RQ", "rich_best_PID", "rich_best_ntot", "Mx_eKX", "Q2", "W", "y", "vz", "bdt_pass", "bdt_score", "Mx_epiX" ] # ------------------------------------------------------- # Load eKX scored sample # ------------------------------------------------------- df = uproot.open( "/work/clas12/CooperBe/MLStuff/scored_data_v01/nSidis_005125.root:PhysicsEvents" ).arrays(cols, library="pd") # ------------------------------------------------------- # Apply RICH acceptance and quality cuts # ------------------------------------------------------- df = df[ (df["theta"] < 20) & (df["rich_best_ntot"] > 2.5) & (df["rich_RQ"] > 0.1) & (df["rich_best_PID"] != -9999)& (df["Mx_epiX"] != -9999) ] outDir = "../../figures/Data_Application/" # ------------------------------------------------------- # Define kaon selections # ------------------------------------------------------- # EB kaon ID kCut_EB = df[ (df["pid"] == 321) ] # BDT kaon ID kCut_BDT = df[ (df["bdt_pass"])&(df["pid"] == 321) ] # RICH kaon ID kCut_RICH = df[ (df["rich_best_PID"] == 321) ] # ------------------------------------------------------- # Check statistics # ------------------------------------------------------- print("Events with good RICH:") print("Total:", len(df)) print("EB K:", len(kCut_EB)) print("BDT K:", len(kCut_BDT)) print("RICH K:", len(kCut_RICH)) # ------------------------------------------------------- # Plot Mx(eKX) # ------------------------------------------------------- plt.figure(figsize=(7,5)) plt.hist( kCut_EB["Mx_epiX"], bins=100, histtype="step", label=r"$ep\rightarrow ehX,\ h=EB\ K^+$", density=False ) plt.hist( kCut_BDT["Mx_epiX"], bins=100, histtype="step", label=r"$ep\rightarrow ehX,\ h=BDT\ K^+$", density=False ) plt.hist( kCut_RICH["Mx_epiX"], bins=100, histtype="step", label=r"$ep\rightarrow ehX,\ h=RICH\ K^+$", density=False ) plt.xlabel(r"$M_X(epiX)$ [GeV]") plt.ylabel("Counts") plt.title(r"$ep\rightarrow epiX$ Kaon PID comparison (RICH acceptance)") # Let matplotlib determine range first # Uncomment after checking the distribution: # plt.xlim(0,5) plt.legend() plt.grid(False) plt.savefig( outDir + "Mx_epiX_RICH_overlap.png", dpi=150, bbox_inches="tight" ) plt.xlim(0, 1) plt.show() plt.close()


# In[7]:


import uproot
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Columns needed
# -------------------------------------------------------
cols = [
    "p",
    "theta",
    "Mx_epiX",
    "bdt_pass"
]


# -------------------------------------------------------
# Load exclusive pion data
# -------------------------------------------------------
df = uproot.open(
    "~/ML_Files/data_epiN_v01/scored/epiN_dataset.root:PhysicsEvents"
).arrays(cols, library="pd")


outDir = "../../figures/Data_Application/"


# -------------------------------------------------------
# First: look at missing mass to choose neutron window
# -------------------------------------------------------

plt.figure(figsize=(7,5))

plt.hist(
    df["Mx_epiX"],
    bins=200,
    histtype="step"
)

plt.xlabel(r"$M_X(e\pi^+)$ [GeV]")
plt.ylabel("Counts")
plt.title(r"Exclusive pion sample: $ep\rightarrow e\pi^+(n)$")

plt.grid(False)
plt.show()


# -------------------------------------------------------
# Select neutron peak
# Adjust these values based on the plot above
# -------------------------------------------------------

neutron_low = 0.85
neutron_high = 1.05


pi_sample = df[
    (df["Mx_epiX"] > neutron_low) &
    (df["Mx_epiX"] < neutron_high) &
    (df["p"] > 3) &
    (df["theta"] < 20)
]


# -------------------------------------------------------
# Apply BDT kaon selection
# -------------------------------------------------------

pi_fake = pi_sample[
    pi_sample["bdt_pass"]
]


print("--------------------------------")
print("Clean pion sample:", len(pi_sample))
print("Pions passing BDT:", len(pi_fake))

if len(pi_sample) > 0:
    print(
        "Overall pi -> K fake rate:",
        len(pi_fake)/len(pi_sample)
    )
print("--------------------------------")


# -------------------------------------------------------
# Fake rate vs momentum
# -------------------------------------------------------

p_bins = np.linspace(3, 5, 10)

p_centers = []
fake_rates = []
fake_errors = []


for low, high in zip(p_bins[:-1], p_bins[1:]):

    bin_events = pi_sample[
        (pi_sample["p"] >= low) &
        (pi_sample["p"] < high)
    ]

    if len(bin_events) == 0:
        continue


    bin_fake = bin_events[
        bin_events["bdt_pass"]
    ]


    N = len(bin_events)
    N_fake = len(bin_fake)

    rate = N_fake / N

    # Binomial statistical uncertainty
    error = np.sqrt(rate*(1-rate)/N)


    p_centers.append((low+high)/2)
    fake_rates.append(rate)
    fake_errors.append(error)


# -------------------------------------------------------
# Plot fake rate
# -------------------------------------------------------

plt.figure(figsize=(7,5))


plt.errorbar(
    p_centers,
    fake_rates,
    yerr=fake_errors,
    marker="o",
    linestyle=""
)


plt.xlabel(r"$momentum$ GeV/c")
plt.ylabel(r"Pion -> Kaon Mis-ID")
plt.title(
    r"BDT $\pi$-mis-ID rate in $ep\rightarrow e\pi^+(n)$ data"
)

plt.grid(False)


plt.savefig(
    outDir + "pi_misID_BDT_data.png",
    dpi=150,
    bbox_inches="tight"
)


plt.show()
plt.close()


# In[ ]:




