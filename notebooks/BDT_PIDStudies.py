#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import uproot
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append("../scripts/")

from pathlib import Path
import common_functions as au
from baseline_chi2pid import passes_kplus_chi2pid_cut

#open root file
#df = uproot.open(
 #   "/volatile/clas12/cooperb/SULI/pid_training_test.root:PhysicsEvents"
#).arrays(library="pd")
cols = ["pid", "mc_matching_pid", "p", "theta", "beta", "chi2pid", "rich_RQ", "vz","ftof_energy_1B", "ftof_time_1B", "ftof_path_1B", "ecin_path", "ecin_energy", "ecin_time"]
kinematics =["Mx_eKX","Mx_epiX","Mx_epX", "Q2", "W", "y"]

for kin in kinematics:
    cols.append(kin)

df = uproot.open("/volatile/clas12/cooperb/SULI/pid_training_v2.root:PhysicsEvents").arrays(cols, library="pd")
direct ="/work/clas12/CooperBe/Argonne2026/suli2026_pid/figures/ConfusionMatrix/"
allPlots=[]





# In[ ]:





# In[2]:


df_val = pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/val.parquet")
mod, mod_df = au.load_model_and_data("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib", df_val)


# In[3]:


#creates a list of matching events (excludes events that failed matching)
matched = df[df["mc_matching_pid"] != -9999].copy()
baseline=matched[
(matched["Q2"]>2)&
(matched["W"]>2)&
((matched["y"]>0)&(matched["y"]<0.75))
]
matched=baseline
baseline=matched[
    ((matched["pid"]==321)&(matched["Mx_eKX"]>1.6))|
    ((matched["pid"]==211)&(matched["Mx_epiX"]>1.5))|
    ((matched["pid"]==2212)&(matched["Mx_epX"]>1))]
matched=baseline


# In[4]:


tStart = 5
tEnd = 35
tBinNum = 5

pStart = 0.5
pEnd = 3.2
pBinNum = 10

tBinEdges = np.linspace(tStart, tEnd, tBinNum + 1)
pBinEdges = np.linspace(pStart, pEnd, pBinNum + 1)


# In[5]:


csvFile = "../figures/full_range/optimized_thresholds_Matrix.csv"

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
feature_names=au.get_feature_names("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib")
print(feature_names)
matched = au.apply_model_to_df(mod, matched, feature_names)



# In[6]:


fig, ax2 = plt.subplots(figsize=(7, 5)) #plots just the matching pid as a histogram
ax2.hist(matched["mc_matching_pid"].clip(-2212, 2212), bins=100, histtype="step", density=True)
ax2.set_xlabel("mc matching pid"); ax2.set_ylabel("Density");
ax2.set_xticks(np.linspace(-2212, 2212, 10))
fig.tight_layout()
fig.savefig(direct+"matchedPIDs.png", dpi=150)
plt.close(fig)


# In[7]:


fig, ax2 = plt.subplots(figsize=(7, 5)) #plots just the True pid as a histogram
ax2.hist(matched["pid"].clip(-2212, 2212), bins=100, histtype="step", density=True)
ax2.set_xlabel("pid"); ax2.set_ylabel("Density");
ax2.set_xticks(np.linspace(2212, 2212, 10))
fig.tight_layout()
fig.savefig(direct+"PIDs.png", dpi=150)
plt.close(fig)


# In[8]:


bdtMask=au.apply_optimized_bdt_cut(matched, threshold_df=results_df)


# In[9]:


# plots the confusion matrix with all generated particles (K- and Pi- included)
subset = matched[((matched["mc_matching_pid"] == 211) | (matched["mc_matching_pid"] == 321) | (matched["mc_matching_pid"]==2212))& (matched["mc_matching_pid"]!='All')]
top_truths = subset["mc_matching_pid"].value_counts().head(8).index
contam = pd.crosstab(
    subset["pid"],
    subset["mc_matching_pid"].where(subset["mc_matching_pid"].isin(top_truths), other="other"),
    margins=False,
)

print("\nContamination matrix (EB pid × MC-truth pid):") #these three lines are for checking that the matrix is correct
print(contam)
print(contam.shape)

fig, ax2 = plt.subplots(figsize=(7, 5))
ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0, 1, 2, 3])
ax2.set_xticklabels([211, 321, 2212])
ax2.set_yticklabels([-321, -211, 211, 321])
ax2.set_xlabel("True PID")
ax2.set_ylabel("Reconstructed PID")
im = ax2.imshow(contam, cmap='coolwarm', origin='lower')
fig.colorbar(im, ax=ax2)
fig.savefig("figures/ALLGENconfuseMatrix.png", dpi=150)
allPlots.append(fig)
plt.close(fig)


# In[ ]:


#plots confusion matrix with only kaons, pions and protons
subset = matched[((matched["mc_matching_pid"] == 211) | (matched["mc_matching_pid"] == 321) | (matched["mc_matching_pid"]==2212))]
subset2 = subset[(subset["pid"]!=-211)&(subset["pid"]!=-321)]
subset2=subset2[bdtMask]
top_truths = subset2["mc_matching_pid"].value_counts().head(8).index
contam = pd.crosstab(
    subset2["pid"],
    subset2["mc_matching_pid"].where(subset2["mc_matching_pid"].isin(top_truths), other="other"),
    margins=False,
)
contam2 = pd.crosstab(
    subset2["pid"],
    subset2["mc_matching_pid"].where(subset2["mc_matching_pid"].isin(top_truths), other="other"),
    margins=True,
)
print("\nContamination matrix (EB pid × MC-truth pid):") #these three lines are for checking that the matrix is correct
print(contam2)
print(contam2.shape)


contam = contam.astype(float)
for i in contam.index:
    for j in contam.columns:
        contam.loc[i, j] *= 100/contam2.loc[i, "All"]
print("\nContamination matrix (EB pid × MC-truth pid):") #these three lines are for checking that the matrix is correct
print(contam)
print(contam.shape)

fig, ax2 = plt.subplots(figsize=(7, 5))
ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0, 1, 2])
ax2.set_xticklabels([211, 321, 2212])
ax2.set_yticklabels([211, 321, 2212])
ax2.set_xlabel("True PID")
ax2.set_ylabel("Reconstructed PID")
ax2.set_title("Contamination and Purity Matrix")
im = ax2.imshow(contam, cmap='coolwarm', origin='lower')
for i in range(contam.shape[0]):
    for j in range(contam.shape[1]):
        ax2.text(
            j, i,
            f"{contam.iloc[i, j]:.2f}%",
            ha="center",
            va="center"
        )
fig.colorbar(im, ax=ax2)
fig.savefig(direct+"BDT_PCMatrix.png", dpi=150)
plt.show()
allPlots.append(fig)
plt.close(fig)


# In[ ]:


#plots confusion matrix with only kaons, pions and protons
subset = matched[((matched["mc_matching_pid"] == 211) | (matched["mc_matching_pid"] == 321) | (matched["mc_matching_pid"]==2212))]
subset2 = subset[(subset["pid"]!=-211)&(subset["pid"]!=-321)]
top_truths = subset2["mc_matching_pid"].value_counts().head(8).index
contam = pd.crosstab(
    subset2["pid"],
    subset2["mc_matching_pid"].where(subset2["mc_matching_pid"].isin(top_truths), other="other"),
    margins=False,
)
contam2 = pd.crosstab(
    subset2["pid"],
    subset2["mc_matching_pid"].where(subset2["mc_matching_pid"].isin(top_truths), other="other"),
    margins=True,
)
print("\nContamination matrix (EB pid × MC-truth pid):") #these three lines are for checking that the matrix is correct
print(contam2)
print(contam2.shape)


contam = contam.astype(float)
for i in contam.index:
    print("\nContamination matrix (EB pid × MC-truth pid):") #these three lines are for checking that the matrix is correct
    print(contam)
    print(contam.shape)

fig, ax2 = plt.subplots(figsize=(7, 5))
ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0, 1, 2])
ax2.set_xticklabels([211, 321, 2212])
ax2.set_yticklabels([211, 321, 2212])
ax2.set_xlabel("True PID")
ax2.set_ylabel("Reconstructed PID")
ax2.set_title("Confusion Matrix")
im = ax2.imshow(contam, cmap='coolwarm', origin='lower')
for i in range(contam.shape[0]):
    for j in range(contam.shape[1]):
        ax2.text(
            j, i,
            f"{contam.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )
fig.colorbar(im, ax=ax2)
fig.savefig(direct+"confuseMatrixALT.png", dpi=150)
plt.show()
allPlots.append(fig)
plt.close(fig)


# In[ ]:


#plots confusion matrix with only kaons, pions and protons
subset = matched[((matched["mc_matching_pid"] == 211) | (matched["mc_matching_pid"] == 321) | (matched["mc_matching_pid"]==2212))]
subset2 = subset[(subset["pid"]!=-211)&(subset["pid"]!=-321)]
top_truths = subset2["mc_matching_pid"].value_counts().head(8).index
contam = pd.crosstab(
    subset2["pid"],
    subset2["mc_matching_pid"].where(subset2["mc_matching_pid"].isin(top_truths), other="other"),
    margins=False,
)
contam2 = pd.crosstab(
    subset2["pid"],
    subset2["mc_matching_pid"].where(subset2["mc_matching_pid"].isin(top_truths), other="other"),
    margins=True,
)
print("\nContamination matrix (EB pid × MC-truth pid):") #these three lines are for checking that the matrix is correct
print(contam2)
print(contam2.shape)


contam = contam.astype(float)
for i in contam.index:
    for j in contam.columns:
        contam.loc[i, j] *= 100/contam2.loc["All", j]
print("\nContamination matrix (EB pid × MC-truth pid):") #these three lines are for checking that the matrix is correct
print(contam)
print(contam.shape)

fig, ax2 = plt.subplots(figsize=(7, 5))
ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0, 1, 2])
ax2.set_xticklabels([211, 321, 2212])
ax2.set_yticklabels([211, 321, 2212])
ax2.set_xlabel("True PID")
ax2.set_ylabel("Reconstructed PID")
ax2.set_title("Mis-ID and Efficiency")
im = ax2.imshow(contam, cmap='coolwarm', origin='lower')
for i in range(contam.shape[0]):
    for j in range(contam.shape[1]):
        ax2.text(
            j, i,
            f"{contam.iloc[i, j]:.2f}%",
            ha="center",
            va="center"
        )
fig.colorbar(im, ax=ax2)
fig.savefig(direct+"EMMatrix.png", dpi=150)
plt.show()
allPlots.append(fig)
plt.close(fig)


# In[ ]:


with PdfPages("/work/clas12/CooperBe/Argonne2026/suli2026_pid/figures/ConfusionMatrix/" + "AllMatrix.pdf") as pdf:
    for plot in allPlots:
        pdf.savefig(plot)


# In[ ]:




