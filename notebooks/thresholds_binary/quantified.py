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


df_val = pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/val.parquet")

#df_val=au.apply_Sidis_Cuts(df_val)
mod, mod_df = au.load_model_and_data("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib", df_val)



# In[3]:


tStart = 5
tEnd = 35
tBinNum = 5

pStart = 3
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


# In[5]:


#df_test = uproot.open("/volatile/clas12/cooperb/SULI/pid_training_v2.root:PhysicsEvents").arrays(cols, library="pd")

df_test=pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/test.parquet")
df_test=df_test[df_test["mc_matching_pid"]!=-9999]
feature_names=au.get_feature_names("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib")
print(feature_names)
df_test = au.apply_model_to_df(mod, df_test, feature_names)
#df_test=au.apply_Sidis_Cuts(df_test)


# In[6]:


pbinHigh =df_test[((df_test["p"]>2.75)&(df_test["p"]<5))]
pbinTot=df_test[((df_test["p"]>0.5)&(df_test["p"]<5))]




# In[7]:


highchi=pbinHigh[passes_kplus_chi2pid_cut(pbinHigh["chi2pid"], pbinHigh["p"])]


# In[8]:


totchi= pbinTot[passes_kplus_chi2pid_cut(pbinTot["chi2pid"], pbinTot["p"])]


# In[9]:


highbdt =pbinHigh[au.apply_optimized_bdt_cut(pbinHigh, threshold_df=results_df)]


# In[10]:


totbdt =pbinTot[au.apply_optimized_bdt_cut(pbinTot, threshold_df=results_df)]


# In[ ]:


chi_tot_eff= au.compute_efficiency(pbinTot, passes_kplus_chi2pid_cut(pbinTot["chi2pid"], pbinTot["p"]) )
chi_tot_cont = au.compute_contamination(totchi)
bdt_tot_eff = au.compute_efficiency(pbinTot, au.apply_optimized_bdt_cut(pbinTot, threshold_df=results_df))
bdt_tot_cont= au.compute_contamination(totbdt)
chi_high_eff = au.compute_efficiency(pbinHigh, passes_kplus_chi2pid_cut(pbinHigh["chi2pid"], pbinTot["p"]) )
chi_high_cont = au.compute_contamination(highchi) 
bdt_high_eff =au.compute_efficiency(pbinHigh, au.apply_optimized_bdt_cut(pbinHigh, threshold_df=results_df))
bdt_high_cont =au.compute_contamination(highchi)


# In[1]:


print(" chi_tot_eff - bdt_tot_eff")
print( chi_tot_eff - bdt_tot_eff)
print( "chi_high_eff - bdt_high_eff")
print( chi_high_eff - bdt_high_eff)
print( "chi_tot_cont - bdt_tot_cont")
print( chi_tot_cont - bdt_tot_cont)
print( "chi_high_cont - bdt_high_cont")
print( chi_high_cont - bdt_high_cont)


# In[ ]:




