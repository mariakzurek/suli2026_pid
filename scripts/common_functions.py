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


import sys
from pathlib import Path




##############################################################################################################
#   PLEASE READ
#   This script does not do anything on it's own, it is a list of functions that are useful in analysis
#   to use these functions do "from functions.py import [name of desired function]" When performing you're imports
# 
#   WHAT THIS PROGRAM CONTAINS
#         -functions to compute purity, contamination, efficiency, and Mis-ID
#         -functions to automatically create bins on data
#         -function that will apply basic SIDIS analysis cuts
#         -function that will overlay plots
#         -
#
#
#   PLANNED
#        -function to Load BDT ML model
#        -function to make df have the ML BDT Classifier "score" variable
#        
#
#
##############################################################################################################

def compute_contamination(df, pid=None):              
    #Computes the contamnation from an input df, optionally set pid to see a specific particle's contribution
    r=0
    rErr=0
    if pid==None:
        temp=df[df["pid"]==321]
        a= (temp["mc_matching_pid"]!=321).sum()
        b= temp["pid"].sum()
        r=0
        rErr=99
        if b!=0:
            r=a/b
        if a!=0:
            rErr = r*math.sqrt((1/a)+(1/b))
    else:
        temp=df[df["pid"]==321]
        a= (temp["mc_matching_pid"]==pid).sum()
        b= (temp["pid"]).sum()
        r=0
        rErr=99
        if b!=0:
            r=a/b
        if a!=0:
            rErr = r*math.sqrt((1/a)+(1/b))
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

def compute_efficiency(df, cut, raw):
    #Will compute efficiency, raw gives the raw, uncut efficiency (PID efficiency)
    #while cut a cut you want to test the efficiency of (must be masked cut)
    if not raw:
        temp = df[df["pid"]==321]
    else:
        temp=df
    up=temp[cut]
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

def overlayPlots(ax, plots, labels):

    for obj, label in zip(plots, labels):
        obj.set_label(label)

    ax.legend()


import joblib
import json
import pathlib
import numpy as np

def load_model_and_data(model_path, df):

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


