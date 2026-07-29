
import glob
import sys
sys.path.append("../../scripts/")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import awkward as ak
import uproot
from scipy.integrate import quad
from scipy.optimize import curve_fit

import common_functions as au


def rel_shift(csvPath, delta_c, c):

    # Read the CSV
    df = pd.read_csv(csvPath)

    # List to hold calculated values
    new_values = []

    # Loop over each row
    for _, row in df.iterrows():

        # Read values from specified columns
        x = row[delta_c]
        y = row[c]

        # Calculate relative shift
        if y != 0:
            result = x / y
        else:
            result = 9999

        # Store result
        new_values.append(result)

    # Add new column
    df["relative_drift"] = new_values

    # Save updated CSV
    df.to_csv(csvPath, index=False)

    return df

rel_shift("calibration_sensitivity.csv","delta_c","c_calibrated")
rel_shift("rich_contamination_binned.csv","delta_c","contamination_initial")
rel_shift("weighted_contamination_comparison.csv","C_difference","C_unweighted")
rel_shift("weighted_contamination_comparison_granular.csv","C_difference","C_unweighted")
rel_shift("threshold_sensitivity.csv","delta_c_total","C_base")
rel_shift("mx_width_sensitivity.csv","delta_c","contamination_initial")
#rel_shift("per_bin_sweep_uncertianties.csv","delta_c","contamination_initial")







