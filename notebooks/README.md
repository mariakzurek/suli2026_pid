#SULI2026_PID PROGRAMS

Student: Cooper Bell (SULI 2026) PI: Maria Zurek (ANL)

This doccument gives a brief description on what each program in notebooks does

##PROGRAMS

First_Program.ipynb | Preliminary program meant to test uproot, also prints useful information about the dataframe such as confusion matrix and statistics

betaVsP.upynb | Creates a momentum verse beta plot, with each curve having the corresponding PID cut

EXPbetaVsP.ipynb | creates variations of the betaVsP plots but with different cuts

PIDStudies.ipynb | creates histograms of both pid variables along with a plot of the confusion matrix

plotMany.ipynb | plots histograms on top of each other with different PID cuts

EfficiancyWIP.ipynb | prototype program for developing binning capabilities and the computations of purity, contamination, Mis-ID, and efficiency.

compute_baseline.ipynb produces efficiency contamination ect, plots (1D and 2D) with bins on momentum and theta. Variations starting with P or PI are targeting to see the contributions of particular particles, while ones that start with PC followed by P or PI will do the same but with a momentum based chi2pid cut

compute_baselineV2.ipynb | produces efficiency contamination, ect plots, along with showing specific particle contributions to mis-id and contaminations. This program also creates plots useful for determining which chi2cuts work best as a baseline.

baselinesRefined.ipynb | contains refined efficiency definitions

audit.ipynb generates histograms of relevant variables to audit them for existing.

percentError.ipynb | plots percent error beetween data and MC

percentDifference | plots percent difference between data and MC

CutDiagram.ipynb | plots an outline of the cuts on the chi2pid p dependence graph

EfficiencyWIP.ipynb | old program, contains initial plots of raw mis-id contaminations ect. Useful for learning definitions

PIDStudies | produces confusion matricies and plots using baseline cuts. (BDT version uses the BDT)

KinematicCutTest.ipynb | checks how plots are effected by kinematic SIDIS cuts.

Threshold_Tuning | produces a csv of the BDT FOM optimal thresholds for theta-p bins (TEST includes a deviance parameter, these have been made a module in common functions as well)

comparison_basleine.ipynb | compares the BDT FOM optimal cut to the chi2pid baseline with contamination and efficiency (Script version avalible)

fileMixer Used for mixing the root files to be used before training (before build_dataset.py)

RICH_Contam.ipynb | Contains plots of data validation of the BDT