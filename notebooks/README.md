#SULI2026_PID PROGRAMS

Student: Cooper Bell (SULI 2026) PI: Maria Zurek (ANL)

This doccument gives a brief description on what each program in notebooks does

## PROGRAMS

First_Program.ipynb | Preliminary program meant to test uproot, also prints useful information about the dataframe such as confusion matrix and statistics

betaVsP.upynb | Creates a momentum verse beta plot, with each curve having the corresponding PID cut

EXPbetaVsP.ipynb | creates variations of the betaVsP plots but with different cuts

PIDStudies.ipynb | creates histograms of both pid variables along with a plot of the confusion matrix

plotMany.ipynb | plots histograms on top of each other with different PID cuts

EfficiancyWIP.ipynb | prototype program for developing binning capabilities and the computations of purity, contamination, Mis-ID, and efficiency.

compute_baseline.ipynb produces efficiency contamination ect, plots (1D and 2D) with bins on momentum and theta. Variations starting with P or PI are targeting to see the contributions of particular particles, while ones that start with PC followed by P or PI will do the same but with a momentum based chi2pid cut

<<<<<<< HEAD
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


## data_application_scripts/
- RICH_Contam.py | uses the RICH as Truth, prelimminary plots, RICH CONTAM-V2.py does the same but with a MC comparison.
- contam_v3.py | produces finalized comparison between the RICH validated contamination and MC.
- overlap.py | generates heatmaps showing the SIDIS kinematic coverage, and the epi(N) kinematic coverage.
- MC_contam_rich_compar | compares FOM plots with RICH and MC's BDT
- epiN_analysis | uses the epi(N) exclusive reaction to estimate SIDIS contamination by calculating the pion->kaon misID and EB pion efficiency.

Notebook versions of these are avalible but the kernals will fail when using substantial data.

## uncertainty/
- uncertainty_appender.py | adds uncertainties of MC events to the per-bin-sweep-csv
- uncertainty_calibration.py | finds the uncertainty from the platt calibration of the BDT
- uncertainty_mx.py | finds the uncertainty of the epi(N) contamination estimation method by varing the Mx window.
- uncertainty_RICH.py | finds the uncertainty of the RICH method by varting rich_RQ
- uncertainty_weighting.py | finds the uncertainty from ommitting the p-theta weighting in training
- relative_shift.py | calculates delta_c/c and appends it to each csv passed through it
- csv2md.ipynb | creates a .md table from a csv
- headline.py | calculates quadratures and plots them for representative bins using the CSV's.




RICH_Contam.ipynb | Contains plots of data validation of the BDT
=======
- baselinesRefined.ipynb | contains refined efficiency definitions

- audit.ipynb generates histograms of relevant variables to audit them for existing.

- percentError.ipynb | plots percent error beetween data and MC

- percentDifference | plots percent difference between data and MC

- CutDiagram.ipynb | plots an outline of the cuts on the chi2pid p dependence graph

- EfficiencyWIP.ipynb | contains initial plots of raw mis-id contaminations ect. Useful for learning definitions

- PIDStudies | produces confusion matricies and plots using baseline cuts. (BDT version uses the BDT)

- KinematicCutTest.ipynb | checks how plots are effected by kinematic SIDIS cuts.

- Threshold_Tuning | produces a csv of the BDT FOM optimal thresholds for theta-p bins (TEST includes a deviance parameter, these have been made a module in common functions as well)

- comparison_basleine.ipynb | compares the BDT FOM optimal cut to the chi2pid baseline with contamination and efficiency (Script version avalible)

- fileMixer Used for mixing the root files to be used before training (before build_dataset.py)

- RICH_Contam.ipynb | Contains plots of data validation of the BDT

  




- 


- 




>>>>>>> 710292d (work done on 7/15/2026)
