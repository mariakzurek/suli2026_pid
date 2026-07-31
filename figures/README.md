# GUIDE TO NAVIGATING THE FIGURES

## AUDIT
audit contains histograms used to confirm the existance of variables used in analysis and training.

## BetaVsP
BetaVsP contains beta versus momentum plots with various cuts, these were generated with betaVsP.ipynb and EXPbetaVsP.ipynb
	
## Confusion Matrix
ConfusionMatrix/ contains various plots of the confusion matrix. The plot just called confusion matrix shows counts, while PC (purity contamination) shows results normalized to reoconstructed particles (rows add to 100%), and EM (efficiency mis-id) shows the matrix normalized to generated particles (colums add to 100%).

# DESCRIPTIONS OF BASELINES DIRECTORY

Baselines contains the plots that were used to create an initial plots of efficiency contamination, ect.
This README will detail how to navigate the plots along with everything in here.

## thetaCutPlots Summary
thetaCutPlots contains the efficiency contamination ect plots with 5 bins on theta. Pngs with Baseline in the name contain each of the five theta bins plotted on top of each other, while ones with a 2D in the name correspond to 2D heat maps showing binning on momentum and theta

These were generated with compute_baseline.ipynb

You won't find these in the actual thetaCutPlots, but rather in the sub chi directories, each has the same plots but with different chi2pid cuts.

- chi1 -> |chi2pid|<1 cuts
- chi2 -> |chi2pid|<2 cuts
- chi3 -> |chi2pid|<3 cuts
- chiVp -> Momentum Dependent chi2pid cut

## globalThetaCuts Summary
globalThetaCuts uses a base theta cut rather than binning on theta, here baseline efficiency contamination ect plots can be found.

These were generated with compute_baselineV2.ipynb

Since Contamination and Mis-ID have specific contributions made by protons and pions, they have been labeled TOTAL, where one can see each one's contribution to the total mis-id/contamination. If the individual contributions are needed as seperate plots, see individualParticleContributions/ where they are stored and labeled apropriately.


## feature_audit_week3
contains all feature audit plots used in deciding on the training variables.
kp_sidis contains the plots with the SIDIS cuts
each folder is a variable that has 9 plots

## full_range/evaluate_outputs/ 
This contains the cannonical plots from evaluate.py on the current full-range model

## full_range/FOM/
This contains the FOM plots used during optimization as pdfs primarily.


## Data_Application
contains plots relating to data validations done during week 8, epiN is it's own sub folder

## binary_MLP/
contains connonical results from the binary MLP model, contians comparisons and FOMs as sub folders


## otherPlots/
Contains refined binning on contaminations and efficiencies, these were presented at the Clas12 Collaboration meeting.

## optimized/
contains plots of FOM and optimized contaminations. EXP sub folder contains the same plots but with deviation from max FOM.
