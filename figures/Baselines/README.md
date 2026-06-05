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


