# Report for week1 and week2

**Student:** Cooper Bell (SULI 2026)
**PI:** Maria Zurek (ANL)

##Terminology
Anyone who is familar with a tuple in mathmatics may recognize an ntuple as an ordered set of n elements. In this project the concept an ntuple isn't much different. For all practical purposes, an ntuple is a row in the dataframe, corresponding to an event. 
Each column is a property such as momentum, pid, ect. It follows then that the row count is the number of events in the selection. 
Each colum is associated with class, which is some form of data-label the machine laerning will use to identify related data.
Class balance is a process in which majority classes (class labels that apear the most frequently) are removed, this allows the minority classes to have a larger impact and make the machine learning less biased.
It should be noted that RICH banks have been populated.

##




# SULI 2026 — SUMMARY OF Efficiency PLOTS
**Student:** Cooper Bell (SULI 2026)
**PI:** Maria Zurek (ANL)

## SUMMARY OF BASE LINE
Overall the momentum based cut combined with lower valued theta bins (no 30-35) will serve as the baseline as that produces the highest efficiency and purity and the lowest contaminations and mis-id.

## Base Efficiency plots
Under figures/efficiencies/globalCuts/ there are plots are general plots of efficiency and purity (with momentum cuts 0.5-5, and theta 5-35). There are also contamination and misID plots, ones that have a particle name appended only show that particles contribution to K+. Files with TOTAL contain total contamination/mis-id along with the contributions (Note how the contributions add to the total)

## BEST THETA OPERATION
Theta ranges generally gave better efficiencies and purities at lower ranges of theta across the whole momentum range. Theta range 29-35 has poor statistics and should be avoided

Locations:
Efficiency Plot with theta cuts: figures/efficiency/chiCut3/basic/efficiencyKBaseline.png
Purity Plot with theta cuts: figures/efficiency/chiCut3/basic/efficiencyKBaseline.png
Contamination Plot with theta cuts: figures/efficiency/chiCut3/basic/contaminationKBaseline.png
Mis-ID Plot with theta cuts: figures/efficiency/chiCut3/basic/misIdKBaseline.png

## BEST CHI2PID PERFORMANCE
Chi2cuts of all cuts performed similarlly to each other within low momentum ranges. The Dynamic chi2pid cut as a function of momentum performed the best with efficiencies and purity at high momentum. It is important to note however it fails past 4.25 GeV/c due to having too low of statistics to properly give a value (This can be seen by the giant error bars and the setting to zero). To avoid this above 4GeV/c one may want to increase the number of events in their sample in order to improve the number of events, or change binning, primarliy one just needs a higher number of events in the bins past 4 GeV/c.

Locations:
Efficiency: figures/efficiency/globalCut/CHIefficiencyK.png
Mis-ID: figures/efficiency/globalCut/CHImisIdK.png
Purity: figures/efficiency/globalCut/CHIpurityK.png
Contamination: figures/efficiency/globalCut/CHIcontaminationK.png

