## Introduction
In semi inclusive deep inelastic scattering experiments, the study of hadrons give insight into the structure of the proton. 
Of particular interest is kaons, which give access to the strange quark. At high momentum ranges, the signitures of Kaons and Pions become similar, resulting in lower purity of K+ samples. Current methods use a momentum based chi2pid cut, where chi2pid is the number of signed sigma between the measured and calculated vertex times. 
In order to improve the purity, a machine learning classifier (gradient boosted decsion tree (BDT)) was developed, with the goal of having lower contamination, whilst maintaining efficiency. The model was trained using Monte Carlo hipo files from Clas-Dis. The model was tested against a baseline method using cuts on chi2pid. A second family of ML, a small neural MLP network, was also tested but performed similar to the BDT model. Contamination results were validated using two methods: contamination using a Ring Image Cherenkov Detector (RICH) as a truth, and using an exclusive epi(N) reaction to find pione->kaon misidentification, which was used to estimate contamination. This repo also contains a small MLP neural network model which was found to perform similar to the BDT model.



# Feature Audit

Since machine learning models can only be trained on simulated MC data, due to needing a truth, a feature audit must be done to determine what MC variables agree with data.
The audit comprises of looking at a 3 theta bins and 3 momentum bins for a total of 9 bins. In each bin several statistical tests are perfomred including, Wasserstien Norm, Population Stability Index, Max Local Residue, Kolmogorov-Smirnov and Chi2. These are then culiminated as a final drift decision which is KEEP, CANDIDATE, or DROP, If all bins are KEEP then the variable agree very well, if it is CANDIDATE, it should be reviewed manually, and if several bins are DROP, it most likely agrees very poorly. In all cases a manual check should be done just in case.

See the example variable of momentum:

../figures/feature_audit_week3/kp_sidis/

<table>
  <tr>
    <td><img src="../figures/feature_audit_week3/kp_sidis/p/p_p1-2_theta5-15.png" width="300"></td>
    <td><img src="../figures/feature_audit_week3/kp_sidis/p/p_p1-2_theta15-25.png" width="300"></td>
    <td><img src="../figures/feature_audit_week3/kp_sidis/p/p_p1-2_theta25-35.png" width="300"></td>
  </tr>
  <tr>
    <td><img src="../figures/feature_audit_week3/kp_sidis/p/p_p2-3_theta5-15.png" width="300"></td>
    <td><img src="../figures/feature_audit_week3/kp_sidis/p/p_p2-3_theta15-25.png" width="300"></td>
    <td><img src="../figures/feature_audit_week3/kp_sidis/p/p_p2-3_theta25-35.png" width="300"></td>
  </tr>
  <tr>
    <td><img src="../figures/feature_audit_week3/kp_sidis/p/p_p3-5_theta5-15.png" width="300"></td>
    <td><img src="../figures/feature_audit_week3/kp_sidis/p/p_p3-5_theta15-25.png" width="300"></td>
    <td><img src="../figures/feature_audit_week3/kp_sidis/p/p_p3-5_theta25-35.png" width="300"></td>
  </tr>
</table>



# BDT Classifier
The feature list of the BDT model used in this project is: beta, ftof_energy_1B, ftof_time_1B, ftof_path_1B, chi2pid, ecin_path, ecin_energy, ecin_time.
The hyperparameters used in the BDT model was 200 trees, max depth of 6, learning rate of 0.05
The model aslo made use of a platt calibration that was performed during training, and the model produced the following reliability diagram.

![Main plot](../figures/full_range/reliability_diagram.png)

Once the classifier was trained, the feature of merit (FOM), which quantifies the ability to identify kaons from the statistical background was optimized. The model in this project used the highest BDT threshold within a 3% deviation from the max FOM, which biases towards lower contaminations at the cost of some efficiency. 

# BDT MC results

The following section will outline the BDT's Performance on MC data:

BDT at constant efficiency (80%):
![Main plot](../figures/full_range/evaluate_outputs/contamination_fixed_eff_theta.png)

BDT at Matched chi2pid efficiency:
![Main plot](../figures/full_range/evaluate_outputs/low_5-11_ml_compare_chi2pid_contam.png)


At matched efficiency to the chi2pid method, the BDT was able to perform with much lower contaminations, while not decreaing the efficiency further.

After FOM optimization The contamination and efficiency performed as such:

![Main plot](../figures/binary_MLP/comparisons/efficiency_All_Three.png)
![Main plot](../figures/binary_MLP/comparisons/contamination_All_Three.png)

here the small neural MLP is also shown, which performed nearly identical to the BDT after FOM optimization.
SSSS
# Data Validations

The data validations used to methods. The first was computing contaminations using a RICH (Ring Image Cherenkov Detector) as the truth, and the second being the use of an exclusive epi(N) reaction sample to estimate contamination from mis-Id and pion efficiency.

eKX RICH truth methodology, due to the immense accuracy of the RICH, the contamination was calculated using the RICH as a form of "experimental truth". This was compared to MC in the same kinematic range of the RICH
work/suli2026_pid/figures/Data_Application/bdt_rich_contamination.png
![Main plot](../figures/Data_Application/bdt_rich_contamination.png)
![Main plot](../figures/Data_Application/contam_rich_mc.png)

The second method involved the use of the epi(N) exclusive reaction channel. This reaction, due to conservation of strangeness, cannot have kaons (eK(N) is not possible), as a result everything in this dataset is a true pion. Using the BDT, the ratio of the reconstructed kaons over the total number of events in the sample corresponds to the pion->kaon misidentification rate. Similarly the number of EB pions over the total number gives the EB pion efficiency. By taking the SIDIS EB pion events and dividing by the efficiency per bin gives an aproximation for true pions, then multiplying by pion->kaon misId gives a number of fake kaons, which is then used to aproximate contamination.
![Main plot](../figures/Data_Application/epiN/kaon_contamination_vs_p.png)
![Main plot](../figures/mc_Data_Application/epiN/kaon_contamination_vs_p.png)

These methods both show a similar performance to the MC tests within the respective kinematic regionse. Showing that the BDT improves contaminations in data as well.

# Systematic Uncertainties

From the files produced throughout this analysis, several systematic uncertainty sources were identified, these can be found in notes/systematics_summary.md as tables. The following plots are for the MC 
![Main plot](../notebooks/uncertainty/pngs/uncertainty_vs_momentum_MC.png)
![Main plot](../notebooks/uncertainty/pngs/contamination_with_syst_MC.png)

Here the reweighting contributes to the total uncertainty in high momentum, future studies should apply p-theta reweighting in training. The largest source of systematic uncertainty is with the BDT thresholds.

These next plots are for the RICH validations:
![Main plot](../notebooks/uncertainty/pngs/uncertainty_vs_momentum_RICH.png)
![Main plot](../notebooks/uncertainty/pngs/contamination_with_syst_RICH.png)

The RICH RQ had signifigant effect on the systematic uncertainty, future studies should be selective with their rich_RQ range when reproducing this study.

The next plots are for the exclusive epi(N) reaction validations:

![Main plot](../notebooks/uncertainty/pngs/uncertainty_vs_momentum_MX.png)
![Main plot](../notebooks/uncertainty/pngs/contamination_with_syst_MX.png)

The systematic uncertainty is highly sensitive to the Mx fitting window, epiN_analysis.py should be used to exam each individual fit when interpreting results during development.


# Conclusion and Future

Overall the Binary BDT Machine Learning classifier was able to improve contamination and efficincy results compared to the baseline chi2pid cutting method: Improving contamination by up to ~10% in low momentum bins (0.5-2.5 GeV/c), and was able to achieve a relatively consistant efficiency around 80% on the full momentum range. The MLP model performed nearly identical to the BDT, and with it's increased processing times make the BDT the more viable option in this current state. Both Data validations confirm the BDT's performance in how the contamination varies across momentum bins (within the ranges these tests are valid). This tool will allow the studies of the ep->epKX channel to have a purer sample, with higher statistics than would otherwise be avalible using standard event-builder cutting methods.

Future analysis should focus on the following:
- retraining with rewieghting based on the Mc/data populations (from the auditting phase)
- Tuning of hyperparameters for both the BDT and MLP families
- Creation of a multicalss identifier for MLP and BDT, as opposed to the current binary models.
