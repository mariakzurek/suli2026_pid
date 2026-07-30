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


