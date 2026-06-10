# VARIABLE AUDIT RESULTS

## Tier Defininitions
- Tier1 | Variables that passed the automatic audit as candidates, no issue
- Tier2 | Variables that failed the automatic audit, but still are usable under further inspection
- Tier3 | Variables that failed the automatic audit, but follow a general trend of the data


## Tier1 Variables
- ftof_energy_1B
- ftof_time_1B
- ftof_path_1A
- ftof_path_1B
- ecin_path
- ecout_path
- pcal path
- Beta
- chi2pid
- p
- theta

NOTES: ftof_1B is fully usable with teir 1 alone, but 1A should not be done without teir 2 variables. Kineamtic variables Beta - theta, should not be used in training but effect weighting

## Tier2 Variables
- ftof_energy_1A
- nphe-hhtc
- ecin_energy
- ecin_time



## Tier3 Variables
- ftof_time_1A
- ecout_energy
- ecout_time

