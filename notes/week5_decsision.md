# Week five tier deicison

## DECISION RESULT
- The project should continue using the tier2 varaibles (chi2pid included)

## REASONING

Three seperate tiers of variables were analyized as following

- tier1| beta ftof_energy_1B ftof_time_1B ftof_path_1B
- tier2| beta ftof_energy_1B ftof_time_1B ftof_path_1B chi2pid ecin_path ecin_energy ecin_time
- tier3| beta ftof_energy_1B ftof_time_1B ftof_path_1B chi2pid ecin_path ecin_energy ecin_time ftof_energy_1A ftof_time_1A ftof_path_1A ecout_energy ecout_time ecout_path nphe_htcc

Unsurprisingly, tier1 performed the worst, with only having improvements in the contamination at high momentum when matched to the baseline chi2pid efficiencies, while 2&3 far surpased it in the other ranges.

Tier3 performed signifigantly better than tier 2, but upon further inspection showed that it was largely influenced by the variable ftof_time_1A, which failed the variable audit. As a result, this leaves tier2 as the model that performs the best, and would be applicable to non-mont-carlo data.
