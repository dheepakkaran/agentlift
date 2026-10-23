# Can LLM agents stand in for a real A/B test?

## Question

Before running a real e-mail experiment, can a population of LLM customer agents tell you which arm wins and roughly how large the lift will be?

## Setup

The real experiment is the Hillstrom MineThatData e-mail test: 64,000 customers randomized across Mens E-Mail, Womens E-Mail and No E-Mail, with website visits over the following two weeks as the outcome. Randomization checks out (max |SMD| 0.016, 0 covariates over the 0.1 threshold).

Each of 100 sampled customers becomes an agent, built from covariates only, and is asked the same question once per condition in separate calls. Outcomes and the real arm label never enter a prompt. The simulation is hashed and locked before it is compared with the real results, so the prediction is blind.

## Results

| Arm | Real ATE (pp) | Real 95% CI | Simulated ATE (pp) | Absolute error (pp) | Sign agrees |
| --- | ---: | :---: | ---: | ---: | :---: |
| Mens E-Mail | +7.66 | +7.00 to +8.32 | -2.89 | 10.55 | no |
| Womens E-Mail | +4.52 | +3.89 to +5.16 | -2.38 | 6.90 | no |

- Agents: **100** simulated customers, each run under all three conditions (300 agent responses).
- Winning arm: the simulation does not reproduce the real ordering of the three arms (Kendall tau -1.00).
- Primary metric: mean absolute ATE error **8.73 pp** across the two e-mail arms.
- Baseline calibration: the simulated no-e-mail visit rate is **57.78 pp** away from the real control rate of 10.62%.
- Segment fidelity: Spearman rho **-0.07** over 24 segment-by-arm comparisons with at least 15 simulated customers.
- Anchored to the real control rate (uses real data, so not a blind prediction): mean absolute ATE error **6.50 pp**.
- Relevance effect: agents lift visits by +5.0 pp when the e-mail matches the customer's buying history and -12.5 pp when it does not (men's arm); the mean matched-minus-mismatched gap across both arms is **19.3 pp**.
- Ablations: best variant `A1_persona` at 2.22 pp, worst `A3_temperature` at 9.21 pp.

## Where the simulation is reliable

- Relevance reasoning: agents separate a relevant e-mail from an irrelevant one by 19.3 pp, and their lift for a matching e-mail (+5.0 pp men's, +6.6 pp women's) is close to the real arm effects. The failure is not that agents ignore the e-mail, it is that they treat a mismatched one as a reason to stay away.

## Where it is not

- Arm ranking: the simulated ordering differs from the real one, so picking the winner is not safe here.
- Direction of effect: at least one arm's simulated sign is the opposite of the real one.
- Effect magnitude: mean absolute ATE error is 8.73 pp, too large to size an experiment from.
- Baseline level: the simulated no-e-mail rate is off by 57.78 pp, so absolute rates cannot be read off the simulation.
- Segment ordering: Spearman rho -0.07 against the real segment lifts.

## Recommendation

Do not use this simulation to choose an arm. It got the ordering wrong and missed the effect size by 8.73 pp on average. Its value here is as a prompt-and-persona testbed, not a prediction.

## Ablations

| ablation       | variant         | config                     |   mean_ate_abs_error_pp |   baseline_calibration_error_pp | arm_ranking_match   |   segment_spearman_rho |   reason_citation_share |
|:---------------|:----------------|:---------------------------|------------------------:|--------------------------------:|:--------------------|-----------------------:|------------------------:|
| baseline       | baseline        | large/full/neutral/t0.7    |                   5.234 |                          38.350 | False               |                 -0.314 |                   1.000 |
| A1_persona     | minimal persona | large/minimal/neutral/t0.7 |                   2.218 |                          60.950 | False               |                 -0.429 |                   1.000 |
| A2_model       | small model     | small/full/neutral/t0.7    |                   6.158 |                          32.783 | False               |                 -0.086 |                   1.000 |
| A3_temperature | temperature 0.0 | large/full/neutral/t0      |                   9.208 |                          67.350 | False               |                 -0.314 |                   1.000 |
| A4_framing     | realism framing | large/full/realism/t0.7    |                   8.874 |                          44.117 | False               |                 -0.257 |                   1.000 |

Spread of the primary metric across the 5 variants is 2.87 pp, from `A1_persona` to `A3_temperature`.

## Limitations

- The simulation covers 100 customers against 64,000 real ones, so its confidence intervals are wide.
- One dataset, one domain. E-mail response is not evidence about pricing, UI or other channels.
- The Hillstrom data is public and may sit in the model's training data, which would flatter these results rather than hurt them.
- Anchored numbers rescale the simulation using the real control rate, so they are a diagnostic, not a prediction.
- Agent reasons were scored with a fixed keyword list; 100% of visit reasons cite the e-mail or a persona feature.

_Mode: quick. Every number above comes from the committed files in `results/`._
