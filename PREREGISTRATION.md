# Pre-registration of the out-of-time test

Written and committed before the test period (quarters from 2020) was used for anything. The
commit that adds this file is the record of the date.

## Data and split

- Event base: `data/events_reviewed.csv`, the author's review of `data/events_draft.csv`. The
  command that opens the test (`python -m distress.cli test`) refuses to run without it.
- Training: firm-quarters up to 2018-12-31; outcomes observed up to 2019-12-31.
- Embargo: 2019, so that no 12-month outcome window of a training row reaches the test.
- Test: firm-quarters from 2020-01-01; 12-month PD metrics use rows whose window ends by the data
  date.
- Hyper-parameters: fixed inside the training years (fit to 2015, validate 2017 to 2018). Nothing
  is tuned on the test.

## Primary test (H2)

- Model: discrete-time hazard logit with an L2 penalty (`logit_hazard`).
- Baseline arm: accounting, disclosure and macroeconomic predictors (`experiment.BASE`), which
  include the going-concern flag from the text rule on the auditor's report.
- Treated arm: the baseline plus `experiment.SIGNALS`, the 12-month counts of each distress
  category read by the language model from material-fact subjects, and of any category (11
  columns). The company's own recovery announcements are never a predictor.
- Statistic: AUC of the 12-month PD on the test rows.
- Test: DeLong, one-sided (treated better), 5% level.

## Secondary analyses

All with Holm's correction within each family, and reported as secondary:

- H1: each model against the Altman score and against the hazard logit (DeLong, one-sided).
- H2 ablation: each other model with and without the signals.
- H3: calibration of the PD at 12, 24 and 36 months (observed rate, mean PD, Brier score,
  calibration intercept and slope) on rows whose window is observed.
- Intervals: 95% percentile intervals from 1,000 bootstrap samples of firms.
- Robustness: a narrow event definition (judicial recovery and bankruptcy only) and an annual
  refit with a moving origin over the test years.

## Amendment, 2026-11-21 (before the test period was opened)

The second reader's labels arrived after this file was committed. On the 3,000 subjects both
models read, agreement (`results/llm_agreement.json`) is kappa 0.60 or higher in eight categories
and lower in three: going concern (0.00; 11 flags from the first reader, 1 from the second),
subsidiary recovery (0.50) and sanction (0.51). One analysis is added; nothing above changes.

- Secondary H2 test on reliable categories: the same model, statistic and test as the primary,
  with the treated arm restricted to the 12-month counts of the categories with kappa of 0.60 or
  higher (`experiment.reliable_signals`), without the any-category count, which mixes in the
  categories left out. Seven columns: debt restructuring, covenant, missed payment, asset sale,
  emergency capital, executive change and rating downgrade.
- The threshold was chosen after the agreement values were seen and before any test row was
  used. Emergency capital (0.601) sits on it.
- The primary test stays as registered; this one is reported beside it, as secondary.
