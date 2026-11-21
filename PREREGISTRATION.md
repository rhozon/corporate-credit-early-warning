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
