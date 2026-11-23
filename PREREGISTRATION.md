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

## Amendment, 2026-11-22 (before the test period was opened)

The event base was to be the author's review of the draft list. It is instead the check by a
second reader under a fixed rule, which anyone can rerun from the cached answers. Nothing else
changes.

- Reader: Gemma 4 31B, temperature 0, reads up to four of the company's filings nearest each
  draft event from 2008 on, with the company's tax number, and answers whether this company
  filed and on what date (`data/events_second_reader.csv`).
- Rule (`review_aid.rule_sheet`): an event the reader confirms, or dates within seven days,
  stands; one it dates elsewhere moves to that date, or goes if the company already has an event
  within seven days of it; one it attributes to another company goes; where the reader cannot
  tell, the first reader's decision stands.
- `python -m distress.cli second-review` writes `data/events_reviewed.csv`.

## Corrections, 2026-11-23 (after the test period was opened)

An independent review of the code, after the results were published, found errors in how
several registered quantities were computed. They were corrected and the whole test was run
again; the published results are the corrected run. The hypotheses, models, primary test,
split, metrics and thresholds above are unchanged. Each correction is a fix to code that did
not do what this file describes, and each is tested; none was chosen by looking at a test
result.

- Going-concern flag. The text rule also matched the auditor-responsibility paragraph that every
  audit opinion carries since the 2016 reports, so it flagged about 95% of annual filings from
  then on. The rule now accepts only statements of a material uncertainty about the filer
  (`features.going_concern_rule`).
- H3 sample. Rows whose window had not closed by the data end were kept only when they ended in
  a filing, which overstated the observed rates. Rows are now chosen by date alone
  (`experiment.calibration_by_horizon`).
- Episodes. An episode holding the registry's current status was closed at its last filing, so
  firms still in recovery went back to the risk set, and some firms in recovery since before 2011
  were not left-truncated. Such episodes now stay open, a bankruptcy never ends, and a review that
  rejects an episode's first document reopens it at the company's next own filing
  (`events.build_events`, `events.apply_review`).
- Trailing twelve months for firms whose fiscal year does not end in December
  (`features.point_in_time`).
- Added beside DeLong, not replacing it: a paired bootstrap that resamples firms whole, for every
  AUC comparison, since a firm's quarters are not independent.
