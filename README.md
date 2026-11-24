# corporate-credit-early-warning

[![ci](https://github.com/rhozon/corporate-credit-early-warning/actions/workflows/ci.yml/badge.svg)](https://github.com/rhozon/corporate-credit-early-warning/actions/workflows/ci.yml)

Early warning of judicial and out-of-court recovery for Brazilian listed non-financial
companies, built only from public data: CVM filings and Central Bank series. Eight models,
from the Altman score to DeepHit, are fitted on one point-in-time panel and compared on one
out-of-time test. Distress signals read by language models from material-fact filings are
tested by ablation, with the primary test written down before the test period was opened
([PREREGISTRATION.md](PREREGISTRATION.md)).

> **Correction (2026-11-23).** An independent code review after the first publication found
> errors that changed published results: the going-concern rule also matched the
> auditor-responsibility paragraph that every audit opinion carries since 2016; the H3 sample
> kept rows with an open window only when they had ended in a filing; episodes still open in
> the registry were closed at their last filing; and trailing twelve months were wrong for
> fiscal years ending in March. Each was fixed with a test and the whole test was run again;
> hypotheses, models, split and primary test did not change. The fixes are listed in
> [PREREGISTRATION.md](PREREGISTRATION.md), the paper opens with an erratum, and every number
> below is from the corrected run. Three first-version conclusions changed: the hazard logit
> is a much stronger benchmark, the models now overestimate the 36-month rate, and the
> exploratory signal gain is gone.

> **Version 2 (2026-11-24).** Company names are now masked as whole words (a short name such as
> "Oi" used to cut "apoio"), and the registry's trade name is masked too. The 1,055 subjects whose
> masked text changed were read again and every result that uses the signals was run again; no
> conclusion changed. The version adds `results/drivers.csv` (Cox hazard ratios and XGBoost SHAP)
> and an exploratory comparison with the change of audit firm, holds each parallel call's cost
> estimate against the spending cap, and is recorded in [PREREGISTRATION.md](PREREGISTRATION.md).

The repository also holds the systematic literature review (PRISMA 2020) and the paper.

## What is and is not published

Only firms that already had a recovery or bankruptcy event are ever named next to a
probability, score or alert. The models score every listed company, but firm-level
predictions and the document-level distress labels stay in `data/private/`, which is not
tracked; `results/` holds aggregates. `tests/test_public_guard.py` fails the build if a
README, the paper or a results file names a company without an event within 200 characters of
a probability, score, ranking, alert or percentage, or if a published table pairs a company
identifier (name, CVM code or CNPJ) with a risk column.

## Quickstart

```bash
pip install -e ".[models,dev]"
pytest -v

python -m distress.cli events --sheet review/events.xlsx     # event base from CVM filings
python -m distress.cli suggest-review                       # second reader of each draft event
python -m distress.cli second-review                        # reviewed event base (fixed rule)
python -m distress.cli panel                                 # point-in-time firm-quarter panel
python -m distress.cli signals --workers 4                   # material-fact signals (two readers)
python -m distress.cli fulltext-sample --n 200               # what the subject line misses
python -m distress.cli validate                              # model choices inside the training years
python -m distress.cli test                                  # the out-of-time test (needs events_reviewed.csv)
python -m distress.cli slr screen|combine|extract|core      # systematic review
python -m distress.cli refs                                  # Crossref-verified bibliography
```

Raw CVM and Central Bank files are downloaded into `data/raw/` on first use. Language-model
answers are cached by a hash of model, prompt and schema, so a rerun makes no calls: unpack
`cache/llm_answers.jsonl.gz` with `distress.llm.client.unpack_cache` to rerun the signal
classification and its agreement without an API key. Answers that were not valid JSON are
cached too, so a rerun takes the same fallback; for the second reader's 18 such requests,
whose original answers were not kept, the cache holds a placeholder that fails the same way.
In the shipped copy, names of companies without an event are masked in the memo drafts
(`python -m distress.cli pack`). `python -m distress.cli guard <files>` scans any published
file, such as the site pages, before release.

## Data and events

- **Statements**: ITR (quarterly, 2011 onwards) and DFP (annual, 2010 onwards), consolidated
  where available. Each filing counts from its delivery date to the CVM (`DT_RECEB`), and a
  restatement only from its own delivery date. Rows whose latest statement is older than 550
  days are dropped.
- **Auditor's report**: a text rule flags the going-concern paragraph.
- **Registration forms (FCA)**: the audit firm each form lists, for the change of audit firm in
  the past twelve months. It is not a registered predictor and enters only an exploratory
  comparison.
- **Event filings (IPE)**: material facts, and the category used by companies in recovery.
- **Macro (SGS)**: Selic, IPCA, IBC-Br, unemployment, corporate non-performing loans, lending
  spread, each lagged by its publication delay.

An event is a filing for judicial recovery, a filing or court approval of out-of-court
recovery, or a bankruptcy. Candidates come from the registry status, the recovery category of
event filings and material facts that mention recovery; a language model (Qwen3-235B,
temperature 0) separates the company's own filing from ongoing proceedings, exits and other
companies' cases. Episode rules are in `distress/events.py`: an episode opens at the
company's own filing and closes at its exit; a gap of more than a year or an escalation
(out-of-court to judicial, or to bankruptcy) opens a new one; liquidations are excluded; firms
already in recovery in January 2011 are left out. Before the test is opened, a second reader
(Gemma 4 31B) reads the company's filings nearest each draft event since 2008 and says whether
the company itself filed, and when; a fixed rule (`distress/review_aid.py`) then keeps, moves or
drops the event (`data/events_second_reader.csv`, `data/events_reviewed.csv`).

## Models

All share one interface (`distress/models/`) and the same predictors (accounting ratios,
disclosure behaviour, macro): Altman Z'' (2005) with logistic calibration, discrete-time
hazard logit, Cox on stacked landmarks, XGBoost PD classifier, XGBoost survival (Cox
objective), random survival forest, DeepSurv and DeepHit. Survival models give PD curves at
12, 24 and 36 months.

Split: training quarters to 2018, an embargo year (2019), test from 2020. Choices were made on
a split inside the training years (fit to 2015, validate 2017 and 2018).

Evaluation (`distress/evaluate.py`, `distress/stats.py`): AUC, PR-AUC, KS, Brier and
calibration for the 12-month PD; Harrell and Uno C-index, time-dependent AUC and integrated
Brier score for the curves; firm-clustered bootstrap intervals; DeLong tests with Holm's
correction.

## Language-model signals

Material-fact subjects are classified into eleven distress categories (debt restructuring,
covenant, missed payment, asset sale for liquidity, emergency capital, going concern, CEO or
CFO change, rating downgrade, own recovery, subsidiary recovery, sanction) by Gemma 4 26B, with
the company's registered and trade names (as whole words) and tax number masked. A second model (Gemma 4 31B) reads a 3,000-document
subsample for agreement, and 200 documents are read in full to measure what the subject line
misses. The company's own recovery announcements are
never a predictor.

## Agentic credit memo

`distress/agents/`: an orchestrator calls deterministic tools (panel row, PD curve, drivers,
macro, recent filings) and two language-model subagents (reader, writer). A deterministic
critic checks every number and date in the draft against the tool outputs and blocks the memo
if one has no source.
`python -m distress.cli memos` writes memos on four past cases (Americanas, Ambipar, Raízen,
Casas Bahia), each as of twelve months before the filing, to `results/memos/`.

## Systematic review

`slr/protocol.md` (written before the search, with two dated amendments), records, both
screeners' decisions, PRISMA counts, abstract-level coding of every included study and
full-text coding of a stratified core. Every reference in `paper/references.bib` was resolved
on Crossref and accepted only when author, year and title matched.

## Results

Out-of-time test from 2020, 13,121 firm-quarters, 133 followed by a filing within 12 months
(`results/test_h1.csv`, `test_delong.csv`, `test_h3.csv`):

| Model | AUC, 12-month PD | 95% interval (firms resampled) |
|:---|---:|---:|
| XGBoost survival | 0.859 | 0.808 to 0.908 |
| Random survival forest | 0.852 | 0.799 to 0.902 |
| Cox | 0.849 | 0.788 to 0.907 |
| XGBoost PD | 0.845 | 0.783 to 0.901 |
| DeepSurv | 0.841 | 0.784 to 0.891 |
| Hazard logit | 0.816 | 0.741 to 0.885 |
| Altman score | 0.786 | 0.717 to 0.848 |
| DeepHit | 0.750 | 0.673 to 0.819 |

- H1: with DeLong's test and Holm's correction, Cox and XGBoost survival beat both
  benchmarks. With a paired bootstrap that resamples firms whole, the three tree ensembles
  beat the Altman score, but no model's edge over the hazard logit is significant. The
  ensembles stay ahead of the Altman score with the narrow event definition and, for XGBoost
  survival, in four of five years of an annual refit.
- H3: on rows whose window has closed, the observed rates are 1.0% at 12 months, 2.0% at 24
  and 2.9% at 36. At 36 months most models overestimate; XGBoost survival is closest
  (2.87%) and XGBoost PD underestimates. Cox has the lowest 36-month Brier score.
- H2, pre-registered: no gain from the signals (AUC 0.798 with, 0.816 without; one-sided
  DeLong p = 0.99). Winsorising at the 99th percentile had turned the five rarest signal
  counts into zeros; an exploratory rerun that keeps them gives 0.819 (p = 0.35; 0.45 with
  firms resampled), no gain either (`results/test_h2_exploratory.json`).
- Drivers (`results/drivers.csv`): in the XGBoost PD the largest SHAP contributions come from
  interest coverage, working capital over assets and quarter-ends with a twelve-month loss; in
  the Cox model, losses, the going-concern flag, negative equity and filing delay raise the
  hazard, and working capital and cash lower it.
- Change of audit firm, exploratory: added to the base set, it moves the AUC by between -0.056
  and 0.033 across the models, and no change is significant after Holm's correction
  (`results/test_auditor_exploratory.csv`).

## License

MIT.
