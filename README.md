# corporate-credit-early-warning

[![ci](https://github.com/rhozon/corporate-credit-early-warning/actions/workflows/ci.yml/badge.svg)](https://github.com/rhozon/corporate-credit-early-warning/actions/workflows/ci.yml)

Early warning of judicial and out-of-court recovery for Brazilian listed non-financial
companies, built only from public data: CVM filings and Central Bank series. Eight models,
from the Altman score to DeepHit, are fitted on one point-in-time panel and compared on one
out-of-time test. Distress signals read by language models from material-fact filings are
tested by ablation, with the primary test written down before the test period was opened
([PREREGISTRATION.md](PREREGISTRATION.md)).

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
`cache/llm_answers.jsonl.gz` with `distress.llm.client.unpack_cache` to reproduce every label
without an API key.

## Data and events

- **Statements**: ITR (quarterly, 2011 onwards) and DFP (annual, 2010 onwards), consolidated
  where available. Each filing counts from its delivery date to the CVM (`DT_RECEB`), and a
  restatement only from its own delivery date. Rows whose latest statement is older than 550
  days are dropped.
- **Auditor's report**: a text rule flags the going-concern paragraph.
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
the company's names and tax number masked. A second model (Gemma 4 31B) reads a 3,000-document
subsample for agreement, and 200 documents are read in full to measure what the subject line
misses. The company's own recovery announcements are
never a predictor.

## Agentic credit memo

`distress/agents/`: an orchestrator calls deterministic tools (panel row, PD curve, drivers,
macro, recent filings) and two language-model subagents (reader, writer). A deterministic
critic checks every number and date in the draft against the tool outputs and blocks the memo
if one has no source.

## Systematic review

`slr/protocol.md` (written before the search, with two dated amendments), records, both
screeners' decisions, PRISMA counts, abstract-level coding of every included study and
full-text coding of a stratified core. Every reference in `paper/references.bib` was resolved
on Crossref and accepted only when author, year and title matched.

## Results

To be filled from `results/` after the out-of-time test.

## License

MIT.
