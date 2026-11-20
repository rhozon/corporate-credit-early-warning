# Systematic review protocol

Written and committed before the search was run. Reporting follows PRISMA 2020 (Page et al., 2021).

## Questions

1. Which model families (statistical, machine learning, deep learning, survival) have been used to predict
   corporate default, bankruptcy or financial distress, and how were they validated?
2. Has text from corporate disclosures, read by text mining or language models, been shown to add predictive
   information to accounting and market variables?
3. What is the evidence for Brazilian companies?

## Sources and search

- Database: OpenAlex (open, reproducible, API access). Crossref is used only to verify each cited DOI.
- Period: 2000 to 2026. Types: article, review, preprint. Languages: English and Portuguese.
- Strings, searched in title and abstract (exact strings in `distress/slr/review.py`, `STRANDS`):
  - (a) default or distress prediction with survival, machine learning, deep learning or neural network models;
  - (b) language models, text mining or NLP applied to credit risk, default or distress;
  - (c) Brazil and bankruptcy, distress, insolvency or judicial recovery.

## Eligibility

Included when all hold: empirical study; non-financial firms as the unit of analysis, or text and language
models applied to credit risk of firms or borrowers; outcome is default, bankruptcy, judicial recovery,
insolvency or distress; a predictive model with reported performance.

Excluded: consumer-only credit scoring without text methods; banks or insurers as the only subjects; sovereign
risk; return anomalies of distressed stocks; theory without data; editorials and narrative reviews (reviews are
read for background but are not part of the synthesis).

## Screening

Title and abstract are screened independently by two open-weights language models with the criteria above,
at temperature 0, with cached answers. Agreement is reported. Disagreements are resolved by a third model, and
every decision with its reason is published in `slr/screening.csv` for audit. No human screening is claimed.

## Data extraction

From included studies with open-access full text: country, data and period, number of firms and events,
model families compared, validation design (in-sample, cross-section holdout, out-of-time), metrics, use of
text, main finding. Studies without open access contribute their abstract only, and are flagged as such.

## Synthesis

Narrative synthesis with tables by model family, validation design and use of text, ordered by the impact of
the publication venue (OpenAlex citation statistics), which is reported but is not an inclusion criterion.
Gaps are stated against the three questions.

## Amendments

**Amendment 1, 2026-11-20 (before screening).** Anonymous access to OpenAlex exhausted its daily budget during
the first search (HTTP 401), before any strand was complete, so no OpenAlex record was used. The search was run
on the Semantic Scholar Academic Graph (bulk search endpoint) with the same three strings translated to its
syntax (`S2_STRANDS` in `distress/slr/review.py`) and the same period. That endpoint has no document-type
filter; editorials and non-empirical records are removed at screening.

**Amendment 2, 2026-11-20 (before screening).** The screening models are Gemma 4 26B-A4B and Gemma 4 31B
(open weights, Google AI Studio). Only two suitable models were available at no cost, so disagreements are not
resolved by a third model: a record either model includes goes on to full-text assessment (an inclusive rule
that protects recall). Eligibility is checked again on the full text of open-access records.
