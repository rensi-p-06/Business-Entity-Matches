# Business Entity Resolution — Methodology Document

> Note: the organizers' own `Documentation_template.md` file wasn't available
> in this environment, so this follows the four required sections listed in
> the problem statement PDF exactly. Drop your official template's content
> in if the section headings differ.

## 1. Methodology Used

We treat this as the standard **blocking → pairwise classification** entity
resolution pipeline:

1. **Blocking (candidate generation):** for every Source-1 record, generate a
   bounded set of plausible Source-2/Source-3 candidates using inverted-index
   token lookups, scoped by country (see §2).
2. **Feature engineering:** compute a fixed set of textual similarity
   features for every (S1, candidate) pair (see §3).
3. **Classification:** a gradient-boosted tree classifier scores each pair
   with a match probability; pairs above a tuned decision threshold become
   the final matches.
4. **Threshold selection:** tuned via 5-fold cross-validation over the
   Source-1 training entities (out-of-fold predictions), rather than a
   single holdout split, because F_0.5 is precision-heavy and a small
   holdout gives a noisy, unstable threshold estimate.
5. **Output:** one row per Source-1 test entity in both
   `matching_results.tsv` (final matches) and `candidate_pairs.tsv` (the
   candidate set the model actually scored).

No external data, APIs, or lookups are used anywhere in the pipeline — every
feature is derived purely from `business_name`, `business_address`, and
`country` as provided.

## 2. Candidate Generation / Blocking Strategy

Implemented in `src/blocking.py` as inverted-index token blocking:

- **Name-token index:** every "significant" token (≥3 chars) of the
  normalized business name (legal suffixes like Inc/Corp/Pvt/Ltd canonicalized
  first) maps to `(country, token) -> {entity_ids}`.
- **Name-prefix index:** the first 4 characters of each significant name
  token also form a bucket, to tolerate typos later in the word that would
  otherwise change the exact token (e.g. "Zephay" vs "Zephey").
- **Postal-code index:** `(country, postal_code) -> {entity_ids}`, from the
  best-effort trailing 5/6-digit number in the address (US ZIP / India PIN).
- **Address-token index:** `(country, address_token) -> {entity_ids}` for
  tokens ≥3 chars, after canonicalizing common street abbreviations
  (Road/Rd, Avenue/Ave, etc.).

For a Source-1 record, candidates are the **union** of everything found
across all four indices for its own tokens — a single shared significant
word (in either the name or the address) is enough to bring two records
together, which is what survives word-order changes, missing components, and
partial typos. Over-generic buckets (>500 members — a token so common it
isn't discriminative) are dropped so one common word can't blow up the
candidate set.

Candidates are then cheaply ranked by token overlap and capped at
`--max-candidates` (default 100) per Source-1 entity before feature
computation, purely to bound the pairwise-feature workload — the ranking
itself never decides matches, only which candidates get scored.

**Complexity:** this is effectively O(N) rather than O(N²) — a record's
lookup cost depends only on the (bounded) size of the buckets its own tokens
belong to, not on the size of the other source. This is what makes it
tractable at the ~1.7M-entity scale mentioned in the challenge PDF.

**Recall ceiling:** `train.py` reports, on every run, what fraction of true
training matches are even reachable as candidates — this is the hard upper
bound on achievable recall regardless of the downstream classifier, and is
the first thing to improve if overall F_0.5 is capacity-limited by blocking
rather than by classification.

## 3. Model Architecture and Feature Engineering

**Model:** `sklearn.ensemble.HistGradientBoostingClassifier`, trained from
scratch on the engineered pair features (300 boosting iterations, depth 6,
learning rate 0.08, L2 regularization). This is a classical model fit only on
this challenge's training pairs — it carries no license or pretrained-weight
dependency, so the "MIT/Apache-2.0, ≤8B parameters" constraint is trivially
satisfied. Positive pairs are upweighted by the inverse class ratio to
counter the natural imbalance (true matches are a small fraction of blocked
candidate pairs).

**Features** (12 total, `src/features.py`):

| Feature | Description |
|---|---|
| `name_exact` | Normalized business names are identical |
| `name_seq_ratio` | `difflib.SequenceMatcher` ratio on normalized names |
| `name_token_jaccard` | Jaccard similarity of name token sets (vectorized via multi-hot sparse matrices) |
| `name_len_diff` | Absolute character-length difference of normalized names |
| `name_common_prefix` | Length of the longest common prefix of normalized names |
| `name_tfidf_cosine` | Cosine similarity of char 2–4-gram term-frequency vectors (name) |
| `addr_token_jaccard` | Jaccard similarity of address token sets (vectorized via multi-hot sparse matrices) |
| `addr_tfidf_cosine` | Cosine similarity of char 2–4-gram term-frequency vectors (address) |
| `postal_match` | Extracted postal/PIN codes match |
| `postal_both_present` | Both records have an extractable postal code |
| `country_match` | Country strings match |
| `name_first_token_match` | Alphabetically-first name token matches (order-invariant) |

The two "TF-cosine" features are deliberately **plain term-frequency**
cosine similarity, not classic IDF-weighted TF-IDF (`use_idf=False`). A
per-dataset IDF fit would make cosine values incomparable between the
training corpus and the test corpus, silently shifting the classifier's
effective decision threshold at test time — see `src/features.py` for the
full rationale (this was caught and fixed during local validation).

Name-suffix normalization (Inc/Incorporated → `inc`, Pvt/Private → `pvt`,
etc.) and address-abbreviation normalization (Road → `rd`, Avenue → `ave`,
etc.) are applied before every text-based feature, in `src/normalize.py`.

## 4. Other Relevant Information

- **Local evaluation:** `src/evaluate.py` implements the exact macro-averaged
  F_0.5 formula from the problem statement, including the singleton rule
  (empty-true / empty-predicted scores 1.0).
- **Open-set country handling:** nothing in blocking or features hard-codes
  `{US, India}` — every country-scoped step keys on the raw country string,
  so a test-only label (e.g. France) is handled the same way automatically.
- **Reproducibility:** `code/business_entity_resolution/README.md` has the
  exact commands to retrain and regenerate both output files end-to-end.
- **Known limitation / next steps:** the last two features
  (`name_seq_ratio`, `name_common_prefix`) still use a per-pair Python loop
  — everything else (blocking, char n-gram cosine, token Jaccard, exact/
  length/postal/country comparisons) is vectorized. On a 300k-pair synthetic
  benchmark this loop cost ~7s, well behind blocking/prediction; if it
  becomes the bottleneck at full scale, dropping either feature (after
  checking permutation importance) or approximating them with the existing
  vectorized cosine features are the two easiest next steps.
