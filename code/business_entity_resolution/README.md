# Business Entity Resolution — Amazon ML Challenge 2026

A blocking + classical-ML-classifier pipeline that matches Source-1 business
records against Source-2/Source-3 candidates using only the fields provided
(business_name, business_address, country). No external lookups, APIs, or
reference data are used anywhere in this pipeline.

> **`../../dataset/` in this package is synthetic demo data**, generated
> locally because the real challenge dataset was too large to hand off in
> this environment — it exists only so `output/matching_results.tsv` and
> `output/candidate_pairs.tsv` in this package are real, reproducible
> results of this exact code (out-of-fold train F_0.5 ≈ 0.69, held-out
> synthetic test F_0.5 ≈ 0.88 — see the training log). Swap in the real
> `dataset/train/` and `dataset/test/` and rerun the two commands below.

## Pipeline overview

```
dataset/{train,test}/*.tsv
        │
        ▼
  src/blocking.py     — inverted-index token blocking (name/address tokens,
                         postal code, country) -> candidate pairs per S1 entity
        │
        ▼
  src/features.py     — 12 pairwise similarity features per (S1, candidate)
                         pair: char n-gram cosine, token Jaccard, sequence
                         ratio, postal/country/prefix agreement, etc.
        │
        ▼
  src/model.py         — HistGradientBoostingClassifier trained from scratch
                         on labeled candidate pairs (train_ground_truth.tsv)
        │
        ▼
  src/train.py / src/predict.py — orchestrate the above; predict.py writes
                                    output/matching_results.tsv and
                                    output/candidate_pairs.tsv
```

See `Documentation_template.md` (one level up) for the full methodology
write-up: blocking strategy, feature list, model architecture, and results.

## Setup

```bash
cd code/business_entity_resolution
pip install -r requirements.txt --break-system-packages   # or use a venv
```

## Reproducing end-to-end

Directory layout expected (adjust `--train-dir` / `--test-dir` if yours differs):

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

1. **Train** (5-fold cross-validated threshold selection, then refit on all
   training data — see `src/train.py` docstring for why):

   ```bash
   cd src
   python3 train.py \
       --train-dir ../../../dataset/train \
       --model-out ../models/model.joblib \
       --n-folds 5
   ```

   Prints the blocking recall ceiling, the out-of-fold macro F_0.5 estimate,
   and the chosen decision threshold.

2. **Predict** on the test set:

   ```bash
   python3 predict.py \
       --test-dir ../../../dataset/test \
       --model-path ../models/model.joblib \
       --output-dir ../../../output
   ```

   Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

3. **Validate before submitting**:

   ```bash
   cd ..
   python3 utils/validate_submission.py \
       --matching ../../output/matching_results.tsv \
       --candidate ../../output/candidate_pairs.tsv \
       --test-dir ../../dataset/test
   ```

## Key design notes / gotchas already handled

- **Country is an open set.** Blocking and features never hard-code
  `{US, India}` — France (or any other test-only country label) is handled
  automatically since every country-scoped key is just `(country_string, ...)`.
- **TF-cosine features use `use_idf=False`.** A per-dataset IDF fit would make
  the train-fit cosine scale incomparable to the test-fit cosine scale,
  silently miscalibrating the classifier's threshold at test time (caught
  during local testing on synthetic data — see `src/features.py` docstring).
- **Threshold is chosen via 5-fold out-of-fold CV**, not a single holdout
  split, because a small holdout is noisy under the precision-heavy F_0.5
  metric and can pick an unstable threshold.
- **Blocking recall ceiling is always printed** by `train.py` — the fraction
  of true matches that even *could* be found given the candidate-generation
  strategy, since that upper-bounds achievable recall regardless of the model.

## Scaling to the full dataset

Both the blocking stage and the feature-computation stage are vectorized so
they scale roughly linearly rather than blowing up on the full test set:

- **Blocking** (`src/blocking.py`) is O(N): inverted-index lookups only
  touch a record's own token buckets. The candidate-ranking step uses plain
  Python dicts for the score lookup rather than repeated `DataFrame.loc[id]`
  calls in a loop — the latter was measured at ~100s for 300k pairs during
  local benchmarking (see the comment in `generate_candidates`); switching
  to dicts dropped that to ~3s for the same workload.
- **Features** (`src/features.py`): the two similarity families that
  dominate feature count — char n-gram cosine and token Jaccard — are each
  computed as a single vectorized sparse-matrix operation over *all* pairs
  at once (fancy-indexing + row-wise multiply-and-sum), not a per-pair Python
  loop. The exact-match/length/postal/country features are plain NumPy array
  comparisons. Only two features (`name_seq_ratio`, `name_common_prefix`)
  have no convenient vectorized form and stay in a lightweight per-pair loop.

On a synthetic benchmark of ~300k candidate pairs (3,000 S1 entities x ~100
candidates each), end-to-end inference (blocking + features + predict) runs
in a few seconds rather than the ~2 minutes it took before these two fixes.
