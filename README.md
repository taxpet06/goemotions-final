# When Do Emotion Classifiers Fail?

**Data scarcity, pretraining, and label ambiguity in 27-way emotion detection.**
CS 74 (Fall 2026) final project.

We compare four systems on GoEmotions (prior baseline, TF-IDF + logistic regression,
BiLSTM + GloVe, fine-tuned RoBERTa-base). Then we ask *where* they fail: how per-emotion F1
depends on training frequency and on human rater agreement (fitted jointly, because the
two are confounded), whether pretraining helps most on rare emotions, and what kinds of
errors the best model makes.

Results: `results/` (tables as CSV, `summary.json`, `figures/`). Slides: `slides/slides.pdf` (source
`slides/slides.html`, speaker script in `slides/speaker_notes.md`).

## Headline results (test, 5,385 comments)

| System | Macro-F1 [95% CI] | Micro-F1 | Macro ROC-AUC |
|---|---|---|---|
| Prior | 0.042 | 0.126 | 0.50 |
| TF-IDF + LR | 0.457 [0.433, 0.476] | 0.530 | 0.90 |
| BiLSTM + GloVe (3 seeds) | 0.498 ± 0.005 [0.470, 0.518] | 0.572 | 0.92 |
| RoBERTa-base (3 seeds) | 0.534 ± 0.003 [0.510, 0.552] | 0.602 | 0.93 |

- Rater agreement (Krippendorff's α) and log training frequency are correlated (r = 0.53). Fitted jointly,
  agreement predicts per-emotion F1 (RoBERTa: +0.16 F1 per SD, R² = 0.89) and frequency adds nothing.
- RoBERTa's lead over TF-IDF is larger on the 8 rare emotions than on frequent ones by +0.05 at full data
  (CI crosses 0) and by +0.09 at 10% data, where RoBERTa with 4.3k comments roughly matches TF-IDF with all 43k.
- Most errors are emotion-vs-neutral, not confusions between sibling emotions.

## Team and contributions

We did not use GitHub to manage our work, so the commit history does not show who did what.
Every member contributed equally, according to the division of work in our project proposal and our
individual contribution surveys.

## Layout

| File | What it does | Runs on |
|---|---|---|
| `prep.py` | Cleans the official split: drops 80 dev/test comments duplicated in train and 5 test comments duplicated in dev; adds Ekman labels | CPU, seconds |
| `common.py` | Tokenizer, loading, per-label threshold tuning, F1, Krippendorff's α, bootstrap indices | n/a |
| `baselines.py` | Prior, always-neutral, TF-IDF + LR grid (C × n-gram), TF-IDF learning curve and Ekman runs | CPU, ~15 min |
| `neural.py` | One BiLSTM or RoBERTa run (model, lr, seed, label set, train fraction) | GPU |
| `cluster/` | Scripts that ran `neural.py` on Dartmouth Discovery (`runs/*.sh` list every run) | n/a |
| `analysis.py` | Model selection, test metrics, bootstrap CIs, agreement, joint regression, gap test, threshold sweep, learning curves, Ekman, confusions, figures, error sample | CPU, ~2 min |
| `error_summary.py` | Summarises the hand-coded error sample: category breakdown and Fleiss' κ | CPU |
| `test_common.py` | Checks F1, thresholds, tokenizer and Krippendorff's α against sklearn / `krippendorff` | CPU |

## Reproduce

```bash
# 1. Data (Apache 2.0; a copy of the raw files we used is in data/raw/)
B=https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data
for f in train.tsv dev.tsv test.tsv emotions.txt ekman_mapping.json; do curl -sSfLo data/raw/$f $B/$f; done
for i in 1 2 3; do curl -sSfLo data/raw/goemotions_$i.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_$i.csv; done

# 2. CPU environment
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python test_common.py          # prints "ok"
.venv/bin/python prep.py
.venv/bin/python baselines.py

# 3. Neural runs (GPU). Every run we did is one line in cluster/runs/*.sh, e.g.
pip install -r requirements-gpu.txt      # torch 2.7.1 (cu126 wheels), transformers 4.46.3
python neural.py roberta --lr 2e-5 --seed 0
python neural.py bilstm --hidden 256 --lr 1e-3 --seed 0 --glove glove.6B.300d.txt   # GloVe 6B from nlp.stanford.edu

# 4. Analysis (needs outputs/preds from steps 2-3; our predictions are included)
.venv/bin/python analysis.py
.venv/bin/python error_summary.py        # summarises our hand coding in error_analysis/errors_to_code.csv
```

`outputs/preds/` holds every model's dev and test sigmoid scores (`<system>__seed<k>__<split>.npy`),
so `analysis.py` reproduces every number in `results/` without retraining.
`outputs/logs/` has per-epoch dev scores for each neural run, plus the GPU it ran on.

### Running on Discovery

GPU nodes cannot read the cluster home directory, so `cluster/launch.sh JOB` streams the code and data
into the job as a tar on stdin, builds a venv in the node's `/tmp`, runs `cluster/runs/JOB.sh`, and streams
`outputs/` back as a tar on stdout (`cluster/fetch.sh JOB` unpacks it). Jobs used the free account on the
`gpu_preempt` partition, which assigns whatever GPU is free (our runs landed on V100, RTX 2080 Ti, L40S and H200).
The GPU is recorded in each run's log. We did not set cuDNN determinism flags, so reruns on other hardware
agree only to within seed noise.

## Protocol (what touches what)

- **Train**: model fitting only.
- **Dev**: hyperparameter grid, epoch selection (early stopping), per-label thresholds (grid 0.01 to 0.99).
- **Test**: evaluated once per final model, after all choices were fixed. Neural results are mean ± SD over 3 seeds.
- Bootstrap: 1,000 resamples of test comments, shared across systems (paired).
