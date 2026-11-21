"""Shared helpers: data loading, tokenizer, threshold tuning, metrics, bootstrap.

Used by every model script and by analysis.py, so all systems see the same
splits, the same tokenization and the same evaluation code.
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
PREDS = ROOT / "outputs" / "preds"

EMOTIONS = (ROOT / "data" / "raw" / "emotions.txt").read_text().split()  # 27 emotions + neutral, label-id order
NEUTRAL = EMOTIONS.index("neutral")
EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# [NAME]/[RELIGION] masks kept whole; words; and every other non-space char
# (emoji, "!", "?") as its own token. Used by TF-IDF and the BiLSTM.
_TOKEN = re.compile(r"\[name\]|\[religion\]|\w+|[^\w\s]")


def tokenize(text):
    return _TOKEN.findall(text.lower())


def load_split(split, labels="goemotions"):
    """Return (texts, Y) for a cleaned split. Y is an int8 (n, 28) or (n, 7) matrix."""
    rows = [json.loads(l) for l in (CLEAN / f"{split}.jsonl").open()]
    texts = [r["text"] for r in rows]
    key, k = ("labels", len(EMOTIONS)) if labels == "goemotions" else ("ekman", len(EKMAN))
    Y = np.zeros((len(rows), k), dtype=np.int8)
    for i, r in enumerate(rows):
        Y[i, r[key]] = 1
    return texts, Y


# ---------------------------------------------------------------- metrics

def f1_per_label(Y, P):
    """Per-label F1 for binary matrices Y (gold) and P (pred). F1 = 0 when a label has no gold and no pred."""
    tp = (Y & P).sum(0).astype(float)
    fp = ((1 - Y) & P).sum(0)
    fn = (Y & (1 - P)).sum(0)
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(Y, P):
    return f1_per_label(Y, P).mean()


def micro_f1(Y, P):
    tp = (Y & P).sum()
    return 2 * tp / (Y.sum() + P.sum())


def kripp_alpha_binary(m, n1):
    """Krippendorff's alpha for one binary label. m = raters per comment, n1 = raters who chose it."""
    keep = m >= 2
    m, n1 = m[keep].astype(float), n1[keep].astype(float)
    n0 = m - n1
    o01 = (n0 * n1 / (m - 1)).sum()
    N1 = (n1 * (n1 - 1) / (m - 1)).sum() + o01
    N0 = (n0 * (n0 - 1) / (m - 1)).sum() + o01
    n = N0 + N1
    return 1 - (n - 1) * o01 / (N0 * N1)


GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def tune_thresholds(Y, S):
    """Per-label threshold maximising dev F1 (Lipton et al. 2014). Y gold, S scores, both (n, k)."""
    best = np.full(Y.shape[1], 0.5)
    best_f1 = np.full(Y.shape[1], -1.0)
    for t in GRID:
        f = f1_per_label(Y, (S >= t).astype(np.int8))
        better = f > best_f1
        best[better], best_f1[better] = t, f[better]
    return best


def predict(S, thr):
    return (S >= thr).astype(np.int8)


def bootstrap_idx(n, B=1000, seed=0):
    """B resamples of row indices, shared across models so differences are paired."""
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def load_preds(name, split):
    """All seeds' scores for a model: array (seeds, n, k)."""
    files = sorted(PREDS.glob(f"{name}__seed*__{split}.npy"))
    assert files, f"no predictions for {name} {split}"
    return np.stack([np.load(f) for f in files])
