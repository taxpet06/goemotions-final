"""Checks the hand-written metric code against reference libraries. Run: python test_common.py"""
import krippendorff
import numpy as np
from sklearn.metrics import f1_score

from common import f1_per_label, kripp_alpha_binary, macro_f1, micro_f1, tokenize, tune_thresholds

rng = np.random.default_rng(0)
Y = (rng.random((500, 6)) < [0.5, 0.2, 0.05, 0.3, 0.01, 0.1]).astype(np.int8)
P = (rng.random((500, 6)) < 0.2).astype(np.int8)
assert np.allclose(f1_per_label(Y, P), f1_score(Y, P, average=None, zero_division=0))
assert np.isclose(macro_f1(Y, P), f1_score(Y, P, average="macro", zero_division=0))
assert np.isclose(micro_f1(Y, P), f1_score(Y, P, average="micro"))

# Thresholds: a perfectly separable score gets F1 = 1 on every label.
S = Y * 0.9 + 0.05
assert macro_f1(Y, (S >= tune_thresholds(Y, S)).astype(np.int8)) == 1.0

# Krippendorff's alpha (binary, variable raters per comment) vs the reference package.
for p in (0.05, 0.3):
    m = rng.integers(2, 6, 300)
    truth = rng.random(300) < p
    votes = [[int(t) if rng.random() < 0.8 else int(rng.random() < p) for _ in range(k)] for t, k in zip(truth, m)]
    table = np.full((5, 300), np.nan)  # raters x units
    for u, v in enumerate(votes):
        table[: len(v), u] = v
    ref = krippendorff.alpha(reliability_data=table, level_of_measurement="nominal")
    ours = kripp_alpha_binary(m, np.array([sum(v) for v in votes]))
    assert np.isclose(ours, ref), (ours, ref)

assert tokenize("[NAME] is SO funny 😂!!") == ["[name]", "is", "so", "funny", "😂", "!", "!"]
print("ok")
