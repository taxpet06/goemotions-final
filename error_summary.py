"""Summarise the HAND-CODED error analysis (run after the team fills error_analysis/errors_to_code.csv).

Categories (one per row, lower-case): label_noise, sarcasm, adjacent_emotion, missing_context, model_error.
Shared rows (coder == "shared") are coded by all three; agreement on them is reported as
Fleiss' kappa, and their final category is the majority vote. Three-way ties are listed
for discussion and left out of the breakdown.
"""
import json
from collections import Counter

import matplotlib
import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from common import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CATS = ["label_noise", "sarcasm", "adjacent_emotion", "missing_context", "model_error"]
COLS = {f"coder{i}": f"category_coder{i}" for i in (1, 2, 3)}
df = pd.read_csv(ROOT / "error_analysis" / "errors_to_code.csv").fillna("")
for c in COLS.values():
    df[c] = df[c].str.strip().str.lower()
    bad = set(df[c]) - set(CATS) - {""}
    assert not bad, f"unknown categories in {c}: {bad}"

shared = df[df.coder == "shared"]
assert (shared[list(COLS.values())] != "").all().all(), "every shared row needs all three codes"
table, _ = aggregate_raters(shared[list(COLS.values())].map(CATS.index).to_numpy())
kappa = fleiss_kappa(table)

final, ties = [], []
for _, r in df.iterrows():
    if r.coder == "shared":
        top = Counter(r[list(COLS.values())]).most_common()
        (final.append(top[0][0]) if top[0][1] >= 2 else ties.append(r.id))
    else:
        assert r[COLS[r.coder]], f"row {r.id} not coded by {r.coder}"
        final.append(r[COLS[r.coder]])
counts = Counter(final)
out = {"n_coded": len(final), "shared_fleiss_kappa": kappa, "shared_ties": ties,
       "breakdown": {c: counts[c] for c in CATS}, "share": {c: counts[c] / len(final) for c in CATS}}
(ROOT / "results" / "error_analysis.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

fig, ax = plt.subplots(figsize=(6, 2.6))
ax.barh([c.replace("_", " ") for c in CATS][::-1], [counts[c] for c in CATS][::-1], color="#2a78d6", height=0.6)
ax.set(xlabel=f"Errors (of {len(final)} coded)", title=f"Hand-coded test errors (shared-set Fleiss κ = {kappa:.2f})")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(ROOT / "results" / "figures" / "fig_error_categories.png", dpi=200, bbox_inches="tight")
