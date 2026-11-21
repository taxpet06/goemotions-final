"""All evaluation and investigations (CPU). Reads outputs/preds, writes results/.

1. Model selection on dev (mean dev macro-F1 over seeds, per-label thresholds tuned on dev).
2. Test metrics per system: macro/micro-F1, per-emotion P/R/F1, PR-AUC (AP), ROC-AUC and
   prevalence-normalised AP = (AP - p) / (1 - p)  [proposal feedback #2].
3. Paired bootstrap (1,000 resamples of test comments) for macro-F1 and model differences.
4. Rater agreement per emotion (Krippendorff's alpha) from the raw per-rater files.
5. Joint OLS: per-emotion F1 ~ log10(train count) + alpha, plus the predictor correlation
   [proposal feedback #1]. Same model for the RoBERTa - TF-IDF gap.
6. Rare-vs-frequent gap test, threshold sweep, learning curves, Ekman, label confusions,
   and the 100-error sample for hand coding.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (CLEAN, EKMAN, EMOTIONS, NEUTRAL, PREDS, ROOT, bootstrap_idx, f1_per_label,
                    kripp_alpha_binary, load_preds, load_split, macro_f1, micro_f1, predict, tune_thresholds)

OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
EMO27 = [e for e in EMOTIONS if e != "neutral"]
COLORS = {"tfidf": "#2a78d6", "bilstm": "#eb6834", "roberta": "#1baf7a"}  # one fixed colour per model across all figures
NICE = {"prior": "Prior", "always_neutral": "Always neutral", "tfidf": "TF-IDF + LR",
        "bilstm": "BiLSTM + GloVe", "roberta": "RoBERTa-base"}

_, Ydv = load_split("dev")
_, Yte = load_split("test")
prep = json.loads((CLEAN / "prep_report.json").read_text())
train_n = np.array([prep["train_counts"][e] for e in EMOTIONS])
total_n = np.array([prep["all_split_counts"][e] for e in EMOTIONS])
rare = (total_n < 1000) & (np.arange(len(EMOTIONS)) != NEUTRAL)  # the proposal's "8 emotions under 1,000"


# ------------------------------------------------------------------ 1. selection
def dev_score(name, Yd=Ydv):
    Sd = load_preds(name, "dev")
    return np.mean([macro_f1(Yd, predict(s, tune_thresholds(Yd, s))) for s in Sd])


def configs(family):
    return sorted({f.name.split("__")[0] for f in PREDS.glob(f"{family}=*__seed*__dev.npy")})


selection = {}
for fam in ("tfidf", "bilstm", "roberta"):
    scores = {c: dev_score(c) for c in configs(fam)}
    selection[fam] = {"dev_macro_f1": scores, "chosen": max(scores, key=scores.get)}
SYSTEMS = {"prior": "prior", "always_neutral": "always_neutral",
           **{f: selection[f]["chosen"] for f in ("tfidf", "bilstm", "roberta")}}


# ------------------------------------------------------------------ 2. test metrics
def evaluate(name, Yd=Ydv, Yt=Yte):
    """Per seed: thresholds from dev, binary test predictions, scores."""
    Sd, St = load_preds(name, "dev"), load_preds(name, "test")
    thr = np.stack([tune_thresholds(Yd, s) for s in Sd])
    return St, np.stack([predict(s, t) for s, t in zip(St, thr)]), thr


def ranking(Y, S):
    ap = np.array([average_precision_score(Y[:, k], S[:, k]) for k in range(Y.shape[1])])
    auc = np.array([roc_auc_score(Y[:, k], S[:, k]) for k in range(Y.shape[1])])
    p = Y.mean(0)
    return ap, auc, (ap - p) / (1 - p)


rows, per_emotion, P_test, S_test, THR = [], {}, {}, {}, {}
for sysname, name in SYSTEMS.items():
    St, Pt, thr = evaluate(name)
    P_test[sysname], S_test[sysname], THR[sysname] = Pt, St, thr
    ma = [macro_f1(Yte, p) for p in Pt]
    mi = [micro_f1(Yte, p) for p in Pt]
    rk = [ranking(Yte, s) for s in St]
    tp = np.stack([(Yte & p).sum(0) for p in Pt]).astype(float)
    prec = (tp / np.stack([p.sum(0) for p in Pt]).clip(min=1)).mean(0)
    rec = (tp / Yte.sum(0)).mean(0)
    f1 = np.stack([f1_per_label(Yte, p) for p in Pt])
    per_emotion[sysname] = pd.DataFrame({
        "emotion": EMOTIONS, "train_n": train_n, "test_n": Yte.sum(0), "precision": prec, "recall": rec,
        "f1": f1.mean(0), "f1_sd": f1.std(0), "ap": np.mean([r[0] for r in rk], 0),
        "roc_auc": np.mean([r[1] for r in rk], 0), "ap_norm": np.mean([r[2] for r in rk], 0)})
    rows.append({"system": NICE[sysname], "config": name, "seeds": len(Pt),
                 "macro_f1": np.mean(ma), "macro_f1_sd": np.std(ma), "micro_f1": np.mean(mi), "micro_f1_sd": np.std(mi),
                 "macro_ap": per_emotion[sysname].ap.mean(), "macro_roc_auc": per_emotion[sysname].roc_auc.mean(),
                 "macro_ap_norm": per_emotion[sysname].ap_norm.mean()})
main = pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. paired bootstrap
W = np.zeros((1000, len(Yte)), dtype=np.float32)
for b, idx in enumerate(bootstrap_idx(len(Yte))):
    W[b] = np.bincount(idx, minlength=len(Yte))


def boot_f1(P):
    """(B, k) per-label F1 on each resample, averaged over seeds."""
    out = 0
    for p in P:
        tp = W @ (Yte & p)
        fp = W @ ((1 - Yte) & p)
        fn = W @ (Yte & (1 - p))
        d = 2 * tp + fp + fn
        out = out + np.divide(2 * tp, d, out=np.zeros_like(tp), where=d > 0)
    return out / len(P)


BF = {s: boot_f1(P_test[s]) for s in SYSTEMS}
ci = lambda x: np.percentile(x, [2.5, 97.5])
main["macro_f1_ci_low"], main["macro_f1_ci_high"] = zip(*[ci(BF[s].mean(1)) for s in SYSTEMS])
diffs = []
for a, b in [("roberta", "tfidf"), ("roberta", "bilstm"), ("bilstm", "tfidf"), ("tfidf", "prior")]:
    d = BF[a].mean(1) - BF[b].mean(1)
    diffs.append({"comparison": f"{NICE[a]} - {NICE[b]}", "delta_macro_f1": main.set_index("system").macro_f1[NICE[a]] - main.set_index("system").macro_f1[NICE[b]],
                  "ci_low": ci(d)[0], "ci_high": ci(d)[1], "p_boot(delta<=0)": float((d <= 0).mean())})
diffs = pd.DataFrame(diffs)


# ------------------------------------------------------------------ 4. rater agreement
raw = pd.concat(pd.read_csv(ROOT / "data" / "raw" / f"goemotions_{i}.csv") for i in (1, 2, 3))
kept_ids = {json.loads(l)["id"] for s in ("train", "dev", "test") for l in (CLEAN / f"{s}.jsonl").open()}
raw = raw[raw.id.isin(kept_ids) & ~raw.example_very_unclear]  # very-unclear rows carry no labels
g = raw.groupby("id")
m = g.size().to_numpy()
pos = g[EMOTIONS].sum()
alpha = np.array([kripp_alpha_binary(m, pos[e].to_numpy()) for e in EMOTIONS])
agreement = pd.DataFrame({"emotion": EMOTIONS, "kripp_alpha": alpha, "train_n": train_n})
agreement_meta = {"comments": int(len(m)), "rater_rows": int(len(raw)), "raters_per_comment": dict(pd.Series(m).value_counts().sort_index().astype(int))}


# ------------------------------------------------------------------ 5. joint regression
emo_idx = np.array([EMOTIONS.index(e) for e in EMO27])
logn = np.log10(train_n[emo_idx])
al = alpha[emo_idx]
r_pred, p_pred = stats.pearsonr(logn, al)
rho_pred, _ = stats.spearmanr(logn, al)
z = lambda x: (x - x.mean()) / x.std(ddof=1)
Xz = sm.add_constant(np.column_stack([z(logn), z(al)]))


def ols(y, label):
    fit = sm.OLS(y, Xz).fit()
    c = fit.conf_int()
    out = {"outcome": label, "n": len(y), "r2": fit.rsquared, "adj_r2": fit.rsquared_adj}
    for j, k in ((1, "log_freq"), (2, "agreement")):
        out |= {f"b_{k}": fit.params[j], f"b_{k}_lo": c[j, 0], f"b_{k}_hi": c[j, 1], f"p_{k}": fit.pvalues[j]}
    # single-predictor fits, to show what each would claim on its own
    for k, x in (("log_freq", logn), ("agreement", al)):
        out[f"r_{k}_alone"] = stats.pearsonr(x, y)[0]
    return out


f1e = {s: per_emotion[s].set_index("emotion").f1[EMO27].to_numpy() for s in ("tfidf", "bilstm", "roberta")}
reg = pd.DataFrame([ols(f1e[s], f"F1 {NICE[s]}") for s in f1e] + [ols(f1e["roberta"] - f1e["tfidf"], "F1 gap RoBERTa - TF-IDF")])
predictor_corr = {"pearson_r": r_pred, "pearson_p": p_pred, "spearman_rho": rho_pred, "vif": 1 / (1 - r_pred ** 2), "n_emotions": len(EMO27)}

# Bootstrap the emotion set (27 points) for the RoBERTa coefficients - a check on 27-point OLS.
rng = np.random.default_rng(0)
Xfix = sm.add_constant(np.column_stack([z(logn), z(al)]))
bb = []
for _ in range(2000):
    i = rng.integers(0, len(EMO27), len(EMO27))
    bb.append(np.linalg.lstsq(Xfix[i], f1e["roberta"][i], rcond=None)[0][1:])
bb = np.array(bb)
reg_boot = {"roberta_b_log_freq_ci": list(ci(bb[:, 0])), "roberta_b_agreement_ci": list(ci(bb[:, 1]))}


# ------------------------------------------------------------------ 6a. rare vs frequent gap
gap = BF["roberta"] - BF["tfidf"]  # (B, k)
freq_mask = ~rare & (np.arange(len(EMOTIONS)) != NEUTRAL)
gap_obs = f1e["roberta"] - f1e["tfidf"]
rare27 = rare[emo_idx]
gap_stat = gap[:, rare].mean(1) - gap[:, freq_mask].mean(1)
gap_test = {"rare_emotions": [e for e, r in zip(EMOTIONS, rare) if r],
            "mean_gap_rare": gap_obs[rare27].mean(), "mean_gap_frequent": gap_obs[~rare27].mean(),
            "difference": gap_obs[rare27].mean() - gap_obs[~rare27].mean(),
            "ci": list(ci(gap_stat)), "p_boot(diff<=0)": float((gap_stat <= 0).mean()),
            "spearman_gap_vs_logfreq": stats.spearmanr(logn, gap_obs)[0]}

# ------------------------------------------------------------------ 6b. threshold sweep
sweep = []
for s in ("tfidf", "bilstm", "roberta"):
    Sd, St = load_preds(SYSTEMS[s], "dev"), S_test[s]
    g05 = [macro_f1(Yte, predict(t, 0.5)) for t in St]
    gbest = []
    for d, t in zip(Sd, St):
        tg = max(np.arange(0.05, 1, 0.05), key=lambda x: macro_f1(Ydv, predict(d, x)))
        gbest.append(macro_f1(Yte, predict(t, tg)))
    sweep.append({"system": NICE[s], "global_0.5": np.mean(g05), "global_tuned": np.mean(gbest), "per_label_tuned": main.set_index("system").macro_f1[NICE[s]],
                  "rare_f1_global_0.5": np.mean([f1_per_label(Yte, predict(t, 0.5))[rare].mean() for t in St]),
                  "rare_f1_per_label": per_emotion[s].f1[rare].mean()})
sweep = pd.DataFrame(sweep)

# ------------------------------------------------------------------ 6c. learning curves
lc = []
for s in ("tfidf", "bilstm", "roberta"):
    for frac in (0.1, 0.25, 0.5, 1.0):
        name = SYSTEMS[s] if frac == 1 else f"lc_{s}={frac:g}"
        if frac < 1 and not list(PREDS.glob(f"{name}__seed*")):
            continue
        _, Pt, _ = evaluate(name)
        f = np.stack([f1_per_label(Yte, p) for p in Pt]).mean(0)
        lc.append({"system": NICE[s], "train_frac": frac, "train_comments": int(frac * prep["clean"]["train"]),
                   "seeds": len(Pt), "macro_f1": f.mean(), "rare_macro_f1": f[rare].mean(), "frequent_macro_f1": f[freq_mask].mean()})
lc = pd.DataFrame(lc)

# ------------------------------------------------------------------ 6d. Ekman
_, Edv = load_split("dev", "ekman")
_, Ete = load_split("test", "ekman")
groups = json.loads((ROOT / "data" / "raw" / "ekman_mapping.json").read_text())
collapse = np.zeros((len(EMOTIONS), len(EKMAN)), dtype=np.int8)
for k, es in groups.items():
    for e in es:
        collapse[EMOTIONS.index(e), EKMAN.index(k)] = 1
collapse[NEUTRAL, EKMAN.index("neutral")] = 1
ekman = []
for s in ("tfidf", "roberta"):
    name = f"ekman_{s}"
    if not list(PREDS.glob(f"{name}__seed*")):
        continue
    _, Pt, _ = evaluate(name, Edv, Ete)
    coll = [((p @ collapse) > 0).astype(np.int8) for p in P_test[s]]
    ekman.append({"system": NICE[s], "fine_28way_macro_f1": main.set_index("system").macro_f1[NICE[s]],
                  "ekman_trained_macro_f1": np.mean([macro_f1(Ete, p) for p in Pt]),
                  "fine_collapsed_to_ekman_macro_f1": np.mean([macro_f1(Ete, p) for p in coll]),
                  "seeds": len(Pt)})
ekman = pd.DataFrame(ekman)

# ------------------------------------------------------------------ 6e. label confusions (best model)
best = max(("tfidf", "bilstm", "roberta"), key=lambda s: dev_score(SYSTEMS[s]))
Pb = P_test[best][0]
fn, fp = (Yte & (1 - Pb)), ((1 - Yte) & Pb)
conf = fn.T.astype(int) @ fp.astype(int)  # conf[i, j]: gold i missed while j wrongly predicted, same comment
pairs = sorted(((conf[i, j], EMOTIONS[i], EMOTIONS[j]) for i in range(len(EMOTIONS)) for j in range(len(EMOTIONS)) if i != j), reverse=True)[:15]
same_ekman = lambda a, b: (collapse[EMOTIONS.index(a)] & collapse[EMOTIONS.index(b)]).any()
confusions = pd.DataFrame([{"gold_missed": a, "predicted_instead": b, "count": int(c), "same_ekman_group": bool(same_ekman(a, b))} for c, a, b in pairs])
err_rows = (Yte != Pb).any(1)
n_err = int(err_rows.sum())
err_types = {"comments_with_any_error": n_err,
             "share_errors_with_fn_and_fp_in_same_ekman_group": float(np.mean([any(same_ekman(EMOTIONS[i], EMOTIONS[j]) for i in np.flatnonzero(fn[r]) for j in np.flatnonzero(fp[r])) for r in np.flatnonzero(err_rows)]))}

# ------------------------------------------------------------------ 6f. 100-error sample for hand coding
texts_te = [json.loads(l)["text"] for l in (CLEAN / "test.jsonl").open()]
ids_te = [json.loads(l)["id"] for l in (CLEAN / "test.jsonl").open()]
sample = np.random.default_rng(0).choice(np.flatnonzero(err_rows), 100, replace=False)
names = lambda row: ", ".join(EMOTIONS[i] for i in np.flatnonzero(row)) or "(none)"
coders = ["shared"] * 20 + ["coder1", "coder2", "coder3"] * 27  # 20 coded by all three, rest split
ea = pd.DataFrame({"id": [ids_te[i] for i in sample], "text": [texts_te[i] for i in sample],
                   "gold": [names(Yte[i]) for i in sample], "predicted": [names(Pb[i]) for i in sample],
                   "coder": coders[:100], "category_coder1": "", "category_coder2": "", "category_coder3": "", "notes": ""})
(ROOT / "error_analysis").mkdir(exist_ok=True)
coding_sheet = ROOT / "error_analysis" / "errors_to_code.csv"
if not coding_sheet.exists():  # never overwrite hand coding
    ea.to_csv(coding_sheet, index=False)

# ------------------------------------------------------------------ write tables
main.to_csv(OUT / "main_results.csv", index=False)
diffs.to_csv(OUT / "bootstrap_differences.csv", index=False)
pd.concat([d.assign(system=NICE[s]) for s, d in per_emotion.items()]).to_csv(OUT / "per_emotion.csv", index=False)
agreement.to_csv(OUT / "agreement.csv", index=False)
reg.to_csv(OUT / "joint_regression.csv", index=False)
sweep.to_csv(OUT / "threshold_sweep.csv", index=False)
lc.to_csv(OUT / "learning_curves.csv", index=False)
ekman.to_csv(OUT / "ekman.csv", index=False)
confusions.to_csv(OUT / "top_confusions.csv", index=False)
summary = {"selection": selection, "systems": SYSTEMS, "best_dev_system": best, "predictor_correlation": predictor_corr,
           "regression_bootstrap": reg_boot, "gap_test": gap_test, "agreement_meta": agreement_meta, "error_types": err_types,
           "data": {k: prep[k] for k in ("raw", "removed", "clean")}}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


# ------------------------------------------------------------------ figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "savefig.dpi": 200, "savefig.bbox": "tight"})
MODELS = ("tfidf", "bilstm", "roberta")


def label_points(ax, x, y, which):
    for e in which:
        i = EMO27.index(e)
        ax.annotate(e, (x[i], y[i]), xytext=(4, -3), textcoords="offset points", fontsize=7.5, color="#52514e")


# Fig 1: F1 vs frequency and vs agreement, plus the predictor confound.
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
for s in MODELS:
    for ax, x in ((axs[0], logn), (axs[1], al)):
        ax.scatter(x, f1e[s], s=22, color=COLORS[s], label=NICE[s], edgecolor="white", linewidth=0.8, zorder=3)
        b = np.polyfit(x, f1e[s], 1)
        xx = np.linspace(x.min(), x.max(), 10)
        ax.plot(xx, np.polyval(b, xx), color=COLORS[s], lw=1.6)
axs[0].set(xlabel="Training examples (log10)", ylabel="Test F1 (mean over seeds)", title="F1 vs. training frequency")
axs[1].set(xlabel="Rater agreement (Krippendorff's α)", title="F1 vs. rater agreement")
label_points(axs[0], logn, f1e["roberta"], ["grief", "relief", "pride", "nervousness", "gratitude", "realization"])
label_points(axs[1], al, f1e["roberta"], ["grief", "relief", "pride", "realization", "gratitude"])
axs[0].legend(frameon=False, fontsize=8.5, loc="upper left")
axs[2].scatter(logn, al, s=22, color="#52514e", zorder=3)
label_points(axs[2], logn, al, ["grief", "pride", "gratitude", "realization", "amusement", "love"])
axs[2].set(xlabel="Training examples (log10)", ylabel="Krippendorff's α", title=f"The two predictors (r = {r_pred:.2f})")
fig.tight_layout()
fig.savefig(FIG / "fig_frequency_agreement.png")
plt.close(fig)

# Fig 2: per-emotion F1, sorted by training frequency (dot plot).
order = np.argsort(-train_n)
fig, ax = plt.subplots(figsize=(11, 3.8))
x = np.arange(len(EMOTIONS))
for j, s in enumerate(MODELS):
    ax.scatter(x + (j - 1) * 0.22, per_emotion[s].f1.to_numpy()[order], s=20, color=COLORS[s], label=NICE[s], zorder=3, edgecolor="white", linewidth=0.6)
ax.set_xticks(x, [EMOTIONS[i] for i in order], rotation=60, ha="right", fontsize=8.5)
ax.axvspan(len(EMOTIONS) - rare.sum() - 0.5, len(EMOTIONS) - 0.5, color="#f1f0ec", zorder=0)
ax.text(len(EMOTIONS) - rare.sum() / 2 - 0.5, 0.98, "< 1,000 examples", ha="center", va="top", fontsize=8.5, color="#52514e", transform=ax.get_xaxis_transform())
ax.set(ylabel="Test F1", title="Per-emotion F1, emotions ordered by training frequency", ylim=(0, 1))
ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.38))
fig.savefig(FIG / "fig_per_emotion_f1.png")
plt.close(fig)

# Fig 3: RoBERTa - TF-IDF gap vs frequency.
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.axhline(0, color="#8a8984", lw=1)
ax.scatter(logn, gap_obs, s=24, color=np.where(rare27, "#1baf7a", "#8a8984"), zorder=3, edgecolor="white", linewidth=0.6)
label_points(ax, logn, gap_obs, [e for e, r in zip(EMO27, rare27) if r] + ["gratitude", "admiration"])
ax.set(xlabel="Training examples (log10)", ylabel="F1 gain, RoBERTa − TF-IDF", title="Where does pretraining help?")
fig.savefig(FIG / "fig_gap_vs_frequency.png")
plt.close(fig)

# Fig 4: learning curves.
if len(lc):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for s in MODELS:
        d = lc[lc.system == NICE[s]]
        for ax, col, t in ((axs[0], "frequent_macro_f1", f"{freq_mask.sum()} frequent emotions"), (axs[1], "rare_macro_f1", f"{rare.sum()} rare emotions (< 1,000)")):
            ax.plot(d.train_comments, d[col], marker="o", ms=5, lw=2, color=COLORS[s], label=NICE[s])
            ax.set(xscale="log", xlabel="Training comments", title=t)
            ax.set_xticks(d.train_comments, [f"{n / 1000:.1f}k" for n in d.train_comments])
            ax.minorticks_off()
    axs[0].set_ylabel("Test macro-F1")
    axs[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_learning_curves.png")
    plt.close(fig)

print(main.round(4).to_string())
print(diffs.round(4).to_string())
print(reg.round(3).to_string())
print(json.dumps({"predictor_correlation": predictor_corr, "gap_test": gap_test, "reg_boot": reg_boot, "err": err_types}, indent=1, default=float))
print(sweep.round(4).to_string()); print(lc.round(4).to_string()); print(ekman.round(4).to_string()); print(confusions.head(10).to_string())
