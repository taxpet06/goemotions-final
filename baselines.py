"""Prior, always-neutral and TF-IDF + logistic-regression systems (CPU).

Saves dev/test scores to outputs/preds/<name>__seed<s>__<split>.npy.
TF-IDF grid: C in {0.1, 1, 10} x n-grams {(1,1), (1,2)}; the config with the best
threshold-tuned dev macro-F1 is then reused for the learning curve and Ekman runs.
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from common import PREDS, NEUTRAL, load_split, macro_f1, predict, tune_thresholds, tokenize

PREDS.mkdir(parents=True, exist_ok=True)


def save(name, seed, dev, test):
    np.save(PREDS / f"{name}__seed{seed}__dev.npy", dev.astype(np.float32))
    np.save(PREDS / f"{name}__seed{seed}__test.npy", test.astype(np.float32))


def tfidf(train_texts, Ytr, texts, C, ngram):
    vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                          ngram_range=ngram, min_df=2, sublinear_tf=True)
    X = vec.fit_transform(train_texts)
    clf = OneVsRestClassifier(LogisticRegression(C=C, class_weight="balanced", solver="liblinear"))
    clf.fit(X, Ytr)
    return [clf.predict_proba(vec.transform(t)) for t in texts]


def main():
    tr, Ytr = load_split("train")
    dv, Ydv = load_split("dev")
    te, Yte = load_split("test")
    freq = Ytr.mean(0)

    # Prior: each label drawn independently at its training frequency (0/1 scores, so any threshold keeps them).
    for s in range(3):
        rng = np.random.default_rng(s)
        save("prior", s, rng.random(Ydv.shape) < freq, rng.random(Yte.shape) < freq)
    onehot = lambda n: np.eye(Ytr.shape[1])[[NEUTRAL] * n]
    save("always_neutral", 0, onehot(len(dv)), onehot(len(te)))

    log = {}
    for C in (0.1, 1, 10):
        for ng in ((1, 1), (1, 2)):
            name = f"tfidf=C{C}_ng{ng[1]}"
            Sd, St = tfidf(tr, Ytr, [dv, te], C, ng)
            save(name, 0, Sd, St)
            log[name] = macro_f1(Ydv, predict(Sd, tune_thresholds(Ydv, Sd)))
            print(name, "dev macro-F1", round(log[name], 4), flush=True)
    best = max(log, key=log.get)
    C, ng = float(best.split("C")[1].split("_")[0]), int(best[-1])
    print("best TF-IDF config:", best)

    # Learning curve: same config on nested random subsets of train.
    order = np.random.default_rng(0).permutation(len(tr))
    for frac in (0.1, 0.25, 0.5):
        idx = order[: int(frac * len(tr))]
        Sd, St = tfidf([tr[i] for i in idx], Ytr[idx], [dv, te], C, (1, ng))
        save(f"lc_tfidf={frac}", 0, Sd, St)

    # Ekman 6 + neutral.
    _, Etr = load_split("train", "ekman")
    Sd, St = tfidf(tr, Etr, [dv, te], C, (1, ng))
    save("ekman_tfidf", 0, Sd, St)
    (PREDS.parent / "tfidf_grid.json").write_text(json.dumps({"dev_macro_f1": log, "best": best}, indent=1))


if __name__ == "__main__":
    main()
