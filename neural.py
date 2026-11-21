"""BiLSTM + GloVe and RoBERTa-base multi-label classifiers (GPU).

One run = one (model, hyperparameters, seed). Each epoch is scored on dev with
per-label thresholds tuned on dev; the best dev epoch is kept (early stopping on
dev macro-F1) and its dev + test sigmoid scores are saved. Test scores never feed
back into any choice.

  python neural.py roberta --lr 2e-5 --seed 0
  python neural.py bilstm --lr 1e-3 --hidden 256 --seed 0 --glove glove.6B.300d.txt
  options: --labels ekman, --frac 0.25 (learning curve), --name override
"""
import argparse
import json
import random
import time
from collections import Counter

import numpy as np
import torch
from torch import nn

from common import PREDS, load_split, macro_f1, predict, tokenize, tune_thresholds

p = argparse.ArgumentParser()
p.add_argument("model", choices=["bilstm", "roberta"])
p.add_argument("--lr", type=float, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--epochs", type=int)
p.add_argument("--labels", default="goemotions")
p.add_argument("--frac", type=float, default=1.0)
p.add_argument("--glove", default="glove.6B.300d.txt")
p.add_argument("--name")
a = p.parse_args()
a.epochs = a.epochs or (5 if a.model == "roberta" else 10)
if a.name is None:
    a.name = f"{a.model}=lr{a.lr:g}" + (f"_h{a.hidden}" if a.model == "bilstm" else "")
    if a.labels == "ekman":
        a.name = "ekman_" + a.name.split("=")[0]
    if a.frac < 1:
        a.name = f"lc_{a.model}={a.frac:g}"

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = torch.device("cuda")

tr, Ytr = load_split("train", a.labels)
dv, Ydv = load_split("dev", a.labels)
te, Yte = load_split("test", a.labels)
if a.frac < 1:  # same nested subsets as the TF-IDF learning curve
    idx = np.random.default_rng(0).permutation(len(tr))[: int(a.frac * len(tr))]
    tr, Ytr = [tr[i] for i in idx], Ytr[idx]
K = Ytr.shape[1]

# Weighted BCE: pos_weight = sqrt(neg/pos). Full neg/pos reaches ~450 for grief and
# destabilises training; per-label thresholds fix calibration afterwards anyway.
pos = Ytr.sum(0).clip(min=1)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.sqrt((len(Ytr) - pos) / pos), dtype=torch.float32, device=dev))

# ------------------------------------------------------------------ encoders
if a.model == "roberta":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained("roberta-base")
    enc = lambda texts: tok(texts, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
    X = [enc(t) for t in (tr, dv, te)]
    X = [(x["input_ids"], x["attention_mask"]) for x in X]
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=K).to(dev)
    forward = lambda ids, mask: model(input_ids=ids, attention_mask=mask).logits
    bs = 32
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(tr) + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
else:
    counts = Counter(w for t in tr for w in tokenize(t))
    vocab = {w: i + 2 for i, w in enumerate(w for w, c in counts.items() if c >= 2)}  # 0 pad, 1 unk

    def enc(texts, L=64):
        ids = torch.zeros(len(texts), L, dtype=torch.long)
        for i, t in enumerate(texts):
            w = [vocab.get(x, 1) for x in tokenize(t)][:L] or [1]
            ids[i, : len(w)] = torch.tensor(w)
        return ids, (ids > 0).long()

    X = [enc(t) for t in (tr, dv, te)]
    emb = np.random.default_rng(a.seed).normal(0, 0.1, (len(vocab) + 2, 300)).astype(np.float32)
    emb[0] = 0
    hit = 0
    with open(a.glove, encoding="utf8") as f:
        for line in f:
            w, *v = line.rstrip().split(" ")
            if w in vocab:
                emb[vocab[w]] = np.asarray(v, dtype=np.float32); hit += 1
    print(f"vocab {len(vocab)}, GloVe coverage {hit / len(vocab):.3f}", flush=True)

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=False, padding_idx=0)
            self.lstm = nn.LSTM(300, a.hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.5)
            self.out = nn.Linear(2 * a.hidden, K)

        def forward(self, ids, mask):
            lens = mask.sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.emb(ids)), lens, batch_first=True, enforce_sorted=False)
            h, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=ids.shape[1])
            h = h.masked_fill(mask[..., None] == 0, -1e4).max(1).values  # max-pool over real tokens
            return self.out(self.drop(h))

    model = BiLSTM().to(dev)
    forward = model
    bs = 64
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = None

bf16 = torch.cuda.is_bf16_supported()
amp = torch.bfloat16 if bf16 else torch.float16
scaler = torch.amp.GradScaler(enabled=not bf16)


@torch.no_grad()
def scores(ids, mask):
    model.eval()
    out = []
    for i in range(0, len(ids), 256):
        with torch.autocast("cuda", dtype=amp):
            out.append(torch.sigmoid(forward(ids[i:i + 256].to(dev), mask[i:i + 256].to(dev)).float()).cpu())
    return torch.cat(out).numpy()


Yt = torch.tensor(Ytr, dtype=torch.float32)
best, log = -1, []
for ep in range(1, a.epochs + 1):
    model.train()
    t0 = time.time()
    perm = torch.randperm(len(tr))
    for i in range(0, len(tr), bs):
        b = perm[i:i + bs]
        with torch.autocast("cuda", dtype=amp):
            logits = forward(X[0][0][b].to(dev), X[0][1][b].to(dev))
        loss = loss_fn(logits.float(), Yt[b].to(dev))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if sched:
            sched.step()
    Sd = scores(*X[1])
    f = macro_f1(Ydv, predict(Sd, tune_thresholds(Ydv, Sd)))
    log.append({"epoch": ep, "dev_macro_f1": float(f), "loss": float(loss), "sec": round(time.time() - t0)})
    print(a.name, a.seed, log[-1], flush=True)
    if f > best:
        best = f
        St = scores(*X[2])  # test scores of the dev-selected epoch; never used for selection
        PREDS.mkdir(parents=True, exist_ok=True)
        np.save(PREDS / f"{a.name}__seed{a.seed}__dev.npy", Sd.astype(np.float32))
        np.save(PREDS / f"{a.name}__seed{a.seed}__test.npy", St.astype(np.float32))

(PREDS.parent / "logs").mkdir(exist_ok=True)
(PREDS.parent / "logs" / f"{a.name}__seed{a.seed}.json").write_text(
    json.dumps({"args": vars(a), "best_dev_macro_f1": best, "epochs": log, "gpu": torch.cuda.get_device_name()}, indent=1))
