"""Clean the official GoEmotions split.

- Removes dev/test comments whose exact text also appears in train (leakage),
  and test comments whose text appears in dev (dev is used for tuning).
- Adds the official Ekman 6 + neutral mapping as a second label set.
Writes data/clean/{train,dev,test}.jsonl and data/clean/prep_report.json.
"""
import csv
import json

from common import CLEAN, EMOTIONS, EKMAN, ROOT

RAW = ROOT / "data" / "raw"
ekman_map = json.loads((RAW / "ekman_mapping.json").read_text())
to_ekman = {e: EKMAN.index(k) for k, es in ekman_map.items() for e in es}
to_ekman["neutral"] = EKMAN.index("neutral")


def read(split):
    with (RAW / f"{split}.tsv").open(newline="") as f:
        return [{"text": t, "labels": [int(x) for x in l.split(",")], "id": i}
                for t, l, i in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]


splits = {s: read(s) for s in ("train", "dev", "test")}
report = {"raw": {s: len(r) for s, r in splits.items()}}

train_texts = {r["text"] for r in splits["train"]}
dev_before = splits["dev"]
splits["dev"] = [r for r in dev_before if r["text"] not in train_texts]
dev_texts = {r["text"] for r in splits["dev"]}
test_before = splits["test"]
splits["test"] = [r for r in test_before if r["text"] not in train_texts]
n_test_train = len(test_before) - len(splits["test"])
splits["test"] = [r for r in splits["test"] if r["text"] not in dev_texts]
report["removed"] = {
    "dev_in_train": len(dev_before) - len(splits["dev"]),
    "test_in_train": n_test_train,
    "test_in_dev": len(test_before) - n_test_train - len(splits["test"]),
}
report["clean"] = {s: len(r) for s, r in splits.items()}
report["train_counts"] = {e: sum(i in r["labels"] for r in splits["train"]) for i, e in enumerate(EMOTIONS)}
report["all_split_counts"] = {e: sum(i in r["labels"] for s in splits.values() for r in s) for i, e in enumerate(EMOTIONS)}

CLEAN.mkdir(parents=True, exist_ok=True)
for s, rows in splits.items():
    with (CLEAN / f"{s}.jsonl").open("w") as f:
        for r in rows:
            r["ekman"] = sorted({to_ekman[EMOTIONS[i]] for i in r["labels"]})
            f.write(json.dumps(r) + "\n")
(CLEAN / "prep_report.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: report[k] for k in ("raw", "removed", "clean")}, indent=1))
