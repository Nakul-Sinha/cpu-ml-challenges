"""Canonical 5-fold grouped split for ins. Groups (documents) never straddle folds.
Deterministic; language-balanced round-robin by group size. Writes folds.csv (id,fold).
All agents must use THESE folds."""
import pandas as pd

TRAIN = r"dataset/train.csv"

def build_folds(train):
    sizes = train.groupby(["language", "document_group"]).size().reset_index(name="n")
    fold_of = {}
    for lang, d in sizes.groupby("language"):
        d = d.sort_values(["n", "document_group"], ascending=[False, True])
        loads = [0.0] * 5
        counts = [0] * 5
        for r in d.itertuples():
            # fewest groups first, then lightest load: keeps en's 6 groups spread out
            k = min(range(5), key=lambda i: (counts[i], loads[i]))
            fold_of[r.document_group] = k
            loads[k] += r.n
            counts[k] += 1
    return train.document_group.map(fold_of)

if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    train = pd.read_csv(f"{base}/dataset/train.csv")
    train["fold"] = build_folds(train)
    train[["id", "fold"]].to_csv(f"{base}/solution/folds.csv", index=False)
    print(train.groupby(["fold", "language"]).size().unstack(fill_value=0))
    import json
    train["n_edits"] = train.edits_json.apply(lambda s: len(json.loads(s)))
    print("\nedited-row fraction per fold:")
    print(train.groupby("fold").apply(lambda d: (d.n_edits > 0).mean(), include_groups=False))
