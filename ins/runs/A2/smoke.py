import sys, json, time
import pandas as pd
sys.path.insert(0, "solution"); sys.path.insert(0, "runs/A2")
from elru import replacement_chrf
from seq2seq import Seq2SeqTransducer

train = pd.read_csv("dataset/train.csv")
folds = pd.read_csv("solution/folds.csv")
train = train.merge(folds, on="id"); train["edits"] = train.edits_json.apply(json.loads)
tr = train[train.fold != 0]; va = train[train.fold == 0]
t0 = time.time()
m = Seq2SeqTransducer(epochs=28).fit(tr)
print(f"train time fold0: {time.time()-t0:.1f}s")
# eval on val edits
tot = 0; c = 0.0
ex = []
for r in va.itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]; true = e["replacement"]
        g = m.generate(r.language, src)
        cc = replacement_chrf(g, true); c += cc; tot += 1
        if len(ex) < 15:
            ex.append((r.language, src, g, true, round(cc, 2)))
print(f"GRU val chrf (fold0, all edits) = {c/tot:.4f}  n={tot}")
for lg, s, g, t, cc in ex:
    print(f"  [{cc}] {lg} {s!r} -> gen={g!r} true={t!r}")
