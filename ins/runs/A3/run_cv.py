"""A3 full-pipeline 5-fold CV + OOF + test submission.

Refits detector+transducer on the OTHER folds for each fold (no leakage),
collects OOF token probs, tunes per-language thresholds on pooled OOF,
reports CV ELRU + per-language detail, writes:
  runs/A3/oof_edits.csv        (id, edits_json) at tuned thresholds
  runs/A3/submission_v1.csv    (id, edits_json) full-train-fit, validated
  runs/A3/thresholds.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compose as C
sys.path.insert(0, os.path.join(C.ROOT, "solution"))
import elru

t0 = time.time()
train = pd.read_csv(os.path.join(C.ROOT, "dataset", "train.csv"))
test = pd.read_csv(os.path.join(C.ROOT, "dataset", "test.csv"))
folds = pd.read_csv(os.path.join(C.ROOT, "solution", "folds.csv"))
train = train.merge(folds, on="id")
train["ed"] = train.edits_json.apply(json.loads)

# ---- 5-fold OOF token probs --------------------------------------------------
tp_oof = {}
for k in range(5):
    tr = train[train.fold != k]
    va = train[train.fold == k]
    det = C.Detector(rounds=350).fit(tr)
    tp = det.token_probs(va)
    tp_oof.update(tp)
    print(f"[fold {k}] fit+predict done  ({time.time()-t0:.0f}s)  n_val={len(va)}")

# transducer for OOF assembly: fit per-fold to avoid leakage in replacements too
# (rebuild OOF edits fold-by-fold so replacements never see the val fold)
def oof_predict_with_thr(thr_map):
    pm = {}
    for k in range(5):
        tr = train[train.fold != k]
        va = train[train.fold == k]
        trd = C.Transducer().fit(tr)
        for r in va.itertuples():
            tk, probs = tp_oof[r.id]
            pm[r.id] = C.assemble(r.id, r.text, r.language, tk, probs, trd,
                                  thr_map.get(r.language, 0.5))
    return pm

# ---- threshold tuning on pooled OOF (per-language) ---------------------------
# use a fold-fit transducer for tuning-time assembly (leak-free): build a dict
# id->transducer-fold so tune_thresholds can transduce leak-free.
class OOFTransducer:
    """routes transduce() to the correct fold-fit transducer per id."""
    def __init__(self, train, tp_ids_fold):
        self.byfold = {k: C.Transducer().fit(train[train.fold != k]) for k in range(5)}
        self.idfold = tp_ids_fold
        self._cur = None
    def for_id(self, _id):
        self._cur = self.byfold[self.idfold[_id]]
        return self._cur

idfold = {r.id: r.fold for r in train.itertuples()}

# tune per-language threshold using leak-free per-fold transducers
def tune():
    grid = [round(x, 3) for x in np.arange(0.10, 0.965, 0.02)]
    byfold = {k: C.Transducer().fit(train[train.fold != k]) for k in range(5)}
    thr_map = {}
    for L in ["de", "en", "it"]:
        rows = [r for r in train.itertuples() if r.language == L]
        best_thr, best = grid[0], -1
        for thr in grid:
            pm = {}
            for r in rows:
                tk, probs = tp_oof[r.id]
                trd = byfold[idfold[r.id]]
                pm[r.id] = C.assemble(r.id, r.text, L, tk, probs, trd, thr)
            tmap = {r.id: r.ed for r in rows}
            lmap = {r.id: L for r in rows}
            sc = elru.elru(pm, tmap, lmap)
            if sc > best:
                best, best_thr = sc, thr
        thr_map[L] = best_thr
        print(f"  tuned thr[{L}] = {best_thr}  (isolated lang ELRU {best:.4f})")
    return thr_map, byfold

thr_map, byfold = tune()

# ---- final OOF ELRU at tuned thresholds --------------------------------------
pm = {}
for r in train.itertuples():
    tk, probs = tp_oof[r.id]
    trd = byfold[idfold[r.id]]
    pm[r.id] = C.assemble(r.id, r.text, r.language, tk, probs, trd, thr_map[r.language])
truth = {r.id: r.ed for r in train.itertuples()}
langs = {r.id: r.language for r in train.itertuples()}
score, detail = elru.elru(pm, truth, langs, detail=True)

# reference: fixed 0.5 threshold
pm05 = {}
for r in train.itertuples():
    tk, probs = tp_oof[r.id]
    pm05[r.id] = C.assemble(r.id, r.text, r.language, tk, probs, byfold[idfold[r.id]], 0.5)
score05 = elru.elru(pm05, truth, langs)

print("\n================ A3 FULL PIPELINE v1 ================")
print(f"CV ELRU (tuned thresholds) = {score:.4f}")
print(f"CV ELRU (fixed 0.5)        = {score05:.4f}")
print(f"thresholds = {thr_map}")
for L in ["de", "en", "it"]:
    d = detail[L]
    print(f"  {L}: lang={d['lang_score']:.4f}  edited={d['edited_mean']:.4f}"
          f"(n={d['n_edited']})  unchanged={d['unchanged_mean']:.4f}(n={d['n_unchanged']})")

# ---- save OOF edits ----------------------------------------------------------
oof_rows = [{"id": _id, "edits_json": json.dumps(pm[_id], ensure_ascii=False)}
            for _id in train.id]
pd.DataFrame(oof_rows).to_csv(os.path.join(HERE, "oof_edits.csv"), index=False)
json.dump(thr_map, open(os.path.join(HERE, "thresholds.json"), "w"))

# ---- full-train fit -> test submission --------------------------------------
train["ed"] = train.edits_json.apply(json.loads)
det_full = C.Detector(rounds=350).fit(train)
trd_full = C.Transducer().fit(train)
tp_test = det_full.token_probs(test)
sub = {}
for r in test.itertuples():
    tk, probs = tp_test[r.id]
    sub[r.id] = C.assemble(r.id, r.text, r.language, tk, probs, trd_full,
                           thr_map.get(r.language, 0.5))

# validate submission
assert len(sub) == len(test) == 445, (len(sub), len(test))
assert set(sub) == set(test.id), "id mismatch"
tl = {r.id: len(r.text) for r in test.itertuples()}
bad = [i for i in sub if not elru.validate_edits(sub[i], tl[i])]
assert not bad, f"invalid rows: {bad[:5]}"
sub_rows = [{"id": i, "edits_json": json.dumps(sub[i], ensure_ascii=False)} for i in test.id]
pd.DataFrame(sub_rows).to_csv(os.path.join(HERE, "submission_v1.csv"), index=False)
n_edit_rows = sum(1 for i in sub if sub[i])
print(f"\nsubmission_v1.csv: 445 rows valid; {n_edit_rows} rows with >=1 edit "
      f"({n_edit_rows/445:.1%})")
print(f"total time {time.time()-t0:.0f}s")
