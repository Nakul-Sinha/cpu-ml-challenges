"""Cache the expensive leak-free state (5x detector fit -> OOF row_proba) to disk so
iteration experiments skip the 22s detector refit. Transducers/stores rebuild fast."""
import os, sys, json, time, pickle
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import pandas as pd
import pipeline, m4_ext
from run_m4 import fit_folds

t0 = time.time()
m4_ext.register(pipeline)
train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
train = train.merge(folds, on="id")
train["edits"] = train.edits_json.apply(json.loads)
rows, idfold, row_proba, trs, stf = fit_folds(train, verbose=False)
# strip non-picklable: keep rows (dicts), idfold, row_proba, group_by_id
gbi = {r.id: r.document_group for r in train.itertuples()}
slim_rows = [{k: R[k] for k in ("id", "lang", "text", "tk", "y", "spans", "truth", "fold")} for R in rows]
pickle.dump(dict(rows=slim_rows, idfold=idfold, row_proba=row_proba, group_by_id=gbi),
            open(os.path.join(ROOT, "runs", "N1_state.pkl"), "wb"))
print(f"cached {len(slim_rows)} rows, {time.time()-t0:.0f}s -> N1_state.pkl")
