"""Measure the ELRU cost of forfeiting deletions: take the best (feats_repl) OOF
assignment, inject ORACLE empty-replacement edits at every true deletion span
(leak-free -- oracle only for cost estimation), rescore, report delta."""
import os, sys, json, collections
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P
import m3_ext as M
import elru

train = pd.read_csv(os.path.join(P.ROOT, "dataset", "train.csv"))
folds = pd.read_csv(os.path.join(P.ROOT, "solution", "folds.csv"))
train = train.merge(folds, on="id")
train["edits"] = train.edits_json.apply(json.loads)

M.USE_FEATS = M.USE_IT_REPL = M.USE_EN_REPL = True
M.USE_DEL = M.USE_NPGEN = False
M.register(P)
res = P.run_cv(train, verbose=False)

rows = res["rows"]; ec = res["edits_cache"]; assign = res["nn_assign"]
base_pm = {R["id"]: ec[R["id"]][assign[R["id"]]] for R in rows}
tm = {R["id"]: R["truth"] for R in rows}
lm = {R["id"]: R["lang"] for R in rows}
base, _ = elru.elru(base_pm, tm, lm, detail=True)

# inject oracle deletions: add empty-rep edit at each true deletion span, keeping validity
def inject(pm):
    out = {}
    for R in rows:
        edits = [dict(e) for e in pm[R["id"]]]
        for a, b, rep in R["spans"]:
            if rep == "":
                # drop any existing edit overlapping, then add empty edit
                edits = [e for e in edits if e["end"] <= a or b <= e["start"]]
                edits.append({"start": a, "end": b, "replacement": ""})
        edits.sort(key=lambda e: e["start"])
        edits = P._repair(edits, len(R["text"]))
        out[R["id"]] = edits
    return out

orc = inject(base_pm)
sc, det = elru.elru(orc, tm, lm, detail=True)
print(f"base (forfeit deletions) ELRU = {base:.4f}")
print(f"oracle-deletions        ELRU = {sc:.4f}   delta = +{sc-base:.4f}")
for L in P.LANGS:
    print(f"   {L}: {det[L]['lang_score']:.4f}")
n_del = sum(1 for R in rows for a, b, rep in R["spans"] if rep == "")
print(f"true deletions = {n_del}")
