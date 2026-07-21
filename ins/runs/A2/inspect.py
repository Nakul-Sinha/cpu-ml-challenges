import pandas as pd
oof = pd.read_csv("runs/A2/oof_reps.csv").fillna("")
print("=== worst MULTI cases (de) ===")
m = oof[(oof.mechanism=="multi")&(oof.lang=="de")].sort_values("chrf").head(12)
for r in m.itertuples():
    print(f"[{r.chrf}] {r.src!r}\n   pred={r.pred_rep!r}\n   true={r.true_rep!r}")
print("\n=== worst MULTI cases (it) ===")
m = oof[(oof.mechanism=="multi")&(oof.lang=="it")].sort_values("chrf").head(10)
for r in m.itertuples():
    print(f"[{r.chrf}] {r.src!r}\n   pred={r.pred_rep!r}\n   true={r.true_rep!r}")
print("\n=== SUFFIX cases where it underperforms (chrf<0.5) ===")
m = oof[(oof.mechanism=="suffix")&(oof.chrf<0.5)].head(12)
for r in m.itertuples():
    print(f"[{r.chrf}] {r.lang} {r.src!r} -> pred={r.pred_rep!r} true={r.true_rep!r}")
print("\n=== TRUE deletions (all) ===")
d = oof[oof.true_rep==""]
for r in d.head(20).itertuples():
    print(f"{r.lang} n={len(str(r.src).split())} {r.src!r} -> pred={r.pred_rep!r} [{r.mechanism}]")
print(f"...total {len(d)} deletions; by lang:\n{d.lang.value_counts()}")
print("\n=== en identity fails ===")
m = oof[(oof.mechanism=="identity")&(oof.lang=="en")].sort_values("chrf").head(12)
for r in m.itertuples():
    print(f"[{r.chrf}] {r.src!r} -> true={r.true_rep!r}")
