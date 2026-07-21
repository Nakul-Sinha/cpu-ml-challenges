"""Evaluate the transducer on canonical folds: oracle-span replacement chrF.
Fits stores on train folds only. Reports per-language + per-mechanism coverage/quality.
Writes runs/A2/oof_reps.csv.
"""
import sys, json, collections
import pandas as pd

sys.path.insert(0, "solution")
sys.path.insert(0, "runs/A2")
from elru import replacement_chrf
from transducer import Transducer

USE_MODEL = "--model" in sys.argv
USE_DEL = "--nodel" not in sys.argv

train = pd.read_csv("dataset/train.csv")
folds = pd.read_csv("solution/folds.csv")
train = train.merge(folds, on="id")
train["edits"] = train.edits_json.apply(json.loads)

if USE_DEL:
    from ml import DeletionClf
if USE_MODEL:
    from seq2seq import Seq2SeqTransducer

oof = []
per_lang = collections.defaultdict(lambda: {"n": 0, "chrf": 0.0})
mech = collections.defaultdict(lambda: {"n": 0, "chrf": 0.0})
mech_lang = collections.defaultdict(lambda: {"n": 0, "chrf": 0.0})
del_stat = {"tp": 0, "fp": 0, "fn": 0, "n_true_del": 0}

for k in range(5):
    tr = train[train.fold != k]
    va = train[train.fold == k]
    T = Transducer().fit(tr)
    if USE_DEL:
        _thr = float([a.split("=")[1] for a in sys.argv if a.startswith("--thr=")][0]) if any(a.startswith("--thr=") for a in sys.argv) else 0.60
        T.fit_deletion(tr, DeletionClf(thr=_thr))
    if USE_MODEL:
        m = Seq2SeqTransducer().fit(tr)
        T.model = m
    for r in va.itertuples():
        for e in r.edits:
            src = r.text[e["start"]:e["end"]]
            true = e["replacement"]
            pred, mc = T.predict_dbg(r.language, src, r.text)
            c = replacement_chrf(pred, true)
            per_lang[r.language]["n"] += 1
            per_lang[r.language]["chrf"] += c
            mech[mc]["n"] += 1; mech[mc]["chrf"] += c
            mech_lang[(r.language, mc)]["n"] += 1; mech_lang[(r.language, mc)]["chrf"] += c
            # deletion detection
            if true == "":
                del_stat["n_true_del"] += 1
                if pred == "":
                    del_stat["tp"] += 1
                else:
                    del_stat["fn"] += 1
            elif pred == "":
                del_stat["fp"] += 1
            oof.append({"id": r.id, "start": e["start"], "end": e["end"],
                        "lang": r.language, "src": src, "pred_rep": pred,
                        "true_rep": true, "mechanism": mc, "chrf": round(c, 4)})

# overall metric = mean over langs of per-lang mean chrf  AND micro mean
lang_means = {L: d["chrf"] / d["n"] for L, d in per_lang.items()}
macro = sum(lang_means.values()) / len(lang_means)
micro = sum(d["chrf"] for d in per_lang.values()) / sum(d["n"] for d in per_lang.values())
print("=== ORACLE-SPAN REPLACEMENT chrF ===")
for L in sorted(lang_means):
    print(f"  {L}: chrf={lang_means[L]:.4f}  (n={per_lang[L]['n']})")
print(f"  MACRO (mean over langs) = {macro:.4f}")
print(f"  MICRO (all edits)       = {micro:.4f}")

print("\n=== per-mechanism coverage/quality (overall) ===")
tot = sum(d["n"] for d in mech.values())
for mc, d in sorted(mech.items(), key=lambda x: -x[1]["n"]):
    print(f"  {mc:10s} n={d['n']:5d} ({d['n']/tot:5.1%})  chrf={d['chrf']/d['n']:.4f}")

print("\n=== per-lang x mechanism ===")
for (L, mc), d in sorted(mech_lang.items()):
    print(f"  {L} {mc:10s} n={d['n']:5d}  chrf={d['chrf']/d['n']:.4f}")

print("\n=== deletion detection ===")
tp, fp, fn = del_stat["tp"], del_stat["fp"], del_stat["fn"]
prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
print(f"  true deletions={del_stat['n_true_del']}  tp={tp} fp={fp} fn={fn}  prec={prec:.3f} rec={rec:.3f}")

odf = pd.DataFrame(oof)
odf[["id","start","end","pred_rep","true_rep","mechanism"]].to_csv("runs/A2/oof_reps.csv", index=False)
print(f"\nwrote runs/A2/oof_reps.csv ({len(oof)} rows)")

# dump induced stores (fit on ALL train) for delivery/inspection
Tall = Transducer().fit(train)
import json as _j
store = {
  "n_exact": len(Tall.exact), "n_norm": len(Tall.norm),
  "n_mark_tpl": len(Tall.mark_tpl), "n_mark_tpl_bo": len(Tall.mark_tpl_bo),
  "n_suffix_rules": len(Tall.suffix_rules),
  "sample_mark_tpl": {f"{k[0]}|{k[1]}|{k[2]}": list(v) for k,v in list(Tall.mark_tpl.items())[:15]},
  "sample_suffix": {f"{k[0]}|{k[1]}": Tall.suffix_rules[k][0] for k in list(Tall.suffix_rules)[:15]},
}
with open("runs/A2/induced_stores.json","w") as f:
    _j.dump(store, f, ensure_ascii=False, indent=1)
print("wrote runs/A2/induced_stores.json", {k:v for k,v in store.items() if k.startswith('n_')})
