"""Run M2 German plug-ins on the base pipeline; report de + overall (non-nested & nested),
per-type recall deltas vs loss map. Leak-free per fold (pipeline handles refits)."""
import os, sys, json, re, collections, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pipeline as P
import m2_ext
import elru

train = pd.read_csv(os.path.join(P.ROOT, "dataset", "train.csv"))
folds = pd.read_csv(os.path.join(P.ROOT, "solution", "folds.csv"))
train = train.merge(folds, on="id"); train["edits"] = train.edits_json.apply(json.loads)
MARKS = set(":*∗/")
def marked(s): return any(c in MARKS for c in s)

def reset():
    P.STORE_BUILDERS = []; P.SPAN_CANDIDATE_GENERATORS = []
    P.REPLACEMENT_HOOKS = []; P.TOKEN_FEATURE_EXTRAS = []
    P.FEAT_NAMES = None; P.EXTRA_NAMES = None; P._TR_MEMO = {}

def run(tag):
    t0 = time.time()
    res = P.run_cv(train, verbose=False)
    d = res["nonnested_detail"]; nn = res["nonnested_elru"]; nz = res["nested_elru"]
    print(f"\n===== {tag} =====  ({time.time()-t0:.0f}s)")
    print(f"  NONNESTED overall={nn:.4f}   NESTED overall={nz:.4f}")
    for L in ["de", "en", "it"]:
        x = d[L]
        print(f"  {L}: lang={x['lang_score']:.4f} edited={x['edited_mean']:.4f}(n={x['n_edited']}) "
              f"unchanged={x['unchanged_mean']:.4f}(n={x['n_unchanged']})  thr={res['nonnested_thr'][L]}")
    return res

# ---- per-type recall of de edited spans at an operating point ----
def de_type_recall(res):
    rows = res["rows"]; ec = res["edits_cache"]; assign = res["nn_assign"]
    by = collections.Counter(); hit = collections.Counter()
    for R in rows:
        if R["lang"] != "de" or not R["spans"]:
            continue
        pred = ec[R["id"]][assign[R["id"]]]
        for (a, b, rep) in R["spans"]:
            src = R["text"][a:b]; ntok = len(src.split())
            if rep == "": typ = "deletion"
            elif ntok == 1 and marked(src): typ = "single_marked"
            elif ntok == 1: typ = "single_plain"
            elif marked(src): typ = "multi_marked"
            else: typ = "multi_plain"
            by[typ] += 1
            # matched if any pred span overlaps >=0.5 IoU
            best = 0.0
            for e in pred:
                ov = max(0, min(b, e["end"]) - max(a, e["start"]))
                best = max(best, ov / max(1, (max(b, e["end"]) - min(a, e["start"]))))
            if best >= 0.5: hit[typ] += 1
    print("  de per-type IoU>=.5 recall:")
    for t in ["multi_plain", "multi_marked", "single_plain", "single_marked", "deletion"]:
        if by[t]: print(f"    {t:14s} {hit[t]/by[t]:.3f} ({hit[t]}/{by[t]})")

CONFIGS = sys.argv[1:] or ["base", "genonly", "featonly", "full"]
results = {}
for cfg in CONFIGS:
    reset()
    if cfg == "base":
        pass
    elif cfg == "genonly":
        os.environ["M2_GEN"] = "1"; os.environ["M2_FEAT"] = "0"; m2_ext.register(P)
    elif cfg == "featonly":
        os.environ["M2_GEN"] = "0"; os.environ["M2_FEAT"] = "1"; m2_ext.register(P)
    elif cfg == "full":
        os.environ["M2_GEN"] = "1"; os.environ["M2_FEAT"] = "1"; m2_ext.register(P)
    elif cfg == "full_rr":
        os.environ["M2_GEN"] = "1"; os.environ["M2_FEAT"] = "1"; os.environ["M2_RERANK"] = "1"; os.environ["M2_ADMIT"] = "0.15"; m2_ext.register(P)
    results[cfg] = run(cfg)
    if cfg != "base":
        de_type_recall(results[cfg])

if "base" in results:
    print("\n=== base de per-type recall (reference) ===")
    de_type_recall(results["base"])
    b = results["base"]["nonnested_elru"]
    for cfg in CONFIGS:
        if cfg != "base":
            print(f"delta overall {cfg}: {results[cfg]['nonnested_elru']-b:+.4f}  (de lang {results['base']['nonnested_detail']['de']['lang_score']:.4f} -> {results[cfg]['nonnested_detail']['de']['lang_score']:.4f})")
