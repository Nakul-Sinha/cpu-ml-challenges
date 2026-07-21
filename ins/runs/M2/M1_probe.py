"""Probe: (1) does a lower threshold floor help de? (2) do all 4 extension points work?"""
import os, sys, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P

train = pd.read_csv(os.path.join(P.ROOT, "dataset", "train.csv"))
folds = pd.read_csv(os.path.join(P.ROOT, "solution", "folds.csv"))
train = train.merge(folds, on="id")
train["edits"] = train.edits_json.apply(json.loads)

# ---- (1) grid floor sweep ----
for floor in (0.05, 0.03, 0.02, 0.01):
    P.GRID = [round(x, 3) for x in np.arange(floor, 0.93, 0.02)]
    r = P.run_cv(train, verbose=False)
    print(f"floor={floor:.2f}  nonnested={r['nonnested_elru']:.4f}  nested={r['nested_elru']:.4f}  "
          f"thr={ {k: float(v) for k,v in r['nonnested_thr'].items()} }")

# reset grid
P.GRID = [round(x, 3) for x in np.arange(0.05, 0.93, 0.02)]

# ---- (2) extension-point smoke test (component-level, fast) ----
print("\n=== extension-point smoke test ===")
rows = P.build_rows(train[train.fold != 0], labeled=True)
val = P.build_rows(train[train.fold == 0], labeled=True)

# reset frozen schema so extras re-freeze with the dummy column
P.FEAT_NAMES = None; P.EXTRA_NAMES = None
calls = {"extra": 0, "store": 0, "hook": 0, "gen": 0, "scorer": 0}

def extra(tokens, i, lang, text):
    calls["extra"] += 1
    return {"tok_is_upper_first": 1.0 if tokens[i][2][:1].isupper() else 0.0}

def store_builder(train_df, stores):
    calls["store"] += 1
    stores["seen_langs"] = sorted(set(train_df.language))

def repl_hook(lang, src, context, stores):
    calls["hook"] += 1
    return None  # inert: defer to A2

def span_gen(tokens, lang, text, aux):
    calls["gen"] += 1
    return [(0, 0, {"why": "first-token-candidate"})] if tokens else []

def span_scorer(cands, tokens, lang, text, aux):
    calls["scorer"] += 1
    return [(a, b, 0.99) for (a, b, meta) in cands]  # approve all

P.TOKEN_FEATURE_EXTRAS = [extra]
P.STORE_BUILDERS = [store_builder]
P.REPLACEMENT_HOOKS = [repl_hook]
P.SPAN_CANDIDATE_GENERATORS = [span_gen]

stores = {}
for b in P.STORE_BUILDERS:
    b(train[train.fold != 0], stores)
stores["span_scorer"] = span_scorer

det = P.Detector().fit(rows, stores)
base_nfeat = None
X, cat = P.featurize(val, det.lex)
print("feature matrix cols (with 1 extra):", X.shape[1], "-> extra col name present:",
      "x_tok_is_upper_first" in P.FEAT_NAMES, "| cat_idx:", cat[:2], "...")
tp = det.token_probs(val)
# assemble a handful with hook + generator+scorer active
made = 0
for R in val[:30]:
    tk, pr = tp[R["id"]]
    ed = P.build_edits(R["id"], R["text"], R["lang"], tk, pr, 0.2, P.Transducer().fit(train[train.fold != 0]), stores)
    made += len(ed)
print("stores populated by builder:", stores.get("seen_langs"))
print("calls:", calls, "| edits produced across 30 rows:", made)
print("ALL EXTENSION POINTS FIRED:", all(calls[k] > 0 for k in ("extra", "store", "hook", "gen", "scorer")))

# restore empties (so importing pipeline stays pristine)
P.TOKEN_FEATURE_EXTRAS = []; P.STORE_BUILDERS = []; P.REPLACEMENT_HOOKS = []; P.SPAN_CANDIDATE_GENERATORS = []
