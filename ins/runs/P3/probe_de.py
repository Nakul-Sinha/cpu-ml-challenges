"""Probe de/en/it submission edited-ratios at several de thresholds (build once)."""
import os, sys, collections, json
sys.path.insert(0, os.path.expanduser("~/insled/runs/P3"))
import importlib.util
spec = importlib.util.spec_from_file_location("solution", os.path.expanduser("~/insled/runs/P3/solution.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

train, test = S.load_frames(os.path.expanduser("~/insled/dataset"))
print(f"train={len(train)} test={len(test)}", flush=True)
art = S.ship_artifacts(train, test)
lang = {r.id: r.language for r in test.itertuples()}
TR = {"de": 0.577, "en": 0.470, "it": 0.704}
for thr in [0.15, 0.19, 0.25, 0.31, 0.35, 0.40]:
    sub = S.assemble_submission(art, de_thr=thr)
    ed = collections.Counter(); tot = collections.Counter()
    for i in sub:
        tot[lang[i]] += 1
        if sub[i]:
            ed[lang[i]] += 1
    parts = []
    for L in ("de", "en", "it"):
        frac = ed[L] / max(tot[L], 1); ratio = frac / TR[L]
        flag = "" if 0.45 <= ratio <= 1.80 else " <<FLAG"
        parts.append(f"{L}={ed[L]}/{tot[L]} r={ratio:.2f}{flag}")
    print(f"de_thr={thr:.2f}  " + "  ".join(parts), flush=True)
