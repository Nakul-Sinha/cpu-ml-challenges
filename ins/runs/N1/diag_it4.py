"""N1 DIAGNOSIS 4: decompose the it edited-row score (detection vs transduction) and
nail the FP-generating mechanism.  Determines whether the achievable lever is boundary
detection, replacement quality, or a specific FP-prone pattern.
"""
import os, sys, json, collections, re
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "solution"))
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
import elru
import pipeline
import m4_ext
from transducer import Transducer

WS = re.compile(r"\S+")
_STRIP = ".,;:()»«\"'“”’`-–—"
MARKS = set(":*∗/")


def toks(t):
    return [(m.start(), m.end(), m.group()) for m in WS.finditer(t)]


def span_type(src):
    nt = len(src.split()); marked = any(c in MARKS for c in src)
    return ("single" if nt == 1 else "multi") + ("_marked" if marked else "_plain")


def main():
    m4_ext.register(pipeline)
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    it = train[train.language == "it"].reset_index(drop=True)

    oof = pd.read_csv(os.path.join(ROOT, "runs", "M4", "oof_edits.csv"))
    oofmap = {r.id: json.loads(r.edits_json) for r in oof.itertuples()}
    truth = {r.id: [{"start": e["start"], "end": e["end"], "replacement": e["replacement"]} for e in r.edits]
             for r in it.itertuples()}
    lm = {r.id: "it" for r in it.itertuples()}
    txt = {r.id: r.text for r in it.itertuples()}
    base = {i: oofmap.get(i, []) for i in truth}

    # ---- A) edited-row decomposition ----
    # For edited rows, compute row_score under:
    #   cur:            current preds
    #   oracle_bound:   current preds but boundaries snapped to best-matching truth span
    #                   (keep current replacement) -> isolates transduction loss
    #   oracle_repl:    current boundaries but replacement = truth of best-overlap span
    #                   -> isolates detection/boundary loss
    ed_ids = [i for i in truth if len(truth[i]) > 0]

    def rowscore(pred, tru):
        return elru.row_score(pred, tru)

    cur_scores = []; ob_scores = []; orr_scores = []
    for i in ed_ids:
        tru = truth[i]; pred = base[i]
        cur_scores.append(rowscore(pred, tru))
        # oracle boundary: for each pred, find best-overlap truth, snap start/end to it
        ob = []
        for e in pred:
            best = None; bj = 0.0
            for t in tru:
                ov = max(0, min(e["end"], t["end"]) - max(e["start"], t["start"]))
                if ov > bj:
                    bj = ov; best = t
            if best is not None:
                ob.append({"start": best["start"], "end": best["end"], "replacement": e["replacement"]})
            else:
                ob.append(e)
        # dedup overlaps by start
        ob.sort(key=lambda e: e["start"])
        ob = elru._repair(ob, len(txt[i])) if not elru.validate_edits(ob, len(txt[i])) else ob
        ob_scores.append(rowscore(ob, tru))
        # oracle replacement: current boundaries, replacement from best-overlap truth
        orr = []
        for e in pred:
            best = None; bj = 0.0
            for t in tru:
                ov = max(0, min(e["end"], t["end"]) - max(e["start"], t["start"]))
                if ov > bj:
                    bj = ov; best = t
            orr.append({"start": e["start"], "end": e["end"],
                        "replacement": best["replacement"] if best else e["replacement"]})
        orr_scores.append(rowscore(orr, tru))
    print(f"=== it EDITED-row score decomposition (n={len(ed_ids)}) ===")
    print(f"  current edited_mean:                 {np.mean(cur_scores):.4f}")
    print(f"  oracle BOUNDARIES (keep cur repl):   {np.mean(ob_scores):.4f}  (+{np.mean(ob_scores)-np.mean(cur_scores):.4f})")
    print(f"  oracle REPLACEMENT (keep cur bound): {np.mean(orr_scores):.4f}  (+{np.mean(orr_scores)-np.mean(cur_scores):.4f})")
    print("  (bigger gain => that axis is the bottleneck)")

    # ---- B) budget analysis: n_pred vs n_true on edited rows ----
    npred = collections.Counter(); ntrue = collections.Counter()
    over = under = exact = 0
    for i in ed_ids:
        p = len(base[i]); t = len(truth[i])
        if p > t: over += 1
        elif p < t: under += 1
        else: exact += 1
    print(f"\n  budget on edited rows: over-predict={over} under-predict={under} exact={exact}")

    # ---- C) FP mechanism on unchanged rows ----
    print("\n=== it UNCHANGED-row FP mechanism (n_unchanged=%d) ===" % sum(1 for i in truth if not truth[i]))
    fp_ids = [i for i in truth if not truth[i] and base[i]]
    print(f"  FP rows = {len(fp_ids)}")
    # classify each FP edit by span_type and by whether replacement changed the source
    typ = collections.Counter(); nchg = collections.Counter(); nedits = collections.Counter()
    ending = collections.Counter()
    for i in fp_ids:
        nedits[len(base[i])] += 1
        for e in base[i]:
            src = txt[i][e["start"]:e["end"]]
            st = span_type(src)
            typ[st] += 1
            changed = e["replacement"] != src
            nchg[("changed" if changed else "identity")] += 1
            core = src.strip(_STRIP).lower()
            if len(core) >= 2:
                ending[core[-2:]] += 1
    print(f"  FP edits by span_type: {dict(typ)}")
    print(f"  FP edits changed-vs-identity: {dict(nchg)}")
    print(f"  FP edits per row: {dict(sorted(nedits.items()))}")
    print(f"  FP source endings (top): {ending.most_common(12)}")

    # ---- D) do FP edits echo a slash-form (bad slash-append) or a real transduction? ----
    slash_fp = sum(1 for i in fp_ids for e in base[i] if "/" in e["replacement"] and "/" not in txt[i][e["start"]:e["end"]])
    print(f"  FP edits that ADDED a slash (slash-append on unchanged): {slash_fp}")

    # ---- E) compare: are FP source cores also frequent in TRUE edited spans? ----
    true_edit_cores = collections.Counter()
    for r in it.itertuples():
        for e in r.edits:
            if e["replacement"]:
                src = r.text[e["start"]:e["end"]]
                if len(src.split()) == 1:
                    true_edit_cores[src.strip(_STRIP).lower()] += 1
    fp_cores = collections.Counter()
    for i in fp_ids:
        for e in base[i]:
            src = txt[i][e["start"]:e["end"]]
            if len(src.split()) == 1:
                fp_cores[src.strip(_STRIP).lower()] += 1
    overlap = sum(1 for c in fp_cores if c in true_edit_cores)
    print(f"\n  distinct FP single-token cores={len(fp_cores)}; also-a-true-edit-core={overlap} "
          f"({overlap/max(1,len(fp_cores)):.0%})  -> if high, cores are ambiguous (context needed)")
    # show a few FP cores with their true-edit frequency and how often they appear edited vs total
    print("  sample FP cores (core, fp_count, true_edit_count):")
    for c, n in fp_cores.most_common(15):
        print(f"      {c!r:20s} fp={n} true_edit={true_edit_cores.get(c,0)}")


if __name__ == "__main__":
    main()
