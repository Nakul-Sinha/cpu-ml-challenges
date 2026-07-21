"""N1 DIAGNOSIS 5: characterize it REPLACEMENT errors on matched spans, and locate the
oracle-replacement gain by span_type + by transduction mechanism.  Guides the transducer
improvement (the safe +.06 edited lever)."""
import os, sys, json, collections, re
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "solution"))
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
import elru
import pipeline
import m4_ext
from transducer import Transducer, _norm

WS = re.compile(r"\S+")
_STRIP = ".,;:()»«\"'“”’`-–—"
MARKS = set(":*∗/")


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
    txt = {r.id: r.text for r in it.itertuples()}
    base = {i: oofmap.get(i, []) for i in truth}
    lm = {i: "it" for i in truth}

    # match predicted to truth by best overlap; collect (type, pred_repl, true_repl, chrf)
    rows = []
    for i in truth:
        if not truth[i]:
            continue
        for e in base[i]:
            best = None; bj = 0.0
            for t in truth[i]:
                ov = max(0, min(e["end"], t["end"]) - max(e["start"], t["start"]))
                if ov > bj:
                    bj = ov; best = t
            if best is None:
                continue
            src = txt[i][best["start"]:best["end"]]
            st = span_type(src)
            chrf = elru.replacement_chrf(e["replacement"], best["replacement"])
            # boundary match quality
            sf = elru.span_f1(e["start"], e["end"], best["start"], best["end"])
            rows.append(dict(id=i, st=st, src=src, pred=e["replacement"], true=best["replacement"],
                             chrf=chrf, sf=sf, psrc=txt[i][e["start"]:e["end"]]))
    df = pd.DataFrame(rows)
    print(f"matched pred/truth pairs on edited rows: {len(df)}")
    print("\n=== mean chrf and span_f1 by type (matched pairs) ===")
    for st, g in df.groupby("st"):
        print(f"  {st:14s} n={len(g):3d}  chrf={g.chrf.mean():.3f}  span_f1={g.sf.mean():.3f} "
              f"chrf<0.9={ (g.chrf<0.9).sum() }")

    # oracle-repl gain localized by type: replace only type-T repls with truth, rescore
    def score_with_fix(fix_types):
        pm = {}
        for i in truth:
            if not truth[i]:
                pm[i] = base[i]; continue
            new = []
            for e in base[i]:
                best = None; bj = 0.0
                for t in truth[i]:
                    ov = max(0, min(e["end"], t["end"]) - max(e["start"], t["start"]))
                    if ov > bj:
                        bj = ov; best = t
                if best is not None and span_type(txt[i][best["start"]:best["end"]]) in fix_types:
                    new.append({"start": e["start"], "end": e["end"], "replacement": best["replacement"]})
                else:
                    new.append(e)
            pm[i] = new
        _s, d = elru.elru(pm, truth, lm, detail=True)
        return d["it"]["lang_score"], d["it"]["edited_mean"]
    base_lang, base_ed = score_with_fix(set())
    print(f"\nbase it lang={base_lang:.4f} edited={base_ed:.4f}")
    for ft in [{"single_plain"}, {"multi_plain"}, {"single_marked"}, {"multi_marked"},
               {"single_plain", "multi_plain"}, {"single_plain", "multi_plain", "single_marked", "multi_marked"}]:
        l, e = score_with_fix(ft)
        print(f"  fix repl {str(sorted(ft)):55s} it lang={l:.4f} (+{l-base_lang:.4f}) edited={e:.4f}")

    # show worst single_plain and multi_plain replacement errors
    for st in ["single_plain", "multi_plain"]:
        sub = df[(df.st == st) & (df.chrf < 0.9) & (df.sf > 0.5)].sort_values("chrf").head(18)
        print(f"\n=== {st} replacement errors (span matched sf>0.5, chrf<0.9), worst {len(sub)} ===")
        for r in sub.itertuples():
            print(f"  src={r.src!r}")
            print(f"     pred={r.pred!r}")
            print(f"     true={r.true!r}   chrf={r.chrf:.2f}")

    # slash-order convention: for single_plain slash edits, does truth put src-form first or second?
    print("\n=== slash-order convention (single_plain, truth has one '/') ===")
    order = collections.Counter()
    for r in it.itertuples():
        for e in r.edits:
            src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
            if len(src.split()) != 1 or rep.count("/") != 1 or " " in rep:
                continue
            core = src.strip(_STRIP)
            a, b = rep.split("/")
            if core == a:
                order["src_FIRST (src/other)"] += 1
            elif core == b:
                order["src_SECOND (other/src)"] += 1
            else:
                order["src_neither"] += 1
    print("  ", dict(order))


if __name__ == "__main__":
    main()
