"""N1 DIAGNOSIS 2: NP-anchor viability, unchanged-FP structure, oracle ceilings for it.

Answers: (a) can an article/prep-anchored NP generator catch the zero-coverage
multi_plain misses?  (b) where do the 56 it unchanged FPs live (group structure)?
(c) oracle ceilings: perfect multi_plain spans, perfect FP removal.
"""
import os, sys, json, collections, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "solution"))
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
import elru

MARKS = set(":*∗/")
_STRIP = ".,;:()»«\"'“”’`-–—"
WS = re.compile(r"\S+")


def toks(t):
    return [(m.start(), m.end(), m.group()) for m in WS.finditer(t)]


def strip(w):
    return w.strip(_STRIP)


def span_type(src):
    nt = len(src.split())
    marked = any(c in MARKS for c in src)
    return ("single" if nt == 1 else "multi") + ("_marked" if marked else "_plain")


def learn_it(df):
    """replicate m3_ext._learn_it signals we care about (full-train, for viability est.)"""
    occ = collections.Counter(); spaninit = collections.Counter(); spaninit_slash = collections.Counter()
    end2_ed = collections.Counter(); end2_tot = collections.Counter()
    end3_ed = collections.Counter(); end3_tot = collections.Counter()
    for r in df[df.language == "it"].itertuples():
        tk = toks(r.text)
        spans = sorted((e["start"], e["end"], e["replacement"]) for e in r.edits)
        startset = {a for a, _, _ in spans}
        rep_first_slash = {}
        for a, b, rep in spans:
            fw = rep.split()[0] if rep.split() else ""
            rep_first_slash[a] = ("/" in fw)

        def inside(s, e):
            return any(s >= a and e <= b for a, b, _ in spans)
        for i, (s, e, w) in enumerate(tk):
            core = strip(w).lower()
            if not core:
                continue
            occ[core] += 1
            isin = inside(s, e)
            if s in startset:
                spaninit[core] += 1
                if rep_first_slash.get(s):
                    spaninit_slash[core] += 1
            if len(core) >= 2:
                end2_tot[core[-2:]] += 1
                if isin:
                    end2_ed[core[-2:]] += 1
            if len(core) >= 3:
                end3_tot[core[-3:]] += 1
                if isin:
                    end3_ed[core[-3:]] += 1
    spaninit_rate = {w: spaninit[w] / occ[w] for w in occ}
    article_set = {w for w in occ if occ[w] >= 3 and spaninit_slash[w] / occ[w] >= 0.40}
    end2_rate = {k: end2_ed[k] / end2_tot[k] for k in end2_tot if end2_tot[k] >= 8}
    end3_rate = {k: end3_ed[k] / end3_tot[k] for k in end3_tot if end3_tot[k] >= 6}
    return dict(occ=occ, spaninit_rate=spaninit_rate, article_set=article_set,
                end2_rate=end2_rate, end3_rate=end3_rate, spaninit_slash=spaninit_slash)


def main():
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    it = train[train.language == "it"].copy()

    T = learn_it(train)  # full-train (viability estimate only; not for CV numbers)
    art = T["article_set"]
    print(f"learned it article_set size={len(art)}; sample spaninit_rates:",
          {w: round(T['spaninit_rate'][w], 2) for w in list(art)[:12]})

    # ---- multi_plain anchor viability ----
    n_mp = 0
    anchor_in_artset = 0
    anchor_spaninit_hi = 0
    follow_end_hi = 0
    both = 0
    first_core_ct = collections.Counter()
    for r in it.itertuples():
        for e in r.edits:
            if e["replacement"] == "":
                continue
            src = r.text[e["start"]:e["end"]]
            if span_type(src) != "multi_plain":
                continue
            n_mp += 1
            parts = src.split()
            fc = strip(parts[0]).lower()
            first_core_ct[fc] += 1
            a_in = fc in art
            a_hi = T["spaninit_rate"].get(fc, 0.0) >= 0.30
            # any following token with high gendered ending edit-rate
            fe = False
            for p in parts[1:]:
                c = strip(p).lower()
                if len(c) >= 2 and T["end2_rate"].get(c[-2:], 0.0) >= 0.30:
                    fe = True
            if a_in:
                anchor_in_artset += 1
            if a_hi:
                anchor_spaninit_hi += 1
            if fe:
                follow_end_hi += 1
            if a_hi and fe:
                both += 1
    print(f"\nmulti_plain spans={n_mp}")
    print(f"  first-token in learned article_set: {anchor_in_artset} ({anchor_in_artset/n_mp:.1%})")
    print(f"  first-token spaninit_rate>=0.30:    {anchor_spaninit_hi} ({anchor_spaninit_hi/n_mp:.1%})")
    print(f"  >=1 following tok end2_rate>=0.30:  {follow_end_hi} ({follow_end_hi/n_mp:.1%})")
    print(f"  BOTH anchor_hi AND follow_end_hi:   {both} ({both/n_mp:.1%})  <- NP-gen addressable")
    print("  top multi_plain first-tokens:", first_core_ct.most_common(12))

    # ---- unchanged-FP structure: group edit activity ----
    print("\n=== it group edit-activity structure ===")
    grp = collections.defaultdict(lambda: dict(rows=0, edited=0, unchanged=0))
    for r in it.itertuples():
        g = grp[r.document_group]
        g["rows"] += 1
        if len(r.edits) > 0:
            g["edited"] += 1
        else:
            g["unchanged"] += 1
    n_groups = len(grp)
    allunchanged = [g for g, d in grp.items() if d["edited"] == 0]
    print(f"it groups={n_groups}; all-unchanged groups={len(allunchanged)}")
    edrates = sorted((d["edited"] / d["rows"], g, d["rows"]) for g, d in grp.items())
    print("group edited-rate distribution (rate, group, nrows):")
    for rate, g, nr in edrates:
        print(f"    {rate:.2f}  {g}  n={nr}")

    # unchanged rows per group and their share
    unchanged_by_group = collections.Counter()
    for r in it.itertuples():
        if len(r.edits) == 0:
            unchanged_by_group[r.document_group] += 1
    tot_unchanged = sum(unchanged_by_group.values())
    print(f"\ntotal it unchanged rows={tot_unchanged}")
    # how many unchanged rows live in groups with edited-rate < 0.2?
    low_groups = {g for g, d in grp.items() if d["edited"] / d["rows"] < 0.2}
    unchanged_in_low = sum(unchanged_by_group[g] for g in low_groups)
    print(f"unchanged rows in low-activity groups (edited-rate<0.2): {unchanged_in_low} "
          f"(these are safe to zero via a doc-prior)")

    # ---- ORACLE ceilings on it ----
    print("\n=== it ORACLE ceilings ===")
    truth = {r.id: [{"start": e["start"], "end": e["end"], "replacement": e["replacement"]} for e in r.edits]
             for r in it.itertuples()}
    lm = {r.id: "it" for r in it.itertuples()}
    # current OOF edits
    oof = pd.read_csv(os.path.join(ROOT, "runs", "M4", "oof_edits.csv"))
    oofmap = {r.id: json.loads(r.edits_json) for r in oof.itertuples()}
    cur = {i: oofmap.get(i, []) for i in truth}
    s_cur, det_cur = elru.elru(cur, truth, lm, detail=True)
    print(f"current it lang={det_cur['it']['lang_score']:.4f} "
          f"edited={det_cur['it']['edited_mean']:.4f} unchanged={det_cur['it']['unchanged_mean']:.4f}")

    # oracle A: remove ALL FPs on unchanged rows (keep edited preds as-is)
    orA = {}
    for i in truth:
        if len(truth[i]) == 0:
            orA[i] = []
        else:
            orA[i] = cur[i]
    s_A, det_A = elru.elru(orA, truth, lm, detail=True)
    print(f"oracle A (zero all unchanged-row FPs): it lang={det_A['it']['lang_score']:.4f} "
          f"unchanged={det_A['it']['unchanged_mean']:.4f}")

    # oracle B: perfect multi_plain spans+repl added to current (edited rows), FPs untouched
    orB = {}
    for r in it.itertuples():
        i = r.id
        base = [dict(e) for e in cur[i]]
        if len(truth[i]) == 0:
            orB[i] = base
            continue
        # add perfect multi_plain edits, remove any current edit overlapping them
        mp = [e for e in r.edits if e["replacement"] != "" and span_type(r.text[e["start"]:e["end"]]) == "multi_plain"]
        keep = []
        for e in base:
            if any(not (e["end"] <= m["start"] or m["end"] <= e["start"]) for m in mp):
                continue
            keep.append(e)
        for m in mp:
            keep.append({"start": m["start"], "end": m["end"], "replacement": m["replacement"]})
        keep.sort(key=lambda e: e["start"])
        keep = elru._repair(keep, len(r.text)) if not elru.validate_edits(keep, len(r.text)) else keep
        orB[i] = keep
    s_B, det_B = elru.elru(orB, truth, lm, detail=True)
    print(f"oracle B (perfect multi_plain edited): it lang={det_B['it']['lang_score']:.4f} "
          f"edited={det_B['it']['edited_mean']:.4f}")

    # oracle C: A + B combined
    orC = {}
    for i in truth:
        orC[i] = [] if len(truth[i]) == 0 else orB[i]
    s_C, det_C = elru.elru(orC, truth, lm, detail=True)
    print(f"oracle C (A+B): it lang={det_C['it']['lang_score']:.4f}")

    # oracle D: perfect single_plain too (both single+multi plain perfect + FP removal)
    orD = {}
    for r in it.itertuples():
        i = r.id
        if len(truth[i]) == 0:
            orD[i] = []
            continue
        base = [dict(e) for e in cur[i]]
        plains = [e for e in r.edits if e["replacement"] != "" and "plain" in span_type(r.text[e["start"]:e["end"]])]
        keep = []
        for e in base:
            if any(not (e["end"] <= m["start"] or m["end"] <= e["start"]) for m in plains):
                continue
            keep.append(e)
        for m in plains:
            keep.append({"start": m["start"], "end": m["end"], "replacement": m["replacement"]})
        keep.sort(key=lambda e: e["start"])
        keep = elru._repair(keep, len(r.text)) if not elru.validate_edits(keep, len(r.text)) else keep
        orD[i] = keep
    s_D, det_D = elru.elru(orD, truth, lm, detail=True)
    print(f"oracle D (perfect ALL plain edited + FP removal): it lang={det_D['it']['lang_score']:.4f} "
          f"edited={det_D['it']['edited_mean']:.4f} unchanged={det_D['it']['unchanged_mean']:.4f}")


if __name__ == "__main__":
    main()
