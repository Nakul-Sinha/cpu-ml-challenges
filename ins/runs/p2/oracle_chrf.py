"""Oracle-span chrF harness: isolates TRANSDUCTION quality.

Leak-free per fold: fit transducer on train[fold!=k], predict the replacement for
every TRUE edit span in fold k, score replacement_chrf vs the true replacement (the
canonical elru metric).  Breaks down by (lang, type).  Compares baseline vs enhanced
(with ablation flags) so each lever is measured in isolation of detection.

Usage: python oracle_chrf.py            # baseline vs full enhanced
       python oracle_chrf.py ablate     # per-flag ablation
"""
import os, sys, json, re, collections
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = HERE
for cand in (HERE, os.path.expanduser("~/insled")):
    if os.path.exists(os.path.join(cand, "dataset", "train.csv")):
        ROOT = cand; break
    if os.path.exists(os.path.join(cand, "train.csv")):
        ROOT = cand; break
sys.path.insert(0, os.path.join(ROOT, "solution"))
import elru
import transducer as T_base
import transducer_p2 as T_p2

WS = re.compile(r"\S+")
MARKS = set(":*∗/")


def load():
    tp = os.path.join(ROOT, "dataset", "train.csv")
    if not os.path.exists(tp):
        tp = os.path.join(HERE, "train.csv")
    fp = os.path.join(ROOT, "solution", "folds.csv")
    if not os.path.exists(fp):
        fp = os.path.join(HERE, "folds.csv")
    tr = pd.read_csv(tp); folds = pd.read_csv(fp)
    tr = tr.merge(folds, on="id"); tr["edits"] = tr.edits_json.apply(json.loads)
    return tr


def etype(src, rep):
    if rep == "":
        return "deletion"
    st = [m.group() for m in WS.finditer(src)]
    has_mark = any(c in MARKS for c in src)
    if len(st) == 1:
        return "single_mark" if has_mark else "single_plain"
    return "multi_mark" if has_mark else "multi_plain"


def make_enh(**flags):
    def factory():
        t = T_p2.Transducer()
        for k, v in flags.items():
            setattr(t, k, v)
        return t
    return factory


def evaluate(tr, factory, label):
    # per (lang,type): list of chrf; also overall per-lang mean over edited spans
    by = collections.defaultdict(list)
    span_by_lang = collections.defaultdict(list)   # chrf per non-empty edit span
    for k in range(5):
        trdf = tr[tr.fold != k]
        vadf = tr[tr.fold == k]
        T = factory().fit(trdf)
        for r in vadf.itertuples():
            for e in r.edits:
                src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
                if rep == "":
                    continue
                pred = T.predict(r.language, src, {"text": r.text, "start": e["start"],
                                                   "end": e["end"], "lang": r.language})
                if pred is None:
                    pred = src
                c = elru.replacement_chrf(pred, rep)
                by[(r.language, etype(src, rep))].append(c)
                span_by_lang[r.language].append(c)
    return by, span_by_lang


def report(by, span_by_lang, label):
    print(f"\n===== {label} =====")
    print(f"  {'lang/type':22s} {'n':>4s}  {'mean_chrf':>9s}")
    for key in sorted(by):
        v = by[key]
        print(f"  {key[0]+' '+key[1]:22s} {len(v):>4d}  {sum(v)/len(v):>9.4f}")
    print("  -- per-lang mean over all non-empty edit spans --")
    for L in ("de", "en", "it"):
        v = span_by_lang.get(L, [])
        if v:
            print(f"    {L}: n={len(v)} mean_chrf={sum(v)/len(v):.4f}")


def main():
    tr = load()
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    base_by, base_lang = evaluate(tr, lambda: T_base.Transducer(), "BASELINE")
    report(base_by, base_lang, "BASELINE transducer.py")

    if mode == "ablate":
        configs = [
            ("+decomp only (no append)", make_enh(USE_MULTI_DECOMP=True, USE_APPEND=False, IT_AGREE=False)),
            ("+append only (no decomp)", make_enh(USE_MULTI_DECOMP=False, USE_APPEND=True, IT_AGREE=False)),
            ("+decomp+append (suffix-first)", make_enh(USE_MULTI_DECOMP=True, USE_APPEND=True, APPEND_AFTER_SUFFIX=True, IT_AGREE=False)),
            ("+decomp+append (append-first)", make_enh(USE_MULTI_DECOMP=True, USE_APPEND=True, APPEND_AFTER_SUFFIX=False, IT_AGREE=False)),
            ("FULL (it agree on)", make_enh()),
        ]
    else:
        configs = [("FULL ENHANCED", make_enh())]
    deltas = {}
    for label, fac in configs:
        by, lang = evaluate(tr, fac, label)
        report(by, lang, label)
        # delta vs baseline on key it multi_plain + overall
        d = {}
        for key in (("it", "multi_plain"), ("it", "single_plain"), ("de", "multi_plain")):
            b = base_by.get(key, []); e = by.get(key, [])
            if b and e:
                d[f"{key[0]}_{key[1]}"] = round(sum(e)/len(e) - sum(b)/len(b), 4)
        for L in ("de", "it", "en"):
            b = base_lang.get(L, []); e = lang.get(L, [])
            if b and e:
                d[f"{L}_allspans"] = round(sum(e)/len(e) - sum(b)/len(b), 4)
        deltas[label] = d
    print("\n===== DELTAS vs baseline (chrf) =====")
    for label, d in deltas.items():
        print(f"  {label}: {d}")


if __name__ == "__main__":
    main()
