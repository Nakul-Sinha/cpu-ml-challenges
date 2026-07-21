"""N1 DIAGNOSIS 3: is a leak-free ROW-level edit-activity prior able to remove it
unchanged-row FPs without killing edited recall?  This is the make-or-break test for
the doc/row-prior path (Oracle A = it lang .490).

Row features are RAW-TEXT-ONLY + OOF token-prob summaries (both leak-free):
  - n_tok, n_char, mean/max/top3 token prob, n_tok>=thr for a few thr
  - count of existing slash-double surface forms, colon/star marks
  - group edited-rate (learned per-fold from OTHER folds' labels -> leak-free feature)
Fit per fold on other folds -> predict P(row edited) for held-out fold.
"""
import os, sys, json, collections, re
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "solution"))
import elru

WS = re.compile(r"\S+")
_STRIP = ".,;:()»«\"'“”’`-–—"
_SLASHFORM = re.compile(r"[^\W\d_]/[^\W\d_]", re.UNICODE)
MARKS = set(":*∗")


def toks(t):
    return [(m.start(), m.end(), m.group()) for m in WS.finditer(t)]


def merge_threshold_spans(tk, probs, thr):
    spans = []; i = 0; n = len(tk)
    while i < n:
        if probs[i] >= thr:
            j = i
            while j + 1 < n and probs[j + 1] >= thr:
                j += 1
            spans.append((tk[i][0], tk[j][1]))
            i = j + 1
        else:
            i += 1
    return spans


def main():
    import lightgbm as lgb
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    tp = pd.read_csv(os.path.join(ROOT, "runs", "M4", "oof_token_probs.csv"))
    probs_by_id = {}; toks_by_id = {}
    for _id, g in tp.groupby("id"):
        g = g.sort_values("tok_index")
        probs_by_id[_id] = g.proba.tolist()
        toks_by_id[_id] = list(zip(g.start.tolist(), g.end.tolist()))

    it = train[train.language == "it"].reset_index(drop=True)
    ids = it.id.tolist()
    y = np.array([1 if len(e) > 0 else 0 for e in it.edits])
    fold = it.fold.values

    # group edited-rate, learned per-fold (leak-free) -- computed inside CV loop
    def group_rate_map(sub):
        gr = collections.defaultdict(lambda: [0, 0])
        for r in sub.itertuples():
            gr[r.document_group][1] += 1
            if len(r.edits) > 0:
                gr[r.document_group][0] += 1
        return {g: (v[0] / v[1] if v[1] else 0.5) for g, v in gr.items()}

    def row_feats(r, grate):
        tid = r.id
        pr = probs_by_id.get(tid, [])
        tk = toks_by_id.get(tid, [])
        text = r.text
        n = len(tk)
        pr_arr = np.array(pr) if pr else np.array([0.0])
        top3 = np.sort(pr_arr)[::-1][:3]
        nslash = len(_SLASHFORM.findall(text))
        nmark = sum(1 for ch in text if ch in MARKS)
        f = [
            float(n), float(len(text)),
            float(pr_arr.mean()), float(pr_arr.max()),
            float(top3.mean()), float(top3.sum()),
            float((pr_arr >= 0.3).sum()), float((pr_arr >= 0.45).sum()),
            float((pr_arr >= 0.6).sum()),
            float((pr_arr >= 0.45).mean()),
            float(nslash), float(nslash / max(1, n)),
            float(nmark), float(nmark / max(1, len(text))),
            float(grate.get(r.document_group, 0.5)),
        ]
        return f

    # ---- OOF row-prob via per-fold LGBM ----
    rowp = np.zeros(len(it))
    for k in range(5):
        tr = it[it.fold != k]
        va_idx = np.where(fold == k)[0]
        grate = group_rate_map(tr)  # leak-free: only other-fold rows
        Xtr = np.array([row_feats(r, grate) for r in tr.itertuples()], dtype=np.float32)
        ytr = np.array([1 if len(e) > 0 else 0 for e in tr.edits], dtype=np.int32)
        m = lgb.LGBMClassifier(objective="binary", n_estimators=200, learning_rate=0.05,
                               num_leaves=16, min_child_samples=20, subsample=0.9,
                               colsample_bytree=0.8, reg_lambda=2.0, random_state=0,
                               n_jobs=7, verbosity=-1)
        m.fit(Xtr, ytr)
        Xva = np.array([row_feats(it.iloc[i], grate) for i in va_idx], dtype=np.float32)
        rowp[va_idx] = m.predict_proba(Xva)[:, 1]

    # AUC
    from bisect import bisect_left
    order = np.argsort(rowp)
    pos = y[order].cumsum()
    # simple AUC
    npos = y.sum(); nneg = len(y) - npos
    ranks = np.argsort(np.argsort(rowp)) + 1
    auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    print(f"it row-level P(edited) OOF AUC = {auc:.4f}  (npos={npos} nneg={nneg})")

    # current predicted edits at ship thr 0.45 (span boundaries; recall proxy)
    thr = 0.45
    cur_pred_nonempty = {}
    for r in it.itertuples():
        tk = toks_by_id.get(r.id); pr = probs_by_id.get(r.id)
        cur_pred_nonempty[r.id] = len(merge_threshold_spans(tk, pr, thr)) > 0 if tk else False

    # among unchanged rows we currently FP on, and edited rows we currently fire on,
    # compare rowp distributions
    fp_rows = [i for i in range(len(it)) if y[i] == 0 and cur_pred_nonempty[ids[i]]]
    edfire = [i for i in range(len(it)) if y[i] == 1 and cur_pred_nonempty[ids[i]]]
    print(f"current it FPs (unchanged predicted-edit) = {len(fp_rows)}; edited-rows-fired = {len(edfire)}")
    print(f"  rowp on FP rows:     mean={np.mean([rowp[i] for i in fp_rows]):.3f} "
          f"median={np.median([rowp[i] for i in fp_rows]):.3f}")
    print(f"  rowp on edited-fire: mean={np.mean([rowp[i] for i in edfire]):.3f} "
          f"median={np.median([rowp[i] for i in edfire]):.3f}")

    # ---- GATING SWEEP: zero out it rows with rowp < cut (drop all their edits) ----
    # build current edits map (use oof_edits for real replacement quality)
    oof = pd.read_csv(os.path.join(ROOT, "runs", "M4", "oof_edits.csv"))
    oofmap = {r.id: json.loads(r.edits_json) for r in oof.itertuples()}
    truth = {r.id: [{"start": e["start"], "end": e["end"], "replacement": e["replacement"]} for e in r.edits]
             for r in it.itertuples()}
    lm = {i: "it" for i in ids}
    base_edits = {i: oofmap.get(i, []) for i in ids}
    s0, d0 = elru.elru(base_edits, truth, lm, detail=True)
    print(f"\nbase it lang={d0['it']['lang_score']:.4f} edited={d0['it']['edited_mean']:.4f} "
          f"unchanged={d0['it']['unchanged_mean']:.4f}")
    print("\n=== ROW-GATE sweep (drop ALL edits on it rows with rowp<cut) ===")
    print("cut    fp_removed  edited_killed   it_lang   edited   unchanged")
    for cut in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        gated = {}
        fp_removed = 0; edited_killed = 0
        for i, tid in enumerate(ids):
            if rowp[i] < cut:
                gated[tid] = []
                if y[i] == 0 and cur_pred_nonempty[tid]:
                    fp_removed += 1
                if y[i] == 1 and len(base_edits[tid]) > 0:
                    edited_killed += 1
            else:
                gated[tid] = base_edits[tid]
        s, d = elru.elru(gated, truth, lm, detail=True)
        print(f"{cut:.2f}   {fp_removed:>4d}        {edited_killed:>4d}          "
              f"{d['it']['lang_score']:.4f}   {d['it']['edited_mean']:.4f}   {d['it']['unchanged_mean']:.4f}")

    # ---- SCALE approach: multiply token probs by rowp, re-threshold, re-score span-recall only
    # (quick proxy: does scaling change which rows fire) -- report best achievable lang via gate
    print("\n(Interpretation: pick the cut that maximizes it_lang above; that's the row-prior ceiling on this base.)")


if __name__ == "__main__":
    main()
