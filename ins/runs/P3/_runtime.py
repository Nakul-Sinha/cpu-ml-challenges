# ======================================================================
#  P3 SHIP RUNTIME  (appended after the embedded-module bootstrap)
#  Self-contained: no imports from runs/.  Implements the v4 ship pipeline:
#    de = (1-a)*shared_LGBM + a*BiGRU  (a=0.6) threshold-merge
#    en = shared_LGBM threshold-merge  (BiGRU measured-and-dropped: en frozen)
#    it = NP-gate assembly on shared+IT-rescorer-boosted prob, P2 transducer
#    + de/en group-consistency vote (hi.60/lo.40)
#  Every model FIT AT RUNTIME on train.csv; operating points are pre-committed
#  hyperparameters selected by the honest nested/non-nested CV in pipeline_v4.py.
# ======================================================================
import numpy as np
import pandas as pd
import pipeline, n2_ext, run_m4, run_n1
from transducer import Transducer

# torch is only needed for the de BiGRU lever; import guarded so a torch-less
# environment still ships a valid (de-shared) submission via the fallback.
import random, zlib
try:
    import torch
    import torch.nn as _tnn
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

# ---- determinism: fixed thread count + seeds so re-runs are byte-identical ----
_TORCH_THREADS = 4
if _HAVE_TORCH:
    try:
        torch.set_num_threads(_TORCH_THREADS)
    except Exception:
        pass
    torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

LANGS = pipeline.LANGS
_STRIP = ".,;:()»«\"'“”’`-–—"
MARKS = set(":*∗/")

# ======================================================================
#  BAKED OPERATING POINTS  (selected by honest CV in pipeline_v4.py; see
#  cv_report_v4.json "ops").  These are scalar hyperparameters, not answers.
# ======================================================================
DE_A = 0.6            # de ensemble weight (BiGRU), pre-committed (not CV-maximised)
DE_THR = 0.31         # SHIP de spine threshold on a=0.6 ensembled prob = median of the
                      # per-fold nested picks [.19,.31,.29,.35,.31]; robust pre-commitment.
                      # (ship-fixed honest nested 0.5777; de edited-ratio 0.82, in-band.)
DE_THR_CVOPT = 0.19   # alt: non-nested all-OOF de optimum (submission_cvopt)
EN_THR = 0.39         # en spine threshold (a=0, shared prob) (non-nested optimum)
IT_SPINE = 0.45       # it base-merge spine threshold
IT_GATE = 0.80        # it NP-gate admission threshold
IT_BOOST_SRC = "rescorer"
IT_BOOST_W = 0.60     # it additive-boost weight (non-nested optimum)
GRU_SEEDS = 5         # BiGRU seed-ensemble size (variance reduction)

# ======================================================================
#  P1 LEVER 2 -- BiGRU per-token edit tagger (core copied from p1_lever2.py)
# ======================================================================
NGV = 4096; NG = 24
EMB = 32; LEMB = 8; HID = 48; EPOCHS = 16; BATCH = 32; LR = 3e-3
LANG2I = {"de": 0, "en": 1, "it": 2}


def _h(s, b=NGV):
    return int(zlib.crc32(s.encode("utf-8")) % b) + 1   # 0 = pad


def _tok_ngrams(core):
    s = "^" + core + "$"
    out = []
    for n in (1, 2, 3):
        for i in range(len(s) - n + 1):
            out.append(_h(s[i:i + n]))
            if len(out) >= NG:
                return out
    return out


def _token_scalars(lex, L, w):
    lw = w.lower(); core = w.strip(_STRIP)

    def rt(ed, sn, k, a):
        return lex["rate"](ed, sn, L, k, a)
    inner = w[1:-1] if len(w) > 2 else ""
    sk = pipeline.special_key(w)
    spat = rt(lex["spat_ed"], lex["spat_sn"], (sk[0] + sk[1]) if sk else "", 3.0) if sk else 0.0
    specsuf = rt(lex["suf_ed"], lex["suf_sn"], sk[1], 3.0) if sk else 0.0
    return [
        rt(lex["tok_ed"], lex["tok_sn"], w, 5.0),
        rt(lex["suf3_ed"], lex["suf3_sn"], lw[-3:], 20.0),
        rt(lex["suf4_ed"], lex["suf4_sn"], lw[-4:], 30.0),
        rt(lex["pre3_ed"], lex["pre3_sn"], lw[:3], 20.0),
        spat, specsuf,
        1.0 if any(c in MARKS for c in w) else 0.0,
        1.0 if any((not c.isalnum()) for c in inner) else 0.0,
        1.0 if w[:1].isupper() else 0.0,
        1.0 if (w.isupper() and any(c.isalpha() for c in w)) else 0.0,
        min(len(core), 20) / 20.0,
    ]


NSCAL = 11 + 3


def _build_seqs(rows, lex):
    seqs = {}
    for R in rows:
        L = R["lang"]; tk = R["tk"]; n = len(tk)
        ng = np.zeros((n, NG), np.int64); sc = np.zeros((n, NSCAL), np.float32)
        lg = np.full(n, LANG2I[L], np.int64)
        yy = R.get("y", None)
        y = np.asarray(yy, np.float32) if yy is not None else np.zeros(n, np.float32)
        for i, (s, e, w) in enumerate(tk):
            core = w.strip(_STRIP).lower() or w.lower()
            g = _tok_ngrams(core)
            ng[i, :len(g)] = g[:NG]
            sc[i, :11] = _token_scalars(lex, L, w)
            sc[i, 11] = i / max(n - 1, 1); sc[i, 12] = 1.0 if i == 0 else 0.0
            sc[i, 13] = 1.0 if i == n - 1 else 0.0
        seqs[R["id"]] = (ng, lg, sc, y)
    return seqs


if _HAVE_TORCH:
    class BiGRUTagger(_tnn.Module):
        def __init__(self):
            super().__init__()
            self.emb = _tnn.Embedding(NGV + 1, EMB, padding_idx=0)
            self.lemb = _tnn.Embedding(3, LEMB)
            self.gru = _tnn.GRU(EMB + LEMB + NSCAL, HID, batch_first=True, bidirectional=True)
            self.drop = _tnn.Dropout(0.2)
            self.out = _tnn.Linear(2 * HID, 1)

        def forward(self, ng, lg, sc):
            e = self.emb(ng)
            ngm = (ng > 0).float().unsqueeze(-1)
            tok = (e * ngm).sum(2) / ngm.sum(2).clamp(min=1)
            x = torch.cat([tok, self.lemb(lg), sc], dim=-1)
            hgru, _ = self.gru(x)
            return self.out(self.drop(hgru)).squeeze(-1)


def _pad_batch(ids, seqs):
    T = max(seqs[i][0].shape[0] for i in ids)
    B = len(ids)
    ng = np.zeros((B, T, NG), np.int64); lg = np.zeros((B, T), np.int64)
    sc = np.zeros((B, T, NSCAL), np.float32); y = np.zeros((B, T), np.float32)
    mask = np.zeros((B, T), np.float32)
    for b, i in enumerate(ids):
        a, l, s, yy = seqs[i]; t = a.shape[0]
        ng[b, :t] = a; lg[b, :t] = l; sc[b, :t] = s; y[b, :t] = yy; mask[b, :t] = 1.0
    return (torch.from_numpy(ng), torch.from_numpy(lg), torch.from_numpy(sc),
            torch.from_numpy(y), torch.from_numpy(mask))


def _gru_train_predict(tr_ids, va_ids, seqs, pos_w, seed=0):
    # pin BOTH RNGs per seed so run-to-run output is byte-identical (smoke-test parity),
    # independent of any prior random/torch usage in the process.
    torch.manual_seed(seed)
    random.seed(10_000 + seed)
    model = BiGRUTagger()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    lossf = _tnn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_w))
    order = list(tr_ids)
    model.train()
    for ep in range(EPOCHS):
        random.shuffle(order)
        for b0 in range(0, len(order), BATCH):
            ids = order[b0:b0 + BATCH]
            ng, lg, sc, y, mask = _pad_batch(ids, seqs)
            opt.zero_grad()
            logit = model(ng, lg, sc)
            l = (lossf(logit, y) * mask).sum() / mask.sum()
            l.backward(); opt.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for b0 in range(0, len(va_ids), BATCH):
            ids = va_ids[b0:b0 + BATCH]
            ng, lg, sc, y, mask = _pad_batch(ids, seqs)
            p = torch.sigmoid(model(ng, lg, sc)).numpy()
            for b, i in enumerate(ids):
                t = seqs[i][0].shape[0]
                out[i] = p[b, :t]
    return out


def gru_full_probs(train_rows, test_rows, n_seeds=GRU_SEEDS):
    """Train n_seeds BiGRUs on ALL train rows (full lexicon), average per-token probs
    on test rows.  Deterministic (fixed seeds + fixed thread count)."""
    if not _HAVE_TORCH:
        return {R["id"]: np.zeros(len(R["tk"]), np.float32) for R in test_rows}
    lex = pipeline.build_lexicon(train_rows)
    seqs = _build_seqs(train_rows + test_rows, lex)
    ypos = sum(int(seqs[R["id"]][3].sum()) for R in train_rows)
    yall = sum(seqs[R["id"]][3].shape[0] for R in train_rows)
    pos_w = max(1.0, (yall - ypos) / max(ypos, 1)) ** 0.5
    tr_ids = [R["id"] for R in train_rows]; te_ids = [R["id"] for R in test_rows]
    acc = None
    for s in range(n_seeds):
        pred = _gru_train_predict(tr_ids, te_ids, seqs, pos_w, seed=s)
        if acc is None:
            acc = {i: v.astype(np.float64) for i, v in pred.items()}
        else:
            for i, v in pred.items():
                acc[i] += v
    return {i: (acc[i] / n_seeds) for i in acc}


# ======================================================================
#  P1 LEVER 1 -- IT-only LGBM re-scorer (core copied from p1_lever1.py)
# ======================================================================
import collections as _collections
NBH = 512
RESC_PARAMS = dict(objective="binary", n_estimators=350, learning_rate=0.04, num_leaves=24,
                   min_child_samples=25, subsample=0.85, subsample_freq=1, colsample_bytree=0.8,
                   reg_lambda=3.0, is_unbalance=True, random_state=0, n_jobs=7, verbosity=-1)
IT_CAT_NAMES = ["suf2_id", "suf3_id", "pre2_id", "tok_id", "prev_id", "next_id"]
_IT_FEAT_FROZEN = [False]


def _hb(s, b=NBH):
    return int(zlib.crc32(s.encode("utf-8")) % b)


def learn_it_morph(trdf):
    import json as _json
    suf = [_collections.Counter() for _ in range(4)]
    suft = [_collections.Counter() for _ in range(4)]
    pre_ed = _collections.Counter(); pre_tot = _collections.Counter()
    tok_ed = _collections.Counter(); tok_tot = _collections.Counter()
    lang_ed = 0; lang_tot = 0
    for r in trdf[trdf.language == "it"].itertuples():
        edits = r.edits if isinstance(r.edits, list) else _json.loads(r.edits_json)
        tk = [(m.start(), m.end(), m.group()) for m in pipeline.WORD_RE.finditer(r.text)]
        spans = sorted((e["start"], e["end"], e["replacement"]) for e in edits)

        def inside(s, e):
            return any(s >= a and e <= b for a, b, _ in spans)
        for s, e, w in tk:
            core = w.strip(_STRIP).lower()
            if not core:
                continue
            isin = 1 if inside(s, e) else 0
            lang_ed += isin; lang_tot += 1
            tok_ed[core] += isin; tok_tot[core] += 1
            for L in (1, 2, 3):
                if len(core) >= L:
                    suf[L][core[-L:]] += isin; suft[L][core[-L:]] += 1
            if len(core) >= 2:
                pre_ed[core[:2]] += isin; pre_tot[core[:2]] += 1
    prior = (lang_ed + 0.5) / (lang_tot + 1.0)

    def mk(ed, tot, a):
        return {k: (ed[k] + a * prior) / (tot[k] + a) for k in tot}
    return dict(prior=prior,
                suf1=mk(suf[1], suft[1], 8.0), suf2=mk(suf[2], suft[2], 12.0),
                suf3=mk(suf[3], suft[3], 20.0), pre2=mk(pre_ed, pre_tot, 12.0),
                tok=mk(tok_ed, tok_tot, 5.0), tok_tot=tok_tot)


def _it_feats(R, i, tab, gc, gbi, morph):
    import re as _re
    tk = R["tk"]; n = len(tk); text = R["text"]
    group = gbi[R["id"]]; gs, gsz = gc.get(group, (0.0, 0.0))
    w = tk[i][2]; core = w.strip(_STRIP).lower()
    cl = len(core)
    feats = []
    def add(v):
        feats.append(float(v))
    add(morph["suf1"].get(core[-1:], morph["prior"]) if cl >= 1 else morph["prior"])
    add(morph["suf2"].get(core[-2:], morph["prior"]) if cl >= 2 else morph["prior"])
    add(morph["suf3"].get(core[-3:], morph["prior"]) if cl >= 3 else morph["prior"])
    add(morph["pre2"].get(core[:2], morph["prior"]) if cl >= 2 else morph["prior"])
    add(morph["tok"].get(core, morph["prior"]))
    add(np.log1p(morph["tok_tot"].get(core, 0.0)))
    add(tab["tok_edrate"].get(core, 0.0))
    add(tab["end2_rate"].get(core[-2:], 0.0) if cl >= 2 else 0.0)
    add(tab["spaninit_rate"].get(core, 0.0))
    add(cl); add(1.0 if w[:1].isupper() else 0.0)
    add(1.0 if (w.isupper() and any(c.isalpha() for c in w)) else 0.0)
    add(1.0 if any(c in MARKS for c in w) else 0.0)
    add(1.0 if any((not c.isalnum()) for c in w[1:-1]) else 0.0)
    dprev = 99; dnext = 99
    for d in range(1, 6):
        if i - d >= 0 and dprev == 99:
            pc = tk[i - d][2].strip(_STRIP).lower()
            if pc in tab["anchors"] or tab["spaninit_rate"].get(pc, 0.0) >= 0.30:
                dprev = d
        if i + d < n and dnext == 99:
            nc = tk[i + d][2].strip(_STRIP).lower()
            if nc in tab["anchors"] or tab["spaninit_rate"].get(nc, 0.0) >= 0.30:
                dnext = d
    add(min(dprev, 6)); add(min(dnext, 6))
    add(1.0 if dprev <= 3 else 0.0)

    def hi2(j):
        if 0 <= j < n:
            cj = tk[j][2].strip(_STRIP).lower()
            return len(cj) >= 2 and morph["suf2"].get(cj[-2:], 0.0) >= 0.20
        return False
    chain = sum(1 for j in range(i - 2, i + 3) if hi2(j))
    add(float(chain))
    rl = 0
    if hi2(i):
        rl = 1; j = i - 1
        while hi2(j):
            rl += 1; j -= 1
        j = i + 1
        while hi2(j):
            rl += 1; j += 1
    add(float(rl))
    pc = tk[i - 1][2].strip(_STRIP).lower() if i - 1 >= 0 else ""
    nc = tk[i + 1][2].strip(_STRIP).lower() if i + 1 < n else ""
    add(1.0 if (pc in tab["anchors"] or tab["spaninit_rate"].get(pc, 0.0) >= 0.30) else 0.0)
    add(morph["suf2"].get(nc[-2:], 0.0) if len(nc) >= 2 else 0.0)
    add(morph["suf2"].get(pc[-2:], 0.0) if len(pc) >= 2 else 0.0)
    s0, e0 = tk[i][0], tk[i][1]
    win = text[max(0, s0 - 90):e0 + 90]
    add(1.0 if _re.search(r"[^\W\d_]/[^\W\d_]", win) else 0.0)
    add(i / max(n - 1, 1)); add(1.0 if i == 0 else 0.0)
    add(1.0 if i == n - 1 else 0.0); add(np.log1p(n))
    add(gs); add(np.log1p(gsz))
    catstart = len(feats)
    add(_hb(core[-2:]) if cl >= 2 else 0)
    add(_hb(core[-3:]) if cl >= 3 else 0)
    add(_hb(core[:2]) if cl >= 2 else 0)
    add(_hb(core))
    add(_hb(pc) if pc else 0)
    add(_hb(nc) if nc else 0)
    cat_idx = list(range(catstart, len(feats)))
    return feats, cat_idx


def _it_matrix(itrows, tab, gc, gbi, morph, labeled=True):
    out = {}; cat_idx = None
    for R in itrows:
        X = []; ci = None
        for i in range(len(R["tk"])):
            f, ci = _it_feats(R, i, tab, gc, gbi, morph)
            X.append(f)
        y = np.asarray(R["y"], np.int32) if (labeled and "y" in R) else None
        out[R["id"]] = (np.asarray(X, np.float32), y)
        cat_idx = ci
    return out, cat_idx


def rescorer_full_probs(it_train_rows, it_test_rows, train_df, tab, gc, gbi):
    """Train the IT re-scorer on ALL it train tokens; predict test it tokens.
    Independent view (no shared-prob features), matching the v4 selection."""
    import lightgbm as lgb
    morph = learn_it_morph(train_df)
    mats_tr, cat_idx = _it_matrix(it_train_rows, tab, gc, gbi, morph, labeled=True)
    if not it_train_rows:
        return {}
    Xtr = np.concatenate([mats_tr[R["id"]][0] for R in it_train_rows])
    ytr = np.concatenate([mats_tr[R["id"]][1] for R in it_train_rows])
    m = lgb.LGBMClassifier(**RESC_PARAMS)
    m.fit(Xtr, ytr, categorical_feature=cat_idx)
    mats_te, _ = _it_matrix(it_test_rows, tab, gc, gbi, morph, labeled=False)
    p_it = {}
    for R in it_test_rows:
        X = mats_te[R["id"]][0]
        p_it[R["id"]] = (m.predict_proba(X)[:, 1] if len(X) else np.zeros(0))
    return p_it


# ======================================================================
#  IT NP-gate: fit gate on ALL train it NP candidates (from run_n1)
# ======================================================================
def fit_it_gate(all_rows, det_full, tab_full, gc_full, gbi_tr):
    import lightgbm as lgb
    itrows_tr = [R for R in all_rows if R["lang"] == "it"]
    tp_tr = det_full.token_probs(itrows_tr)
    Xtr, ytr = [], []
    for R in itrows_tr:
        pr = tp_tr[R["id"]][1]
        cs = run_n1.np_cands(R["tk"], R["text"], gbi_tr[R["id"]], pr, tab_full, gc_full)
        for (a, b, f) in cs:
            best = max((max(0, min(b, te) - max(a, ts)) / (max(b, te) - min(a, ts))
                        for (ts, te, rep) in R["spans"] if rep != "" and max(b, te) > min(a, ts)), default=0.0)
            Xtr.append(f); ytr.append(1 if best >= 0.5 else 0)
    gate = lgb.LGBMClassifier(**run_n1.GATE_PARAMS)
    gate.fit(np.asarray(Xtr, np.float32), np.asarray(ytr, np.int32))
    return gate


# ======================================================================
#  SHIP: full-train artifacts (expensive, once) + assembly (cheap, per de_thr)
# ======================================================================
def ship_artifacts(train, test):
    """Fit everything on full train; compute all test-row probabilities.  Returned
    dict is de_thr-independent, so a submission can be assembled at any de threshold."""
    n2_ext.register(pipeline)
    gbi_tr = {r.id: r.document_group for r in train.itertuples()}
    gbi_te = {r.id: r.document_group for r in test.itertuples()}

    stores_full = {}
    for b in pipeline.STORE_BUILDERS:
        b(train, stores_full)
    all_rows = pipeline.build_rows(train, labeled=True)
    det_full = pipeline.Detector().fit(all_rows, stores_full)
    trd_full = Transducer().fit(train)                      # P2 (it-enhanced) transducer
    tab_full = run_n1.learn_tab(train); gc_full = run_n1.group_ctx(train)
    gate_model = fit_it_gate(all_rows, det_full, tab_full, gc_full, gbi_tr)

    test_rows = pipeline.build_rows(test, labeled=False)
    tp_test = det_full.token_probs(test_rows)
    shared = {R["id"]: np.asarray(tp_test[R["id"]][1]) for R in test_rows}
    seq = gru_full_probs(all_rows, test_rows, n_seeds=GRU_SEEDS)     # P1 lever 2 (de)
    it_train_rows = [R for R in all_rows if R["lang"] == "it"]
    it_test_rows = [R for R in test_rows if R["lang"] == "it"]
    gbi_all = {**gbi_tr, **gbi_te}   # re-scorer needs group-ids for train AND test rows
    p_it = rescorer_full_probs(it_train_rows, it_test_rows, train, tab_full, gc_full, gbi_all)  # P1 lever 1 (it)
    return dict(stores_full=stores_full, trd_full=trd_full, tab_full=tab_full, gc_full=gc_full,
                gate_model=gate_model, test_rows=test_rows, tp_test=tp_test, shared=shared,
                seq=seq, p_it=p_it, gbi_te=gbi_te)


def assemble_submission(art, de_thr=DE_THR):
    """Assemble the submission from precomputed artifacts at a chosen de threshold."""
    stores_full = art["stores_full"]; trd_full = art["trd_full"]
    tab_full = art["tab_full"]; gc_full = art["gc_full"]; gate_model = art["gate_model"]
    test_rows = art["test_rows"]; tp_test = art["tp_test"]; shared = art["shared"]
    seq = art["seq"]; p_it = art["p_it"]; gbi_te = art["gbi_te"]

    sub = {}
    for R in test_rows:
        rid = R["id"]; tk = tp_test[rid][0]; L = R["lang"]; text = R["text"]
        sh = shared[rid]
        if L == "de":
            ens = ((1 - DE_A) * sh + DE_A * seq[rid]).tolist()
            sub[rid] = pipeline.build_edits(rid, text, "de", tk, ens, de_thr, trd_full, stores_full)
        elif L == "en":
            sub[rid] = pipeline.build_edits(rid, text, "en", tk, sh.tolist(), EN_THR, trd_full, stores_full)
        else:  # it
            pit = p_it.get(rid, None)
            if pit is not None and len(pit) == len(sh):
                boosted = np.clip(sh + IT_BOOST_W * np.clip(pit - 0.3, 0, None), 0, 1).tolist()
            else:
                boosted = sh.tolist()
            cs = run_n1.np_cands(tk, text, gbi_te[rid], sh.tolist(), tab_full, gc_full)
            gscore = []
            if cs:
                pv = gate_model.predict_proba(np.asarray([c[2] for c in cs], np.float32))[:, 1]
                gscore = [(c[0], c[1], float(p)) for c, p in zip(cs, pv)]
            sub[rid] = run_n1.assemble_it(tk, text, boosted, IT_GATE, gscore, trd_full, stores_full)

    test_by_id = {R["id"]: R for R in test_rows}
    idf = {i: 0 for i in sub}
    sub = run_m4.group_consistency(sub, test_by_id, gbi_te, {0: trd_full}, {0: stores_full}, idf,
                                   vote_langs=run_m4.SHIP_VOTE_LANGS, drop_langs=run_m4.SHIP_VOTE_LANGS,
                                   do_conv=False)
    return sub


def build_submission(train, test, de_thr=DE_THR):
    art = ship_artifacts(train, test)
    return assemble_submission(art, de_thr=de_thr), art["test_rows"]
