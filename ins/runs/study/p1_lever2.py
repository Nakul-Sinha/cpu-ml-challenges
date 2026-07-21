"""P1 LEVER 2 -- neural BiGRU per-token edit tagger (all languages).

Small bidirectional GRU over the token sequence of each row.  Per-token inputs:
  * hashed char-ngram (1..3) token embedding (vocab-free -> handles the cipher),
    mean-pooled over the token's n-grams;
  * language embedding;
  * the pipeline lexicon-rate scalars (tok/suf3/suf4/pre3/specsuf rates) + a few
    structural flags + position.
Sequence context (the BiGRU) sharpens exactly the context-dependent de/it plain
tokens where the independent per-token LGBM caps.  Trained PER FOLD (leak-free OOF),
CPU minutes.  Ensembled with the shared LGBM prob by weighted averaging; the ensemble
weight is tuned NESTED per language.  For it the ensembled prob is additionally tried
as an additive boost (the Lever-1 winning integration).

Reports per-language token PR-AUC (shared vs GRU vs ensemble) and the resulting nested
it / de / en lang_scores + overall on the N3 base.

Run: cd ~/insled && OMP_NUM_THREADS=7 nice -n 10 ~/venv/bin/python runs/P1/p1_lever2.py
"""
import os, sys, json, time, collections, zlib, random
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/insled")
for p in (os.path.join(ROOT, "runs", "M4"), os.path.join(ROOT, "runs", "N2"),
          os.path.join(ROOT, "runs", "N1"), os.path.join(ROOT, "solution"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
import pipeline, elru
from run_n1 import assemble_it as n1_assemble_it, IT_SPINE_THR
from run_m4 import base_cache, base_select, group_consistency, score_edits, fp_counts, per_type_recall, SHIP_VOTE_LANGS
import p1_base

torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.set_num_threads(7)
LANGS = pipeline.LANGS
LANG2I = {"de": 0, "en": 1, "it": 2}
_STRIP = ".,;:()»«\"'“”’`-–—"
MARKS = set(":*∗/")
NGV = 4096          # char-ngram hash buckets
NG = 24             # max ngrams per token
EMB = 32; LEMB = 8; HID = 48; EPOCHS = 16; BATCH = 32; LR = 3e-3


def h(s, b=NGV):
    return int(zlib.crc32(s.encode("utf-8")) % b) + 1   # 0 = pad


def tok_ngrams(core):
    s = "^" + core + "$"
    out = []
    for n in (1, 2, 3):
        for i in range(len(s) - n + 1):
            out.append(h(s[i:i + n]))
            if len(out) >= NG:
                return out
    return out


def token_scalars(lex, L, w):
    """compact pipeline lexicon-rate scalars + structural + (filled elsewhere: position)."""
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


NSCAL = 11 + 3   # lexicon/struct (11) + position (pos, is_first, is_last)


def build_seqs(rows, lex):
    """Per row: (ngram_ids[T,NG], lang[T], scalars[T,NSCAL], y[T])."""
    seqs = {}
    for R in rows:
        L = R["lang"]; tk = R["tk"]; n = len(tk)
        ng = np.zeros((n, NG), np.int64); sc = np.zeros((n, NSCAL), np.float32)
        lg = np.full(n, LANG2I[L], np.int64)
        for i, (s, e, w) in enumerate(tk):
            core = w.strip(_STRIP).lower() or w.lower()
            g = tok_ngrams(core)
            ng[i, :len(g)] = g[:NG]
            base = token_scalars(lex, L, w)
            sc[i, :11] = base
            sc[i, 11] = i / max(n - 1, 1); sc[i, 12] = 1.0 if i == 0 else 0.0
            sc[i, 13] = 1.0 if i == n - 1 else 0.0
        seqs[R["id"]] = (ng, lg, sc, np.asarray(R["y"], np.float32))
    return seqs


class BiGRUTagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NGV + 1, EMB, padding_idx=0)
        self.lemb = nn.Embedding(3, LEMB)
        self.gru = nn.GRU(EMB + LEMB + NSCAL, HID, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(2 * HID, 1)

    def forward(self, ng, lg, sc, mask):
        # ng [B,T,NG] -> mean over valid ngrams -> [B,T,EMB]
        e = self.emb(ng)                                  # [B,T,NG,EMB]
        ngm = (ng > 0).float().unsqueeze(-1)              # [B,T,NG,1]
        tok = (e * ngm).sum(2) / ngm.sum(2).clamp(min=1)  # [B,T,EMB]
        x = torch.cat([tok, self.lemb(lg), sc], dim=-1)   # [B,T,EMB+LEMB+NSCAL]
        hgru, _ = self.gru(x)
        return self.out(self.drop(hgru)).squeeze(-1)      # [B,T]


def pad_batch(ids, seqs):
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


def train_predict(tr_ids, va_ids, seqs, pos_w, epochs=EPOCHS, seed=0):
    torch.manual_seed(seed)
    model = BiGRUTagger()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_w))
    order = list(tr_ids)
    model.train()
    for ep in range(epochs):
        random.shuffle(order)
        for b0 in range(0, len(order), BATCH):
            ids = order[b0:b0 + BATCH]
            ng, lg, sc, y, mask = pad_batch(ids, seqs)
            opt.zero_grad()
            logit = model(ng, lg, sc, mask)
            l = (lossf(logit, y) * mask).sum() / mask.sum()
            l.backward(); opt.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for b0 in range(0, len(va_ids), BATCH):
            ids = va_ids[b0:b0 + BATCH]
            ng, lg, sc, y, mask = pad_batch(ids, seqs)
            p = torch.sigmoid(model(ng, lg, sc, mask)).numpy()
            for b, i in enumerate(ids):
                t = seqs[i][0].shape[0]
                out[i] = p[b, :t].tolist()
    return out


def gru_oof(rows, idfold, verbose=True, t0=None, n_seeds=1):
    """Leak-free OOF BiGRU per-token probs (each row scored by a GRU that never saw its
    fold; sequences featurized with the row's OWN fold-out lexicon).  n_seeds>1 averages
    independent-seed GRUs per fold (variance reduction on small data -- no extra tuning)."""
    if t0 is None:
        t0 = time.time()
    seq_probs = {}
    for k in range(5):
        tr_rows = [R for R in rows if R["fold"] != k]
        va_rows = [R for R in rows if R["fold"] == k]
        lex_k = pipeline.build_lexicon(tr_rows)
        seqs_k = build_seqs(tr_rows + va_rows, lex_k)
        ypos = sum(int(seqs_k[R["id"]][3].sum()) for R in tr_rows)
        yall = sum(seqs_k[R["id"]][3].shape[0] for R in tr_rows)
        pos_w = max(1.0, (yall - ypos) / max(ypos, 1)) ** 0.5
        acc = None
        for s in range(n_seeds):
            pred = train_predict([R["id"] for R in tr_rows], [R["id"] for R in va_rows],
                                 seqs_k, pos_w, seed=s)
            if acc is None:
                acc = {i: np.asarray(v) for i, v in pred.items()}
            else:
                for i, v in pred.items():
                    acc[i] += np.asarray(v)
        for i in acc:
            seq_probs[i] = (acc[i] / n_seeds).tolist()
        if verbose:
            print(f"[GRU fold {k}] pos_w={pos_w:.2f} seeds={n_seeds} ({time.time()-t0:.0f}s)", flush=True)
    return seq_probs


def main():
    t0 = time.time()
    P = p1_base.prepare(verbose=True)
    train = P["train"]; rows = P["rows"]; idfold = P["idfold"]; gbi = P["gbi"]
    shared = P["row_proba"]; trs = P["trs"]; stf = P["stf"]; gate_scores = P["gate_scores"]
    rbi = P["rows_by_id"]
    print(f"[prepare {time.time()-t0:.0f}s]", flush=True)

    # ---- per-fold lexicon (leak-free) + sequences (features frozen with fold lex) ----
    # sequences must be built with the row's OWN fold-out lexicon -> build per fold slice.
    seq_probs = {}
    for k in range(5):
        tr_rows = [R for R in rows if R["fold"] != k]
        va_rows = [R for R in rows if R["fold"] == k]
        lex_k = pipeline.build_lexicon(tr_rows)
        seqs_k = build_seqs(tr_rows + va_rows, lex_k)   # both featurized with fold-out lex
        # class imbalance pos_weight from train tokens
        ypos = sum(int(seqs_k[R["id"]][3].sum()) for R in tr_rows)
        yall = sum(seqs_k[R["id"]][3].shape[0] for R in tr_rows)
        pos_w = max(1.0, (yall - ypos) / max(ypos, 1)) ** 0.5
        pred = train_predict([R["id"] for R in tr_rows], [R["id"] for R in va_rows], seqs_k, pos_w)
        seq_probs.update(pred)
        print(f"[GRU fold {k}] pos_w={pos_w:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- per-language token PR-AUC (shared vs GRU vs ensemble) ----
    print("\n================ TOKEN PR-AUC per language ================")
    ap = {}
    for L in LANGS:
        rL = [R for R in rows if R["lang"] == L]
        y = np.concatenate([np.asarray(R["y"]) for R in rL])
        sh = np.concatenate([np.asarray(shared[R["id"]]) for R in rL])
        gr = np.concatenate([np.asarray(seq_probs[R["id"]]) for R in rL])
        a_sh = average_precision_score(y, sh); a_gr = average_precision_score(y, gr)
        best_e = max((0.2, 0.35, 0.5), key=lambda a: average_precision_score(y, (1 - a) * sh + a * gr))
        a_en = average_precision_score(y, (1 - best_e) * sh + best_e * gr)
        ap[L] = (a_sh, a_gr, a_en, best_e)
        print(f"  {L}: shared={a_sh:.4f}  GRU={a_gr:.4f}  ensemble(a={best_e})={a_en:.4f}")

    # ---- downstream: ensemble prob = (1-a)*shared + a*seq, tune a nested per language ----
    def ens(ids, a):
        return {i: ((1 - a) * np.asarray(shared[i]) + a * np.asarray(seq_probs[i])).tolist() for i in ids}

    # de/en: rebuild base_cache with ensembled probs at a grid of a, select thr nested per fold
    A_GRID = [0.0, 0.15, 0.3, 0.45, 0.6]
    rbl = {L: [R for R in rows if R["lang"] == L] for L in LANGS}
    truth = {R["id"]: R["truth"] for R in rows}

    # baseline de/en/it from shared (a=0) for reference
    def de_en_it_score(seq_a_de, seq_a_en, it_probs, it_gate=0.8):
        # de/en edits via base_cache at ensembled probs; it via assemble at it_probs
        probs = {}
        for R in rows:
            if R["lang"] == "de":
                probs[R["id"]] = ((1 - seq_a_de) * np.asarray(shared[R["id"]]) + seq_a_de * np.asarray(seq_probs[R["id"]])).tolist()
            elif R["lang"] == "en":
                probs[R["id"]] = ((1 - seq_a_en) * np.asarray(shared[R["id"]]) + seq_a_en * np.asarray(seq_probs[R["id"]])).tolist()
            else:
                probs[R["id"]] = it_probs[R["id"]]
        return probs

    # Precompute de/en base caches per a (ensembled probs) once (de/en rows only)
    print("\n[building de/en base caches per ensemble weight...]", flush=True)
    deen_rows = [R for R in rows if R["lang"] in ("de", "en")]
    cache_by_a = {}
    for a in A_GRID:
        rp = ens([R["id"] for R in deen_rows], a)
        cache_by_a[a] = base_cache(deen_rows, idfold, rp, trs, stf)
    print(f"[de/en caches {time.time()-t0:.0f}s]", flush=True)

    # nested de/en threshold+weight selection: fold-k picks (a,thr) per lang on other 4
    def lang_edits_for(L, a, thr):
        return {R["id"]: cache_by_a[a][R["id"]][thr] for R in rbl[L]}

    def lang_score_sub(L, a, thr, ids):
        e = {i: cache_by_a[a][i][thr] for i in ids}
        _s, d = elru.elru(e, {i: truth[i] for i in ids}, {i: L for i in ids}, detail=True)
        return d[L]["lang_score"]

    deen_nested = {}; deen_nn = {}
    for L in ("de", "en"):
        allids = set(R["id"] for R in rbl[L])
        best_nn = max(((a, thr) for a in A_GRID for thr in pipeline.GRID),
                      key=lambda at: lang_score_sub(L, at[0], at[1], allids))
        deen_nn[L] = best_nn
        by_fold = {}
        for k in range(5):
            other = set(R["id"] for R in rbl[L] if R["fold"] != k)
            by_fold[k] = max(((a, thr) for a in A_GRID for thr in pipeline.GRID),
                             key=lambda at: lang_score_sub(L, at[0], at[1], other))
        deen_nested[L] = by_fold

    # assemble nested de/en edits
    ne_edits = {}
    for L in ("de", "en"):
        for k in range(5):
            a, thr = deen_nested[L][k]
            for R in rbl[L]:
                if R["fold"] == k:
                    ne_edits[R["id"]] = cache_by_a[a][R["id"]][thr]
    nn_edits = {}
    for L in ("de", "en"):
        a, thr = deen_nn[L]
        for R in rbl[L]:
            nn_edits[R["id"]] = cache_by_a[a][R["id"]][thr]

    # it: ensemble prob feeding boost (Lever-1 winning integration), a tuned nested
    itrows = rbl["it"]
    truth_it = {R["id"]: R["truth"] for R in itrows}
    def it_assemble(probs):
        out = {}
        for R in itrows:
            k = idfold[R["id"]]
            gs = [(ab[0], ab[1], p) for (ab, p) in gate_scores[R["id"]]]
            out[R["id"]] = n1_assemble_it(R["tk"], R["text"], probs[R["id"]], 0.8, gs, trs[k], stf[k])
        return out
    def it_boost(a, w):
        out = {}
        for R in itrows:
            sh = np.asarray(shared[R["id"]]); gr = np.asarray(seq_probs[R["id"]])
            en_ = (1 - a) * sh + a * gr
            out[R["id"]] = np.clip(sh + w * np.clip(en_ - 0.3, 0, None), 0, 1).tolist()
        return out
    IT_A = [0.0, 0.2, 0.35, 0.5]; IT_W = [0.0, 0.3, 0.5, 0.7]
    it_cache = {}
    for a in IT_A:
        for w in IT_W:
            it_cache[(a, w)] = it_assemble(it_boost(a, w))
    def it_sub_score(a, w, ids):
        e = {i: it_cache[(a, w)][i] for i in ids}
        _s, d = elru.elru(e, {i: truth_it[i] for i in ids}, {i: "it" for i in ids}, detail=True)
        return d["it"]["lang_score"]
    it_allids = set(truth_it)
    it_nn = max(((a, w) for a in IT_A for w in IT_W), key=lambda aw: it_sub_score(aw[0], aw[1], it_allids))
    it_ne_edits = {}; it_nby = {}
    for k in range(5):
        other = set(R["id"] for R in itrows if R["fold"] != k)
        b = max(((a, w) for a in IT_A for w in IT_W), key=lambda aw: it_sub_score(aw[0], aw[1], other))
        it_nby[k] = b
        for R in itrows:
            if idfold[R["id"]] == k:
                it_ne_edits[R["id"]] = it_cache[b][R["id"]]
    it_nn_edits = {i: it_cache[it_nn][i] for i in truth_it}

    # ---- combine + group-vote de/en; score nested + non-nested ----
    def combine_and_vote(deen_map, it_map):
        m = {}
        for R in rows:
            m[R["id"]] = it_map[R["id"]] if R["lang"] == "it" else deen_map[R["id"]]
        return group_consistency(m, rbi, gbi, trs, stf, idfold,
                                 vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    ne = combine_and_vote(ne_edits, it_ne_edits)
    nn = combine_and_vote(nn_edits, it_nn_edits)
    ne_s, ne_d = score_edits(rows, ne); nn_s, nn_d = score_edits(rows, nn)
    fp = fp_counts(rows, nn)

    print("\n================ LEVER 2 (BiGRU ensemble) RESULT ================")
    print(f"  de/en nested (a,thr) by-fold: de={deen_nested['de']}  en={deen_nested['en']}")
    print(f"  it nested (a,w) by-fold: {it_nby}   nonnested={it_nn}")
    print(f"  NON-NESTED overall={nn_s:.4f}  " + " ".join(f"{L}={nn_d[L]['lang_score']:.4f}" for L in LANGS))
    print(f"  NESTED    overall={ne_s:.4f}  " + " ".join(f"{L}={ne_d[L]['lang_score']:.4f}" for L in LANGS))
    print(f"  base N3 nested 0.5503 (de .4237 en .8067 it .4205)  delta={ne_s-0.5503:+.4f}")
    print("  unchanged FP: " + ", ".join(f"{L}={fp[L][0]}/{fp[L][1]}" for L in LANGS))
    print(f"[lever2 {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
