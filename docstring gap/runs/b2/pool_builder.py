"""pool_builder.py  -- Agent B2: candidate-pool recall for Docstring Gap Restoration.

Builds a <=~80-candidate pool per row maximizing ORACLE chrF, from four source
families, all fit ONLY on fold-train (buckets 1-19 for validation reporting):

  A) Anchored-context indexes  (exact + skip-gram + normalized keys, larger top-K)
  B) Fuzzy full-sentence NN     (HashingVectorizer char_wb 4-5, l2, NO idf;
                                 chunked sparse cosine top-k; GATED to weak-anchor rows)
  C) Code-derived candidates    (func-name split, arg names, return/attr tokens;
                                 filtered/scored by LEARNED target-prior)
  D) Global frequent spans      (top target strings)

API:
  pb = PoolBuilder(...); pb.fit(train_df)
  pb.candidates(row)         -> list[(text, source, score)]           # single row
  pb.candidates_batch(df)    -> list[list[(text, source, score)]]     # efficient

Compliance: term-frequency / count-conditional-probability only. No idf, no BM25,
no tf-idf weighting. HashingVectorizer uses alternate_sign=False, norm='l2', raw
hashed term frequencies. Nothing is fit on test.
"""
import re, os, time
from collections import defaultdict, Counter
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from concurrent.futures import ThreadPoolExecutor

GAP = "[GAP]"
_word = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ident = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _norm_tok(t):
    return t.strip(".,;:!?\"'()[]{}`").lower()


def split_ident(name):
    """snake_case + camelCase -> lowercase words."""
    parts = []
    for chunk in name.split("_"):
        if not chunk:
            continue
        parts += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", chunk)
    return [p.lower() for p in parts if p]


def window_text(masked, w=4):
    """+/-w word window around [GAP] (local-context fuzzy view)."""
    i = masked.find(GAP)
    if i < 0:
        return masked
    L = masked[:i].split()[-w:]
    R = masked[i + len(GAP):].split()[:w]
    return " ".join(L) + " " + " ".join(R)


def anchors(masked):
    """Return dict of anchor-key tuples around [GAP]. Keys are used across indexes."""
    i = masked.find(GAP)
    if i < 0:
        L, R = masked.split(), []
    else:
        L = masked[:i].split()
        R = masked[i + len(GAP):].split()
    l1 = L[-1] if L else ""
    l2a = L[-2] if len(L) >= 2 else ""   # second-from-left
    r1 = R[0] if R else ""
    r2b = R[1] if len(R) >= 2 else ""    # second-from-right
    ln1 = _norm_tok(l1)
    rn1 = _norm_tok(r1)
    return {
        "l2r2": (" ".join(L[-2:]), " ".join(R[:2])),
        "l1r1": (l1, r1),
        "l1": (l1,),
        "r1": (r1,),
        "l1r1n": (ln1, rn1),          # normalized (lower, punct-stripped)
        "skipR": (l1, r2b),           # first-left + second-right (skip first right)
        "skipL": (l2a, r1),           # second-left + first-right
    }


# source priority tiers (higher = added first, survives the 80-cap). Ordered by
# measured marginal oracle: l2r2/l1r1 (precise) > learned-code > skip/fuzz > r1/l1
# > unseen-code > global. l1r1n dropped (marginal +0.0005, wasted a cap slot).
SRC_TIER = {
    "l2r2": 10, "l1r1": 9, "codeP": 8, "skipR": 7, "skipL": 7,
    "fuzz": 6, "fuzzw": 6, "r1": 5, "l1": 5, "code": 2, "global": 1,
}
ANCHOR_KEYS = ["l2r2", "l1r1", "skipR", "skipL", "r1", "l1"]


class PoolBuilder:
    def __init__(self, topk_anchor=12, topk_fuzz=20, cap=80,
                 fuzz_ngram=(4, 5), fuzz_feats=2 ** 18, n_threads=5,
                 gate_fuzz=False, use_window=True, topk_fuzzw=12, n_global=50):
        self.topk_anchor = topk_anchor
        self.topk_fuzz = topk_fuzz
        self.topk_fuzzw = topk_fuzzw
        self.cap = cap
        self.fuzz_ngram = fuzz_ngram
        self.fuzz_feats = fuzz_feats
        self.n_threads = n_threads
        self.gate_fuzz = gate_fuzz
        self.use_window = use_window
        self.n_global = n_global

    # ---------------- fit ----------------
    def fit(self, train_df):
        t0 = time.time()
        masked = train_df.masked_docstring.values
        tgts = train_df.target_span.astype(str).values
        codes = train_df.code_context.values

        # A) anchored indexes: key -> Counter(target) pruned to top-K
        idx = {k: defaultdict(Counter) for k in ANCHOR_KEYS}
        for m, tg in zip(masked, tgts):
            a = anchors(m)
            for k in ANCHOR_KEYS:
                key = a[k]
                if key[0] == "" and (len(key) == 1 or key[-1] == ""):
                    continue  # empty anchor -> useless key
                idx[k][key][tg] += 1
        self.idx = {}
        for k, d in idx.items():
            self.idx[k] = {key: c.most_common(self.topk_anchor) for key, c in d.items()}
        self.t_anchor_fit = time.time() - t0

        # D) global frequent spans + learned target prior (for code filtering)
        t1 = time.time()
        gc = Counter(tgts)
        self.global_top = [t for t, _ in gc.most_common(self.n_global)]
        self.n_train = len(tgts)
        self.target_prior = gc  # count of each string as a target (learned prior)

        # C) learn which code-derived token strings ever appear as targets, w/ prior
        # (nothing hand-picked: a code candidate is kept only if it occurred as a
        #  target in fold-train, scored by that frequency)
        self.t_global_fit = time.time() - t1

        # B) fuzzy index: fit HashingVectorizer, transform train (full sentence)
        t2 = time.time()
        self.hv = HashingVectorizer(analyzer="char_wb", ngram_range=self.fuzz_ngram,
                                    n_features=self.fuzz_feats, lowercase=True,
                                    alternate_sign=False, norm="l2")
        self.Xtr = self.hv.transform(masked)
        self.XtrT = self.Xtr.T.tocsr()
        self.train_targets = tgts
        if self.use_window:
            wtxt = [window_text(m) for m in masked]
            self.hvw = HashingVectorizer(analyzer="char_wb", ngram_range=(4, 5),
                                         n_features=self.fuzz_feats, lowercase=True,
                                         alternate_sign=False, norm="l2")
            self.XtrwT = self.hvw.transform(wtxt).T.tocsr()
        self.t_fuzz_fit = time.time() - t2
        self.t_fit = time.time() - t0
        return self

    # ---------------- code-derived candidates ----------------
    def _code_cands(self, code):
        out = []
        m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", code)
        if m:
            name = m.group(1)
            words = split_ident(name)
            if words:
                out.append(" ".join(words))
                out += words
                if len(words) >= 2:
                    out.append(" ".join(words[:2]))
                    out.append(" ".join(words[-2:]))
            for a in m.group(2).split(","):
                a = a.split("=")[0].split(":")[0].strip().lstrip("*")
                if a and a not in ("self", "cls"):
                    out.append(a.replace("_", " "))
                    aw = split_ident(a)
                    if len(aw) >= 2:
                        out.append(" ".join(aw))
        # return-expression + attribute/call identifier tokens
        for rm in re.finditer(r"return\s+([^\n]+)", code):
            for tok in _ident.findall(rm.group(1))[:6]:
                w = split_ident(tok)
                if w:
                    out.append(" ".join(w))
        # score each by learned target-prior; keep those that appear as targets
        scored = []
        seen = set()
        for c in out:
            if c in seen:
                continue
            seen.add(c)
            pr = self.target_prior.get(c, 0)
            scored.append((c, pr))
        # keep candidates with prior>0 first (learned), then a few raw name words
        scored.sort(key=lambda x: -x[1])
        kept = [(c, pr) for c, pr in scored if pr > 0][:6]
        # always allow the func-name phrase + first words even if unseen (grounded fallback)
        extra = [(c, 0) for c, pr in scored if pr == 0][:4]
        return kept + extra

    # ---------------- fuzzy KNN (batch) ----------------
    def _fuzz_knn(self, Xq, k, XT, chunk=256):
        n = Xq.shape[0]
        if n == 0:
            return np.empty((0, k), np.int32), np.empty((0, k), np.float32)
        oi = np.zeros((n, k), np.int32)
        os_ = np.zeros((n, k), np.float32)

        def worker(rng):
            s, e = rng
            for st in range(s, e, chunk):
                en = min(st + chunk, e)
                sims = (Xq[st:en] @ XT).toarray()
                kk = min(k, sims.shape[1])
                part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
                rr = np.arange(sims.shape[0])[:, None]
                pv = sims[rr, part]
                order = np.argsort(-pv, axis=1)
                oi[st:en] = part[rr, order]
                os_[st:en] = pv[rr, order]

        if self.n_threads > 1 and n > chunk:
            bnds = np.linspace(0, n, self.n_threads + 1).astype(int)
            rngs = [(bnds[i], bnds[i + 1]) for i in range(self.n_threads) if bnds[i + 1] > bnds[i]]
            with ThreadPoolExecutor(len(rngs)) as ex:
                list(ex.map(worker, rngs))
        else:
            worker((0, n))
        return oi, os_

    # ---------------- batch candidate generation ----------------
    def candidates_batch(self, df, use_fuzz=True):
        masked = df.masked_docstring.values
        codes = df.code_context.values
        n = len(df)
        # anchored + code + global (fast, per-row)
        pools = []           # list of list[(text, src, score)]
        anchor_strength = np.zeros(n, dtype=np.int8)
        for i in range(n):
            m = masked[i]
            a = anchors(m)
            row = []
            for k in ANCHOR_KEYS:
                key = a[k]
                if key not in self.idx[k]:
                    continue
                lst = self.idx[k][key]
                tot = sum(c for _, c in lst) or 1
                for j, (t, c) in enumerate(lst):
                    row.append((t, k, SRC_TIER[k] + (c / tot)))
                if k in ("l2r2", "l1r1"):
                    anchor_strength[i] = max(anchor_strength[i], 2 if k == "l2r2" else 1)
            for c, pr in self._code_cands(codes[i]):
                # learned code (seen as a target) ranks high; speculative code low
                src = "codeP" if pr > 0 else "code"
                row.append((c, src, SRC_TIER[src] + min(pr / self.n_train, 0.5)))
            for gi, g in enumerate(self.global_top[:12]):
                row.append((g, "global", SRC_TIER["global"] - gi * 0.01))
            pools.append(row)

        # fuzzy (batch, gated to weak-anchor rows if configured)
        if use_fuzz:
            if self.gate_fuzz:
                fmask = anchor_strength < 1     # rows with no l1r1/l2r2 hit
            else:
                fmask = np.ones(n, dtype=bool)
            fidx = np.where(fmask)[0]
            if len(fidx) > 0:
                Xq = self.hv.transform(masked[fidx])
                nbr, nsim = self._fuzz_knn(Xq, self.topk_fuzz, self.XtrT)
                for r, gi in enumerate(fidx):
                    seen = set()
                    for j in range(nbr.shape[1]):
                        t = self.train_targets[nbr[r, j]]
                        if t in seen:
                            continue
                        seen.add(t)
                        pools[gi].append((t, "fuzz", SRC_TIER["fuzz"] + float(nsim[r, j])))
                if self.use_window:
                    Xw = self.hvw.transform([window_text(masked[i]) for i in fidx])
                    wbr, wsim = self._fuzz_knn(Xw, self.topk_fuzzw, self.XtrwT)
                    for r, gi in enumerate(fidx):
                        seen = set()
                        for j in range(wbr.shape[1]):
                            t = self.train_targets[wbr[r, j]]
                            if t in seen:
                                continue
                            seen.add(t)
                            pools[gi].append((t, "fuzzw", SRC_TIER["fuzzw"] + float(wsim[r, j]) * 0.9))

        # dedup (keep max score) + cap
        out = []
        for row in pools:
            best = {}
            for t, s, sc in row:
                if t not in best or sc > best[t][1]:
                    best[t] = (s, sc)
            merged = [(t, v[0], v[1]) for t, v in best.items()]
            merged.sort(key=lambda x: -x[2])
            out.append(merged[:self.cap])
        return out

    def candidates(self, row):
        import pandas as pd
        df = pd.DataFrame([{"masked_docstring": row["masked_docstring"],
                            "code_context": row["code_context"]}])
        return self.candidates_batch(df)[0]
