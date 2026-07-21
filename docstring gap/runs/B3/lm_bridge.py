"""PROBE 1 (Agent B3): word n-gram LM bridge generator.

Train a word-level trigram LM (stupid-backoff, a learned statistical model) over
reconstructed fold-train docstring sentences (buckets 1-19). For a masked
sentence `left [GAP] right`, beam-search 1-4-word bridges maximizing the LM score
of `left + bridge + right`, scoring the join BOTH entering the bridge (left->b1)
and leaving it (bk->right, up to 2 right tokens = trigram re-entry).

Beam-expansion vocabulary per step = LM successors of the current bigram/unigram
context  U  this row's split code identifiers  U  globally frequent target words.

Compliance: pure count-based conditional probabilities + n-gram LM. No idf/tfidf/bm25.
Fit ONLY on buckets 1-19; report on bucket-0 holdout via solution/chrf.py.
"""
import sys, re, time, math, hashlib, collections, json, argparse
import pandas as pd

sys.path.insert(0, "solution")
from chrf import f_pooled, score_lists

GAP = "[GAP]"
BOS = "<s>"
LOGA = math.log(0.4)          # stupid-backoff discount
_tok_re = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
_ident_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(s):
    return _tok_re.findall(s)


def bucket(s):
    return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 20


def split_ctx(masked):
    """Return (left_tokens, right_tokens) around the [GAP] marker."""
    i = masked.find(GAP)
    return tokenize(masked[:i]), tokenize(masked[i + len(GAP):])


def code_words(code):
    """Split identifiers (snake + camel) from a code_context into candidate words."""
    out = []
    seen = set()
    for m in _ident_re.finditer(code):
        ident = m.group(0)
        if ident in ("self", "cls"):
            continue
        # snake + camelCase split
        parts = re.split(r"_+", ident)
        toks = []
        for p in parts:
            toks += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", p) or [p]
        for t in toks:
            tl = t.lower()
            if len(tl) >= 2 and tl not in seen:
                seen.add(tl)
                out.append(tl)
        if len(out) > 40:
            break
    return out[:40]


class LMBridge:
    def __init__(self, order=3, topsucc=12, glob_target=40):
        self.order = order
        self.topsucc = topsucc
        self.glob_target = glob_target

    def fit(self, trn):
        c1 = collections.Counter()
        c2 = collections.Counter()
        c3 = collections.Counter()
        tw = collections.Counter()   # words appearing in target spans
        for masked, tgt in zip(trn.masked_docstring.values, trn.target_span.values):
            full = masked.replace(GAP, str(tgt))
            toks = [BOS, BOS] + tokenize(full) + [BOS]
            for w in toks:
                c1[w] += 1
            for i in range(len(toks) - 1):
                c2[(toks[i], toks[i + 1])] += 1
            for i in range(len(toks) - 2):
                c3[(toks[i], toks[i + 1], toks[i + 2])] += 1
            for w in tokenize(str(tgt)):
                if any(ch.isalnum() for ch in w):
                    tw[w] += 1
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.total1 = sum(c1.values())
        self.V = len(c1)
        # successor tables for beam expansion
        succ2 = collections.defaultdict(list)
        for (a, b, c), n in c3.items():
            succ2[(a, b)].append((n, c))
        self.succ2 = {k: [w for _, w in sorted(v, reverse=True)[:self.topsucc]]
                      for k, v in succ2.items()}
        succ1 = collections.defaultdict(list)
        for (a, b), n in c2.items():
            succ1[a].append((n, b))
        self.succ1 = {k: [w for _, w in sorted(v, reverse=True)[:self.topsucc]]
                      for k, v in succ1.items()}
        self.glob = [w for w, _ in tw.most_common(self.glob_target)]
        return self

    def sb(self, a, b, w):
        """log stupid-backoff score of P(w | a b)."""
        c3v = self.c3.get((a, b, w))
        if c3v:
            return math.log(c3v / self.c2[(a, b)])
        c2v = self.c2.get((b, w))
        if c2v:
            return LOGA + math.log(c2v / self.c1[b])
        c1v = self.c1.get(w, 0)
        return 2 * LOGA + math.log((c1v + 0.5) / (self.total1 + 0.5 * self.V))

    def _cands(self, a, b, cw):
        c = self.succ2.get((a, b))
        if c is None:
            c = self.succ1.get(b, [])
        seen = {BOS}
        res = []
        for w in c:
            if w not in seen and any(ch.isalnum() for ch in w):
                seen.add(w)
                res.append(w)
        for w in cw[:12]:
            if w not in seen:
                seen.add(w)
                res.append(w)
        for w in self.glob[:20]:
            if w not in seen:
                seen.add(w)
                res.append(w)
        return res

    def leaving(self, prefix, right):
        """Score first up-to-2 right tokens given prefix tail (trigram re-entry)."""
        if not right:
            return 0.0
        seq = prefix + right
        s = 0.0
        for i in range(len(prefix), min(len(prefix) + 2, len(seq))):
            a = seq[i - 2] if i >= 2 else BOS
            b = seq[i - 1] if i >= 1 else BOS
            s += self.sb(a, b, seq[i])
        return s

    def pool(self, left, right, cw, beam=16, maxlen=4, prune_wb=1.5, keep=24):
        """Beam-search once; return a pool of scored bridges (components kept so a
        word_bonus can be applied later without re-beaming).
        Returns list of (raw_entering_lm, leaving_lm, length, text) deduped."""
        fleft = [BOS, BOS] + left
        completed = []
        beams = [(0.0, [])]           # (raw_entering_lm, bridge)
        for _ in range(maxlen):
            newb = []
            for esc, bridge in beams:
                seq = fleft + bridge
                a, b = seq[-2], seq[-1]
                for w in self._cands(a, b, cw):
                    nesc = esc + self.sb(a, b, w)
                    nbridge = bridge + [w]
                    lv = self.leaving(fleft + nbridge, right)
                    completed.append((nesc, lv, len(nbridge), " ".join(nbridge)))
                    newb.append((nesc + prune_wb * len(nbridge), nesc, nbridge))
            newb.sort(key=lambda x: x[0], reverse=True)
            beams = [(e, br) for _, e, br in newb[:beam]]
        completed.sort(key=lambda x: x[0] + prune_wb * x[2] + x[1], reverse=True)
        seen = set()
        out = []
        for e, lv, ln, txt in completed:
            if txt in seen:
                continue
            seen.add(txt)
            out.append((e, lv, ln, txt))
            if len(out) >= keep:
                break
        return out

    @staticmethod
    def rank(pool, word_bonus, topn=10):
        scored = sorted(pool, key=lambda x: x[0] + word_bonus * x[2] + x[1],
                        reverse=True)
        return [t[3] for t in scored[:topn]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nval", type=int, default=5000)
    ap.add_argument("--full", action="store_true", help="score full bucket-0")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--out", default="runs/B3/lm_val.csv")
    args = ap.parse_args()

    t0 = time.time()
    train = pd.read_csv("dataset/train.csv", keep_default_na=False)
    b = train.masked_docstring.map(bucket)
    trn = train[b != 0]
    val_all = train[b == 0]
    print(f"loaded trn={len(trn)} val_all={len(val_all)}  {time.time()-t0:.1f}s", flush=True)

    lm = LMBridge().fit(trn)
    print(f"LM fit: V={lm.V} tri={len(lm.c3)} succ2={len(lm.succ2)}  {time.time()-t0:.1f}s", flush=True)

    if args.full:
        val = val_all
    else:
        val = val_all.sample(min(args.nval, len(val_all)), random_state=1)
    refs = val.target_span.astype(str).tolist()

    # ---- one beam pass: build a scored pool per row ----
    t1 = time.time()
    pools = []
    for masked, code in zip(val.masked_docstring.values, val.code_context.values):
        l, r = split_ctx(masked)
        pools.append(lm.pool(l, r, code_words(code), beam=args.beam))
    dt = time.time() - t1
    print(f"pools built n={len(val)}  {dt:.1f}s  ({1000*dt/len(val):.1f}s/1k)", flush=True)

    wbs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] if args.sweep else [args.wb]
    best = (-1.0, args.wb, None)
    for wb in wbs:
        top1 = [LMBridge.rank(p, wb, topn=1)[0] if p else "the" for p in pools]
        s1 = score_lists(top1, refs)
        print(f"  wb={wb:.2f}  argmax chrF={s1:.4f}", flush=True)
        if s1 > best[0]:
            best = (s1, wb, top1)
    s1, wb_used, top1 = best

    # oracle@10 at the chosen word_bonus
    oracle = []
    for p, ref in zip(pools, refs):
        brs = LMBridge.rank(p, wb_used, topn=10) if p else ["the"]
        oracle.append(max((f_pooled(x, ref) for x in brs), default=0.0))
    so = sum(oracle) / len(oracle)
    print(f"CHOSEN wb={wb_used}  argmax chrF={s1:.4f}  oracle@10={so:.4f}", flush=True)

    # save top-10 pool per row for downstream rerankers + the argmax prediction
    rows = []
    for i, (rid, p) in enumerate(zip(val.id.values, pools)):
        brs = LMBridge.rank(p, wb_used, topn=10) if p else ["the"]
        rows.append({"id": rid, "prediction": top1[i], "target_span": refs[i],
                     "cands": "\t".join(brs)})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(json.dumps({"n": len(val), "wb": wb_used, "argmax_chrF": round(s1, 4),
                      "oracle_top10_chrF": round(so, 4),
                      "pool_s_per_1k": round(1000 * dt / len(val), 1),
                      "total_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
