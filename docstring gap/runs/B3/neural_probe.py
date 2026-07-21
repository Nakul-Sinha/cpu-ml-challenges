"""PROBE 2 (Agent B3): neural span-classification model.

Label = target_span if it is among the top-K most frequent spans in fold-train
(buckets 1-19), else OTHER. A genuinely-trained neural model (hashed linear /
EmbeddingBag softmax over K+1 classes) predicts the span class from context.

Features (hashed, count-based -> compliant, NO idf/tfidf/bm25):
  L1/L2 (two left context words), R1/R2 (two right context words),
  the L1_R1 anchor pair, this row's split code identifiers (bag), gap-position bucket.

Model: nn.EmbeddingBag(n_hash, n_classes, mode="sum") + per-class bias == a hashed
multinomial-logistic model. Trained with CrossEntropyLoss over 2-3 epochs, 5 threads.

Reports: top-1 chrF when it commits (argmax != OTHER) + coverage; full-set chrF with
fallback; top-10 oracle (candidate-generation value); and whether P(class) is a good
reranker feature (chrF stratified by confidence). Fit ONLY on buckets 1-19.
"""
import sys, re, time, math, zlib, hashlib, collections, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, "solution")
from chrf import f_pooled, score_lists

GAP = "[GAP]"
_tok_re = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
_ident_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(s):
    return _tok_re.findall(s)


def bucket(s):
    return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 20


def split_ctx(masked):
    i = masked.find(GAP)
    return tokenize(masked[:i]), tokenize(masked[i + len(GAP):])


def code_words(code):
    out, seen = [], set()
    for m in _ident_re.finditer(code):
        ident = m.group(0)
        if ident in ("self", "cls"):
            continue
        for p in re.split(r"_+", ident):
            for t in (re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", p) or [p]):
                tl = t.lower()
                if len(tl) >= 2 and tl not in seen:
                    seen.add(tl)
                    out.append(tl)
        if len(out) > 30:
            break
    return out[:30]


def feat_tokens(masked, code):
    """Build the list of (string) feature tokens for one row."""
    left, right = split_ctx(masked)
    l1 = left[-1] if left else "<none>"
    l2 = left[-2] if len(left) >= 2 else "<none>"
    r1 = right[0] if right else "<none>"
    r2 = right[1] if len(right) >= 2 else "<none>"
    nl = len(left)
    total = nl + len(right) + 1
    posb = int(5 * nl / max(total, 1))
    toks = [f"L1={l1}", f"L2={l2}", f"R1={r1}", f"R2={r2}",
            f"LR={l1}|{r1}", f"L1R2={l1}|{r2}", f"L2R1={l2}|{r1}",
            f"POS={posb}", f"NL={min(nl,8)}"]
    toks += [f"C={w}" for w in code_words(code)]
    # a couple of code-anchored crosses (does a code word sit right at the gap?)
    toks += [f"CR1={w}|{r1}" for w in code_words(code)[:6]]
    return toks


def hash_row(masked, code, nfeat):
    idx = set()
    for t in feat_tokens(masked, code):
        idx.add(zlib.crc32(t.encode("utf-8")) % nfeat)
    return sorted(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kcls", type=int, default=5000)
    ap.add_argument("--nfeat", type=int, default=1 << 17)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--conf", type=float, default=0.30, help="commit threshold")
    ap.add_argument("--out", default="runs/B3/nn_val.csv")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.set_num_threads(5)
    torch.manual_seed(0)

    t0 = time.time()
    train = pd.read_csv("dataset/train.csv", keep_default_na=False)
    b = train.masked_docstring.map(bucket)
    trn = train[b != 0].reset_index(drop=True)
    val = train[b == 0].reset_index(drop=True)
    print(f"loaded trn={len(trn)} val={len(val)}  {time.time()-t0:.1f}s", flush=True)

    # ---- class map: top-K spans in fold-train + OTHER ----
    span_counts = collections.Counter(trn.target_span.astype(str))
    top_spans = [s for s, _ in span_counts.most_common(args.kcls)]
    cls_of = {s: i for i, s in enumerate(top_spans)}      # 0..K-1
    OTHER = args.kcls                                       # class K
    inv = top_spans + ["<OTHER>"]
    ncls = args.kcls + 1
    covered = sum(span_counts[s] for s in top_spans) / len(trn)
    print(f"top-{args.kcls} spans cover {covered:.3f} of trn rows; ncls={ncls}", flush=True)

    def labels(df):
        return np.array([cls_of.get(str(s), OTHER) for s in df.target_span], dtype=np.int64)

    y_trn = labels(trn)
    y_val = labels(val)

    # ---- featurize (hashed index lists) ----
    def featurize(df):
        flat, off = [], [0]
        for masked, code in zip(df.masked_docstring.values, df.code_context.values):
            ix = hash_row(masked, code, args.nfeat)
            flat.extend(ix)
            off.append(len(flat))
        return np.array(flat, dtype=np.int64), np.array(off, dtype=np.int64)

    t1 = time.time()
    Xf_tr, Xo_tr = featurize(trn)
    Xf_va, Xo_va = featurize(val)
    print(f"featurized trn+val  {time.time()-t1:.1f}s  (avg {len(Xf_tr)/len(trn):.1f} feats/row)", flush=True)

    # ---- model: hashed EmbeddingBag softmax (== multinomial logistic) ----
    class HashLin(nn.Module):
        def __init__(self, nf, nc):
            super().__init__()
            self.emb = nn.EmbeddingBag(nf, nc, mode="sum", sparse=True)
            nn.init.zeros_(self.emb.weight)
            self.bias = nn.Parameter(torch.zeros(nc))
        def forward(self, idx, off):
            return self.emb(idx, off) + self.bias

    model = HashLin(args.nfeat, ncls)
    # init bias to class log-prior for a calibrated start
    prior = np.bincount(y_trn, minlength=ncls).astype(np.float64) + 1.0
    with torch.no_grad():
        model.bias.copy_(torch.tensor(np.log(prior / prior.sum()), dtype=torch.float32))
    opt_emb = torch.optim.SparseAdam(model.emb.parameters(), lr=args.lr)
    opt_b = torch.optim.Adam([model.bias], lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    # precompute per-row feature arrays ONCE (fast vectorized minibatch assembly)
    rf_tr = np.split(Xf_tr, Xo_tr[1:-1]) if len(trn) else []
    rf_va = np.split(Xf_va, Xo_va[1:-1]) if len(val) else []
    rl_tr = (Xo_tr[1:] - Xo_tr[:-1]).astype(np.int64)
    rl_va = (Xo_va[1:] - Xo_va[:-1]).astype(np.int64)

    def make_batch(rf, rl, R):
        idx = torch.from_numpy(np.concatenate([rf[r] for r in R]))
        off = torch.from_numpy(np.concatenate(([0], np.cumsum(rl[R])[:-1])).astype(np.int64))
        return idx, off

    refs = val.target_span.astype(str).tolist()

    def evaluate():
        model.eval()
        all_top1, all_conf, all_top10 = [], [], []
        with torch.no_grad():
            for s in range(0, len(val), 8192):
                R = np.arange(s, min(s + 8192, len(val)))
                idx, off = make_batch(rf_va, rl_va, R)
                logits = model(idx, off)
                prob = torch.softmax(logits, dim=1)
                p, c = prob.max(dim=1)
                all_top1.extend(c.tolist()); all_conf.extend(p.tolist())
                logits[:, OTHER] = -1e9
                all_top10.extend(torch.topk(logits, 10, dim=1).indices.tolist())
        commit_pred, commit_ref, full_pred, chrf_top1_all = [], [], [], []
        for i, (cls, conf) in enumerate(zip(all_top1, all_conf)):
            if cls != OTHER and conf >= args.conf:
                full_pred.append(inv[cls]); commit_pred.append(inv[cls]); commit_ref.append(refs[i])
            else:
                full_pred.append("the")
            span1 = inv[cls] if cls != OTHER else inv[all_top10[i][0]]
            chrf_top1_all.append(f_pooled(span1, refs[i]))
        cov = len(commit_pred) / len(val)
        cchrf = score_lists(commit_pred, commit_ref) if commit_pred else 0.0
        fchrf = score_lists(full_pred, refs)
        orc = sum(max(f_pooled(inv[c], refs[i]) for c in all_top10[i]) for i in range(len(val))) / len(val)
        conf_arr = np.array(all_conf); chrf_arr = np.array(chrf_top1_all)
        corr = float(np.corrcoef(conf_arr, chrf_arr)[0, 1])
        return dict(cov=cov, cchrf=cchrf, fchrf=fchrf, orc=orc, corr=corr,
                    top1=all_top1, conf=all_conf, top10=all_top10,
                    conf_arr=conf_arr, chrf_arr=chrf_arr)

    n = len(trn)
    t2 = time.time()
    best = None
    for ep in range(args.epochs):
        model.train()
        perm = np.random.RandomState(ep).permutation(n)
        tot = 0.0
        for s in range(0, n, args.bs):
            R = perm[s:s + args.bs]
            idx, off = make_batch(rf_tr, rl_tr, R)
            yb = torch.from_numpy(y_trn[R])
            loss = lossf(model(idx, off), yb)
            opt_emb.zero_grad(); opt_b.zero_grad()
            loss.backward()
            opt_emb.step(); opt_b.step()
            tot += loss.item() * len(R)
        ev = evaluate()
        print(f"  epoch {ep}: loss={tot/n:.4f}  cov@{args.conf}={ev['cov']:.3f} "
              f"commit_chrF={ev['cchrf']:.4f} full_chrF={ev['fchrf']:.4f} "
              f"oracle@10={ev['orc']:.4f} conf_corr={ev['corr']:+.3f}  {time.time()-t2:.1f}s", flush=True)
        if best is None or ev['fchrf'] >= best['fchrf']:
            best = ev; best['ep'] = ep

    ev = best
    # confidence-stratified chrF at the BEST epoch (reranker-feature diagnostic)
    conf_arr, chrf_arr = ev['conf_arr'], ev['chrf_arr']
    qs = np.quantile(conf_arr, [0.2, 0.4, 0.6, 0.8])
    edges = [-1] + list(qs) + [2]
    print(f"\n=== PROBE 2 results (best epoch {ev['ep']}) ===", flush=True)
    print(f"coverage(commit@{args.conf})={ev['cov']:.3f}  committed_chrF={ev['cchrf']:.4f}", flush=True)
    print(f"full_chrF(fallback 'the')={ev['fchrf']:.4f}  oracle@10={ev['orc']:.4f}", flush=True)
    print(f"P(class) vs top1-chrF: corr={ev['corr']:+.3f}", flush=True)
    print("  confidence-stratified top1 chrF (meanconf, chrF, n):", flush=True)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf_arr > lo) & (conf_arr <= hi)
        if m.any():
            print(f"    conf~{conf_arr[m].mean():.3f}  chrF={chrf_arr[m].mean():.4f}  n={int(m.sum())}", flush=True)

    pd.DataFrame({
        "id": val.id.values, "target_span": refs,
        "nn_pred": [inv[c] for c in ev['top1']],
        "nn_conf": ev['conf'],
        "nn_is_other": [int(c == OTHER) for c in ev['top1']],
        "nn_top10": ["\t".join(inv[c] for c in row) for row in ev['top10']],
    }).to_csv(args.out, index=False)

    print(json.dumps({
        "kcls": args.kcls, "ncls": ncls, "trn_cover": round(covered, 3),
        "best_epoch": ev['ep'],
        "coverage_commit": round(ev['cov'], 3),
        "committed_chrF": round(ev['cchrf'], 4),
        "full_chrF": round(ev['fchrf'], 4),
        "oracle_top10_chrF": round(ev['orc'], 4),
        "conf_chrf_corr": round(ev['corr'], 3),
        "total_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
