"""nnseq family: torch CPU joint-route sequence tagger.
Produces oof_probs.npy (3938x5) + test_probs.npy (1583x5) in canonical order.
"""
import os, sys, argparse, json, time, collections
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.expanduser('~/discourse/runs/nnseq'))
sys.path.insert(0, os.path.expanduser('~/discourse/foundation'))
import torch, torch.nn as nn, torch.nn.functional as F
from common import TYPES, make_folds
import featlib

torch.set_num_threads(5)
ROOT = os.path.expanduser('~/discourse/dataset/public')
RUN = os.path.expanduser('~/discourse/runs/nnseq')
NT = len(TYPES)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='gru', choices=['gru', 'mlp', 'trans'])
    p.add_argument('--text', type=int, default=1)
    p.add_argument('--hidden', type=int, default=96)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--fdrop', type=float, default=0.1)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--anchor_w', type=float, default=1.8)
    p.add_argument('--cw', default='sqrt', choices=['sqrt', 'balanced', 'none'])
    p.add_argument('--ls', type=float, default=0.05)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--dim_t', type=int, default=20)
    p.add_argument('--dim_f', type=int, default=12)
    p.add_argument('--t_buck', type=int, default=4096)
    p.add_argument('--f_buck', type=int, default=1024)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--tag', default='')
    p.add_argument('--quiet', type=int, default=1)
    return p.parse_args()

class Tagger(nn.Module):
    def __init__(self, D, a):
        super().__init__()
        H = a.hidden
        self.a = a
        self.dense_proj = nn.Linear(D, H)
        self.e_pt = nn.Embedding(len(featlib.PARTYPE), 8)
        self.e_pk = nn.Embedding(4, 4)
        self.e_db = nn.Embedding(8, 6)
        self.e_pl = nn.Embedding(8, 8)
        self.e_ro = nn.Embedding(3, 4)
        extra = 8 + 4 + 6 + 8 + 4
        if a.text:
            self.eb_t = nn.EmbeddingBag(a.t_buck, a.dim_t, mode='mean')
            self.eb_f = nn.EmbeddingBag(a.f_buck, a.dim_f, mode='mean')
            extra += a.dim_t + a.dim_f
        self.pre = nn.Linear(H + extra, H)
        self.fdrop = nn.Dropout(a.fdrop)
        self.drop = nn.Dropout(a.dropout)
        if a.seq == 'gru':
            self.rnn = nn.GRU(H, H // 2, batch_first=True, bidirectional=True)
            self.seqdim = H
        elif a.seq == 'trans':
            layer = nn.TransformerEncoderLayer(H, 4, H * 2, a.dropout, batch_first=True)
            self.rnn = nn.TransformerEncoder(layer, 2)
            self.seqdim = H
        else:
            self.rnn = None
            self.seqdim = H
        self.head = nn.Linear(self.seqdim, NT)

    def forward(self, b):
        dense = self.fdrop(b['dense'])
        x = self.dense_proj(dense)
        cats = [self.e_pt(b['par_type']), self.e_pk(b['par_kind']),
                self.e_db(b['depthb']), self.e_pl(b['poslen']), self.e_ro(b['role'])]
        parts = [x] + cats
        if self.a.text:
            tv = self.eb_t(b['t_flat'], b['t_off'])
            fv = self.eb_f(b['f_flat'], b['f_off'])
            L = x.shape[1]
            parts.append(tv.unsqueeze(1).expand(-1, L, -1))
            parts.append(fv.unsqueeze(1).expand(-1, L, -1))
        h = torch.cat(parts, -1)
        h = self.drop(F.relu(self.pre(h)))
        if self.a.seq == 'gru':
            h, _ = self.rnn(h)
        elif self.a.seq == 'trans':
            h = self.rnn(h)
        h = self.drop(h)
        return self.head(h)

def make_batch(rows, idx, dev):
    L = rows[idx[0]]['L']
    dense = torch.tensor(np.stack([rows[i]['dense'] for i in idx]), device=dev)
    def cat(k):
        return torch.tensor(np.stack([rows[i][k] for i in idx]), dtype=torch.long, device=dev)
    b = dict(dense=dense, par_type=cat('par_type'), par_kind=cat('par_kind'),
             depthb=cat('depthb'), poslen=cat('poslen'), role=cat('role'), L=L)
    for pre, key in [('t', 'title_ng'), ('f', 'forum_ng')]:
        flat, off, o = [], [], 0
        for i in idx:
            off.append(o); ng = rows[i][key]; flat.extend(ng); o += len(ng)
        b[pre + '_flat'] = torch.tensor(flat, dtype=torch.long, device=dev)
        b[pre + '_off'] = torch.tensor(off, dtype=torch.long, device=dev)
    y = None
    if rows[idx[0]]['y'] is not None:
        y = torch.tensor(np.stack([rows[i]['y'] for i in idx]), dtype=torch.long, device=dev)
    return b, y

def class_weights(rows, tr_idx, mode):
    if mode == 'none':
        return torch.ones(NT)
    c = np.zeros(NT)
    for i in tr_idx:
        for t in rows[i]['y']:
            c[t] += 1
    if mode == 'sqrt':
        w = (c.sum() / np.maximum(c, 1)) ** 0.5
    else:
        w = c.sum() / (NT * np.maximum(c, 1))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

def standardize(rows, tr_idx):
    alld = np.concatenate([rows[i]['dense'] for i in tr_idx], 0)
    mu = alld.mean(0); sd = alld.std(0); sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)

def apply_std(rows, mu, sd):
    out = []
    for r in rows:
        r2 = dict(r); r2['dense'] = (r['dense'] - mu) / sd
        out.append(r2)
    return out

def buckets(rows, idx):
    d = collections.defaultdict(list)
    for i in idx:
        d[rows[i]['L']].append(i)
    return d

def train_one(rows, fit_idx, es_idx, a, seed, cw, dev='cpu'):
    torch.manual_seed(seed); np.random.seed(seed)
    D = rows[0]['dense'].shape[1]
    model = Tagger(D, a).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    cw = cw.to(dev)
    tr_b = buckets(rows, fit_idx)
    best_f1, best_state, bad = -1, None, 0
    for ep in range(a.epochs):
        model.train()
        order = []
        for L, ids in tr_b.items():
            ids = ids.copy(); np.random.shuffle(ids)
            for s in range(0, len(ids), a.bs):
                order.append(ids[s:s + a.bs])
        np.random.shuffle(order)
        for chunk in order:
            b, y = make_batch(rows, chunk, dev)
            logits = model(b)
            loss = F.cross_entropy(logits.reshape(-1, NT), y.reshape(-1),
                                   weight=cw, label_smoothing=a.ls, reduction='none')
            loss = loss.reshape(y.shape)
            wpos = torch.ones_like(loss)
            wpos[:, b['L'] - 1] = a.anchor_w
            loss = (loss * wpos).sum() / wpos.sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        sched.step()
        f1 = eval_f1(model, rows, es_idx, dev)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1

def predict(model, rows, idx, dev='cpu'):
    model.eval()
    out = {}
    b_by_L = buckets(rows, idx)
    with torch.no_grad():
        for L, ids in b_by_L.items():
            for s in range(0, len(ids), 256):
                chunk = ids[s:s + 256]
                b, _ = make_batch(rows, chunk, dev)
                p = F.softmax(model(b), -1).cpu().numpy()
                for bi, i in enumerate(chunk):
                    out[i] = p[bi]
    return out

def eval_f1(model, rows, idx, dev):
    from sklearn.metrics import f1_score
    pr = predict(model, rows, idx, dev)
    yt, yp = [], []
    for i in idx:
        pred = pr[i].argmax(1)
        yt.extend(rows[i]['y']); yp.extend(pred)
    return f1_score(yt, yp, average='macro')

def main():
    a = get_args()
    t0 = time.time()
    tcfg = dict(t_lo=3, t_hi=4, t_buck=a.t_buck, f_lo=3, f_hi=5, f_buck=a.f_buck)
    train = pd.read_csv(os.path.join(ROOT, 'train.csv'))
    test = pd.read_csv(os.path.join(ROOT, 'test.csv'))
    tr_rows, n_tr = featlib.build_rows(train, tcfg, targets=True)
    te_rows, n_te = featlib.build_rows(test, tcfg, targets=False)
    folds = make_folds(train)

    oof = np.zeros((n_tr, NT), np.float32)
    test_acc = np.zeros((n_te, NT), np.float32)
    n_models = 0
    fold_f1 = []
    for f in range(5):
        tr_idx = [i for i in range(len(tr_rows)) if folds[i] != f]
        va_idx = [i for i in range(len(tr_rows)) if folds[i] == f]
        mu, sd = standardize(tr_rows, tr_idx)
        trs = apply_std(tr_rows, mu, sd)
        tes = apply_std(te_rows, mu, sd)
        cw = class_weights(tr_rows, tr_idx, a.cw)
        rng = np.random.RandomState(100 + f)
        tr_sh = tr_idx.copy(); rng.shuffle(tr_sh)
        n_es = max(int(0.15 * len(tr_sh)), 40)
        es_idx = tr_sh[:n_es]; fit_idx = tr_sh[n_es:]
        va_sum = {i: np.zeros((trs[i]['L'], NT)) for i in va_idx}
        te_sum = {i: np.zeros((tes[i]['L'], NT)) for i in range(len(tes))}
        for sd_i in range(a.seeds):
            model, bf1 = train_one(trs, fit_idx, es_idx, a, seed=sd_i, cw=cw)
            fold_f1.append(bf1)
            vp = predict(model, trs, va_idx)
            for i in va_idx:
                va_sum[i] += vp[i]
            tp = predict(model, tes, list(range(len(tes))))
            for i in range(len(tes)):
                te_sum[i] += tp[i]
            n_models += 1
        for i in va_idx:
            p = va_sum[i] / a.seeds
            for k, g in enumerate(trs[i]['nidx']):
                oof[g] = p[k]
        for i in range(len(tes)):
            p = te_sum[i] / a.seeds
            for k, g in enumerate(tes[i]['nidx']):
                test_acc[g] += p[k]
        if not a.quiet:
            print(f'fold {f} done, mean ES f1 {np.mean(fold_f1[-a.seeds:]):.4f}', flush=True)
    test_probs = test_acc / 5.0
    tag = ('_' + a.tag) if a.tag else ''
    np.save(os.path.join(RUN, f'oof_probs{tag}.npy'), oof)
    np.save(os.path.join(RUN, f'test_probs{tag}.npy'), test_probs)
    print(f'[done] tag={a.tag!r} models={n_models} meanESf1={np.mean(fold_f1):.4f} '
          f'oof_argmax_macroF1={_oof_f1(oof, tr_rows):.4f} time={time.time()-t0:.1f}s', flush=True)

def _oof_f1(oof, rows):
    from sklearn.metrics import f1_score
    yt, yp = [], []
    for r in rows:
        for k, g in enumerate(r['nidx']):
            yt.append(r['y'][k]); yp.append(int(oof[g].argmax()))
    return f1_score(yt, yp, average='macro')

if __name__ == '__main__':
    main()
