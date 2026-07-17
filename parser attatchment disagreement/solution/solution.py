#!/usr/bin/env python
"""ParseRift: predict per-token parser-attachment disagreement (contested). Self-contained CPU.

Usage: python solution.py [<dataset_public_dir>] [<out_submission_csv>]
Defaults: ./dataset/public  and  ./working/submission.csv

Model: char-aware BiLSTM token tagger trained from scratch. Each token is represented by a
learned word embedding (UNK for rare), a character CNN over its characters (handles the ~45% OOV
test tokens and morphology), and shape features (case, punctuation, digits, length). A 2 layer
BiLSTM reads the whole sentence so each prediction uses surrounding context, which is where the
attachment ambiguity actually comes from. An ensemble of seeds is averaged and the decision
threshold is calibrated for MCC on a held out split. No external parsers, taggers, embeddings, or
data; token_id/sentence_id/position carry no signal and are used only to order tokens.
"""
import sys, os, math, time, argparse
from collections import Counter
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

N_SEEDS = int(os.environ.get("PR_SEEDS", "3"))
EPOCHS = int(os.environ.get("PR_EPOCHS", "22"))
THREADS = int(os.environ.get("PR_THREADS", "8"))
MAXCHAR = 18
N_SHAPE = 10
PUNCT = {".", ",", "!", "?", ";", ":", "-", "(", ")", '"', "...", "'", "/", "$", "&"}

def shape_feats(tok):
    return [float(tok[:1].isupper()), float(tok.isupper()), float(tok.islower()),
            float(any(c.isdigit() for c in tok)), float(all(not c.isalnum() for c in tok)),
            float(len(tok) == 1), min(len(tok), 15)/15.0, float(tok.isalpha()),
            float("'" in tok), float(tok in PUNCT)]

def build_vocab(tokens, min_freq=2):
    wc = Counter(tokens); word2i = {"<pad>": 0, "<unk>": 1}
    for w, c in wc.most_common():
        if c >= min_freq: word2i[w] = len(word2i)
    char2i = {"<pad>": 0, "<unk>": 1}
    for ch in sorted(set("".join(tokens))): char2i[ch] = len(char2i)
    return word2i, char2i

def encode(df, word2i, char2i, has_label=True):
    out = []
    for sid, g in df.groupby("sentence_id"):
        g = g.sort_values("position"); toks = g.token.tolist()
        wi = [word2i.get(t, 1) for t in toks]
        ci = [[char2i.get(c, 1) for c in t[:MAXCHAR]] + [0]*(MAXCHAR-min(len(t), MAXCHAR)) for t in toks]
        sh = [shape_feats(t) for t in toks]
        y = g.contested.tolist() if has_label else [0]*len(wi)
        out.append({"sid": sid, "tid": g.token_id.tolist(), "wi": wi, "ci": ci, "sh": sh, "y": y})
    return out

class Tagger(nn.Module):
    def __init__(self, nword, nchar, wdim=96, cdim=24, chid=48, hid=128):
        super().__init__()
        self.we = nn.Embedding(nword, wdim, padding_idx=0); self.ce = nn.Embedding(nchar, cdim, padding_idx=0)
        self.cconv = nn.ModuleList([nn.Conv1d(cdim, chid, k, padding=k//2) for k in (3, 4, 5)])
        self.drop = nn.Dropout(0.4)
        self.lstm = nn.LSTM(wdim+chid*3+N_SHAPE, hid, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(2*hid, 96), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(96, 1))
    def forward(self, wi, ci, sh):
        B, T = wi.shape
        w = self.we(wi); c = self.ce(ci).view(B*T, ci.shape[2], -1).transpose(1, 2)
        c = torch.cat([F.relu(cv(c)).max(2)[0] for cv in self.cconv], 1).view(B, T, -1)
        x = self.drop(torch.cat([w, c, sh], -1))
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)

def collate(batch):
    T = max(len(s["wi"]) for s in batch); B = len(batch)
    wi = np.zeros((B, T), np.int64); ci = np.zeros((B, T, MAXCHAR), np.int64)
    sh = np.zeros((B, T, N_SHAPE), np.float32); y = np.zeros((B, T), np.float32); mask = np.zeros((B, T), np.float32)
    for i, s in enumerate(batch):
        L = len(s["wi"]); wi[i, :L] = s["wi"]; ci[i, :L] = s["ci"]; sh[i, :L] = s["sh"]; y[i, :L] = s["y"]; mask[i, :L] = 1
    return torch.from_numpy(wi), torch.from_numpy(ci), torch.from_numpy(sh), torch.from_numpy(mask), torch.from_numpy(y)

def mcc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return 0.0 if den == 0 else (tp*tn-fp*fn)/den

def best_threshold(y, prob):
    best_t, best_m = 0.5, -1
    for t in np.arange(0.15, 0.85, 0.01):
        m = mcc(y, (np.asarray(prob) >= t).astype(int))
        if m > best_m: best_m, best_t = m, t
    return best_t, best_m

def train_one(train, nword, nchar, pw, seed, epochs, log=lambda *a: None):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    net = Tagger(nword, nchar); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        net.train(); idx = list(range(len(train))); rng.shuffle(idx)
        for b in range(0, len(idx), 32):
            batch = [train[i] for i in idx[b:b+32]]
            wi, ci, sh, mask, y = collate(batch)
            loss = F.binary_cross_entropy_with_logits(net(wi, ci, sh), y, weight=mask, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval(); return net

@torch.no_grad()
def predict(nets, data):
    """Average sigmoid probs across ensemble. Returns dict token_id->prob."""
    acc = {}
    for net in nets:
        for b in range(0, len(data), 64):
            batch = data[b:b+64]; wi, ci, sh, mask, y = collate(batch)
            prob = torch.sigmoid(net(wi, ci, sh)).numpy()
            for i, s in enumerate(batch):
                for j, tid in enumerate(s["tid"]):
                    acc[tid] = acc.get(tid, 0.0) + prob[i, j]/len(nets)
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="./dataset/public")
    ap.add_argument("out", nargs="?", default="./working/submission.csv")
    args = ap.parse_args()
    torch.set_num_threads(THREADS)
    t0 = time.time()
    tr = pd.read_parquet(os.path.join(args.root, "train.parquet"))
    te = pd.read_parquet(os.path.join(args.root, "test.parquet"))
    word2i, char2i = build_vocab(tr.token.tolist())
    pos = float(tr.contested.mean()); pw = torch.tensor([(1-pos)/pos])
    data = encode(tr, word2i, char2i, True)
    print(f"[{time.time()-t0:.0f}s] vocab w={len(word2i)} c={len(char2i)} sents={len(data)} pos={pos:.3f}", flush=True)

    # 1) held-out split to calibrate threshold + report val MCC
    rng = np.random.default_rng(0); sids = [d["sid"] for d in data]; rng.shuffle(sids)
    nval = int(len(sids)*0.15); vset = set(sids[:nval])
    tr_data = [d for d in data if d["sid"] not in vset]; va_data = [d for d in data if d["sid"] in vset]
    nets_cv = [train_one(tr_data, len(word2i), len(char2i), pw, s, EPOCHS) for s in range(N_SEEDS)]
    vp = predict(nets_cv, va_data)
    yv = [yy for s in va_data for yy in s["y"]]; pv = [vp[t] for s in va_data for t in s["tid"]]
    thr, valmcc = best_threshold(yv, pv)
    print(f"[{time.time()-t0:.0f}s] val MCC={valmcc:.4f} at threshold={thr:.3f} (ensemble of {N_SEEDS})", flush=True)

    # 2) retrain ensemble on ALL train data, predict test
    nets = [train_one(data, len(word2i), len(char2i), pw, 100+s, EPOCHS) for s in range(N_SEEDS)]
    te_data = encode(te, word2i, char2i, False)
    tp = predict(nets, te_data)
    print(f"[{time.time()-t0:.0f}s] predicted {len(tp)} test tokens", flush=True)

    # 3) write submission (one row per test token_id, order as in test.parquet)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sub = te[["token_id"]].copy()
    sub["contested"] = [int(tp[t] >= thr) for t in te.token_id.tolist()]
    sub.to_csv(args.out, index=False)
    os.makedirs("working", exist_ok=True); sub.to_csv(os.path.join("working", "submission.csv"), index=False)
    print(f"[{time.time()-t0:.0f}s] wrote {len(sub)} rows -> {args.out}; contested rate {sub.contested.mean():.3f}", flush=True)

if __name__ == "__main__":
    main()
