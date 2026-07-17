"""ParseRift prototype: char-aware BiLSTM token tagger for parser-disagreement (contested).
Predicts per-token contested bit; metric = MCC. Validates on a sentence-level split with
threshold tuning. Usage: python proto_pr.py <dataset_public_dir> [--epochs N]"""
import sys, os, argparse, math, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.15)
    return ap.parse_args()

MAXCHAR = 18
def shape_feats(tok):
    return [
        float(tok[:1].isupper()), float(tok.isupper()), float(tok.islower()),
        float(any(c.isdigit() for c in tok)), float(all(not c.isalnum() for c in tok)),  # all punct
        float(len(tok) == 1), min(len(tok), 15)/15.0, float(tok.isalpha()),
        float("'" in tok), float(tok in {".", ",", "!", "?", ";", ":", "-", "(", ")", '"', "..."}),
    ]
N_SHAPE = 10

def build_vocab(tokens, min_freq=2):
    from collections import Counter
    wc = Counter(tokens)
    word2i = {"<pad>": 0, "<unk>": 1}
    for w, c in wc.most_common():
        if c >= min_freq: word2i[w] = len(word2i)
    chars = sorted(set("".join(tokens)))
    char2i = {"<pad>": 0, "<unk>": 1}
    for ch in chars: char2i[ch] = len(char2i)
    return word2i, char2i

def encode_sentence(df_sent, word2i, char2i):
    toks = df_sent.token.tolist()
    wi = [word2i.get(t, 1) for t in toks]
    ci = []
    for t in toks:
        cc = [char2i.get(c, 1) for c in t[:MAXCHAR]]
        cc = cc + [0]*(MAXCHAR-len(cc))
        ci.append(cc)
    sh = [shape_feats(t) for t in toks]
    return wi, ci, sh

def load_sentences(df, word2i, char2i, has_label=True):
    out = []
    for sid, g in df.groupby("sentence_id"):
        g = g.sort_values("position")
        wi, ci, sh = encode_sentence(g, word2i, char2i)
        y = g.contested.tolist() if has_label else [0]*len(wi)
        out.append({"sid": sid, "tid": g.token_id.tolist(), "wi": wi, "ci": ci, "sh": sh, "y": y})
    return out

class Tagger(nn.Module):
    def __init__(self, nword, nchar, wdim=96, cdim=24, chid=48, hid=128):
        super().__init__()
        self.we = nn.Embedding(nword, wdim, padding_idx=0)
        self.ce = nn.Embedding(nchar, cdim, padding_idx=0)
        self.cconv = nn.ModuleList([nn.Conv1d(cdim, chid, k, padding=k//2) for k in (3, 4, 5)])
        cout = chid*3
        self.drop = nn.Dropout(0.4)
        self.lstm = nn.LSTM(wdim+cout+N_SHAPE, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(2*hid, 96), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(96, 1))

    def forward(self, wi, ci, sh, mask):
        B, T = wi.shape
        w = self.we(wi)                                  # B,T,wdim
        c = self.ce(ci)                                  # B,T,C,cdim
        c = c.view(B*T, ci.shape[2], -1).transpose(1, 2)  # BT,cdim,C
        c = torch.cat([F.relu(cv(c)).max(2)[0] for cv in self.cconv], 1)  # BT,cout
        c = c.view(B, T, -1)
        x = torch.cat([w, c, sh], -1)
        x = self.drop(x)
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)                  # B,T

def mcc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return 0.0 if den == 0 else (tp*tn-fp*fn)/den

def best_threshold(y, prob):
    y = np.asarray(y); prob = np.asarray(prob)
    best_t, best_m = 0.5, -1
    for t in np.arange(0.1, 0.85, 0.02):
        m = mcc(y, (prob >= t).astype(int))
        if m > best_m: best_m, best_t = m, t
    return best_t, best_m

def make_batches(data, bs, rng, shuffle=True):
    idx = list(range(len(data)))
    if shuffle: rng.shuffle(idx)
    for b in range(0, len(idx), bs):
        yield [data[i] for i in idx[b:b+bs]]

def collate(batch):
    T = max(len(s["wi"]) for s in batch)
    B = len(batch)
    wi = np.zeros((B, T), np.int64); ci = np.zeros((B, T, MAXCHAR), np.int64)
    sh = np.zeros((B, T, N_SHAPE), np.float32); y = np.zeros((B, T), np.float32); mask = np.zeros((B, T), np.float32)
    for i, s in enumerate(batch):
        L = len(s["wi"]); wi[i, :L] = s["wi"]; ci[i, :L] = s["ci"]; sh[i, :L] = s["sh"]
        y[i, :L] = s["y"]; mask[i, :L] = 1
    return (torch.from_numpy(wi), torch.from_numpy(ci), torch.from_numpy(sh),
            torch.from_numpy(mask), torch.from_numpy(y))

def main():
    args = parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root = args.root
    tr = pd.read_parquet(os.path.join(root, "train.parquet"))
    word2i, char2i = build_vocab(tr.token.tolist())
    print(f"vocab words={len(word2i)} chars={len(char2i)}", flush=True)
    data = load_sentences(tr, word2i, char2i, True)
    sids = [d["sid"] for d in data]; rng.shuffle(sids)
    nval = int(len(sids)*args.val_frac); vset = set(sids[:nval])
    train = [d for d in data if d["sid"] not in vset]; val = [d for d in data if d["sid"] in vset]
    print(f"sent train={len(train)} val={len(val)}", flush=True)

    net = Tagger(len(word2i), len(char2i))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    pos = tr.contested.mean(); pw = torch.tensor([(1-pos)/pos])
    best = -1
    for ep in range(args.epochs):
        net.train(); t0 = time.time(); tot = 0
        for batch in make_batches(train, 32, rng):
            wi, ci, sh, mask, y = collate(batch)
            logit = net(wi, ci, sh, mask)
            loss = F.binary_cross_entropy_with_logits(logit, y, weight=mask, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        if (ep+1) % 4 == 0 or ep == args.epochs-1:
            net.eval(); yv = []; pv = []
            with torch.no_grad():
                for batch in make_batches(val, 64, rng, False):
                    wi, ci, sh, mask, y = collate(batch)
                    prob = torch.sigmoid(net(wi, ci, sh, mask)).numpy()
                    for i, s in enumerate(batch):
                        L = len(s["wi"]); yv += s["y"]; pv += prob[i, :L].tolist()
            t, m = best_threshold(yv, pv)
            if m > best: best = m
            print(f"ep{ep+1:3d} loss={tot/max(1,len(train)//32):.3f} valMCC={m:.4f} thr={t:.2f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"BEST valMCC={best:.4f}")

if __name__ == "__main__":
    main()
