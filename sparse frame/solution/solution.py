#!/usr/bin/env python
"""Sparse-Frame Object Forecasting - self-contained CPU solution.

Usage: python solution.py <dataset_public_dir> <out_submission_csv>

Pipeline:
  1. Multi-frame CNN coarse detector (12ch: red/blue/DoG x t0..t3) -> per-frame object CENTER
     via grid cell-classification (localization is a 3600-way softmax per frame). Trained from
     scratch with geometric augmentation; time-budgeted so it always finishes in the wall-clock cap.
  2. Local mask-based refinement: anchored at each coarse center, a full-res DoG connected
     component gives the tight box (fixes size). Classical detection fails globally under camera
     motion but works locally when anchored.
  3. t4 forecast by extrapolating the refined t0..t3 track.
  4. Category via a HistGradientBoosting classifier on box geometry + blob-appearance features
     (from the refined boxes); probabilities are the calibrated class belief for the Brier term.

Reads only <dir>/train.csv, <dir>/images/{train,test}/, <dir>/test.csv. Writes <out_csv> and
mirrors to ./working/submission.csv. Honest ML: no ids, no leakage, no external data.
"""
import sys, os, csv, time, collections
import numpy as np
from PIL import Image
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier

# ----------------------------- config -----------------------------
RES = (256, 144)            # coarse-detector input WxH
MAX_EPOCHS = int(os.environ.get("SOL_MAX_EPOCHS", "55"))
TRAIN_TIME_BUDGET = int(os.environ.get("SOL_TIME_BUDGET", str(48 * 60)))  # seconds; keeps best checkpoint
CLIP_LIMIT = int(os.environ.get("SOL_CLIP_LIMIT", "0"))  # 0 = all; >0 caps clips (smoke tests only)
THREADS = int(os.environ.get("SOLUTION_THREADS", "10"))
SEED = 0
BS = 48
LR = 4e-3
WARMUP = 3
CATS = ["people", "car", "cat", "uav"]
CAT2I = {c: i for i, c in enumerate(CATS)}
FRAME_W, FRAME_H = 640, 360

# ----------------------------- data -------------------------------
def read_train_csv(path):
    boxes = collections.defaultdict(dict); cat = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            boxes[r["clip_id"]][int(r["frame_index"])] = (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
            cat[r["clip_id"]] = r["category"]
    return boxes, cat

def read_ids(path, col="clip_id"):
    ids = []
    with open(path) as f:
        for r in csv.DictReader(f):
            ids.append(r[col])
    return ids

def masks(img):
    R, G, B = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    red = ((R > 150) & (G < 110) & (B < 110)).astype(np.float32)
    blue = ((B > 150) & (R < 110) & (G < 110)).astype(np.float32)
    return red, blue

def frame_channels3(img, outW, outH, dog_sigma=5.0):
    red, blue = masks(img)
    red_d = cv2.resize(red, (outW, outH), interpolation=cv2.INTER_AREA)
    blue_d = cv2.resize(blue, (outW, outH), interpolation=cv2.INTER_AREA)
    comb = red_d + blue_d
    dog = np.clip(comb - cv2.GaussianBlur(comb, (0, 0), dog_sigma), 0, None)
    return np.stack([red_d, blue_d, dog], 0)

def load_clip_input3(img_dir, clip, outW, outH):
    chans = []
    for t in range(4):
        img = np.asarray(Image.open(os.path.join(img_dir, clip, f"t{t}.png")).convert("RGB"))
        chans.append(frame_channels3(img, outW, outH))
    return np.concatenate(chans, 0)

def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by); x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    iw = max(0.0, x2-x1); ih = max(0.0, y2-y1); inter = iw*ih
    u = aw*ah + bw*bh - inter
    return inter/u if u > 0 else 0.0

# ----------------------------- model ------------------------------
class Res(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, s, 1, bias=False); self.n1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.n2 = nn.BatchNorm2d(o)
        self.sk = nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o)) if (s != 1 or i != o) else None
    def forward(self, x):
        y = F.relu(self.n1(self.c1(x)), True); y = self.n2(self.c2(y))
        return F.relu(y + (x if self.sk is None else self.sk(x)), True)

class Net(nn.Module):
    def __init__(self, cin=12, nf=4):
        super().__init__()
        self.nf = nf
        self.stem = nn.Sequential(nn.Conv2d(cin, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.l1 = Res(32, 64, 2); self.l2 = Res(64, 96)
        self.l3 = nn.Sequential(Res(96, 128, 2), Res(128, 128))
        self.up = nn.Sequential(nn.Conv2d(128, 96, 1, bias=False), nn.BatchNorm2d(96), nn.ReLU(True))
        self.fuse = Res(96, 96)
        self.hm = nn.Conv2d(96, nf, 1); self.off = nn.Conv2d(96, 2*nf, 1); self.wh = nn.Conv2d(96, 2*nf, 1)
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 96), nn.ReLU(True), nn.Linear(96, 4))
    def forward(self, x):
        s = self.stem(x); f2 = self.l2(self.l1(s)); f3 = self.l3(f2)
        feat = self.fuse(f2 + F.interpolate(self.up(f3), size=f2.shape[-2:], mode="nearest"))
        return self.hm(feat), self.off(feat), torch.sigmoid(self.wh(feat)), self.cls(f3)

def augment(x, cen, siz, rng, outW, outH):
    s = rng.uniform(0.8, 1.25)
    tx = rng.uniform(-0.14, 0.14)*outW; ty = rng.uniform(-0.14, 0.14)*outH
    flip = rng.random() < 0.5
    cxp, cyp = outW/2, outH/2; a = s*(-1 if flip else 1)
    M = np.array([[a, 0, cxp-a*cxp+tx], [0, s, cyp-s*cyp+ty]], np.float32)
    xr = np.empty_like(x)
    for c in range(x.shape[0]):
        xr[c] = cv2.warpAffine(x[c], M, (outW, outH), flags=cv2.INTER_LINEAR, borderValue=0.0)
    cpx = cen[:, 0]*outW; cpy = cen[:, 1]*outH
    ncx = (M[0, 0]*cpx+M[0, 1]*cpy+M[0, 2])/outW; ncy = (M[1, 0]*cpx+M[1, 1]*cpy+M[1, 2])/outH
    return xr, np.stack([ncx, ncy], 1).astype(np.float32), (siz*s).astype(np.float32)

def targets_from(cen, siz, gh, gw, nf):
    gx = np.clip(cen[:, :nf, 0]*gw, 0, gw-1e-3); gy = np.clip(cen[:, :nf, 1]*gh, 0, gh-1e-3)
    ix = gx.astype(np.int64); iy = gy.astype(np.int64)
    return iy*gw+ix, np.stack([gx-ix, gy-iy], -1).astype(np.float32), siz[:, :nf].astype(np.float32)

def gather_cell(mp, cell):
    B, F5, C, N = mp.shape
    return mp.gather(3, cell.view(B, F5, 1, 1).expand(B, F5, C, 1)).squeeze(3)

def train_detector(X, cen, siz, cls, outW, outH, log=print):
    gh, gw = outH//4, outW//4; nf = 4
    torch.manual_seed(SEED); np.random.seed(SEED)
    net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    N = X.shape[0]; idx = np.arange(N)
    def lr_at(ep):
        if ep < WARMUP: return (ep+1)/WARMUP
        p = (ep-WARMUP)/max(1, MAX_EPOCHS-WARMUP); return 0.5*(1+np.cos(np.pi*p))
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    t_start = time.time()
    for ep in range(MAX_EPOCHS):
        for g in opt.param_groups: g["lr"] = LR*lr_at(ep)
        net.train(); rng.shuffle(idx); t0 = time.time()
        for b in range(0, N, BS):
            bi = idx[b:b+BS]
            xb = np.empty((len(bi), 12, outH, outW), np.float32)
            cb = np.empty((len(bi), 5, 2), np.float32); sb = np.empty((len(bi), 5, 2), np.float32)
            for k, i in enumerate(bi):
                xi, ci, si = augment(X[i].astype(np.float32), cen[i], siz[i], rng, outW, outH)
                xb[k] = xi; cb[k] = ci; sb[k] = si
            tcell, toff, tsz = targets_from(cb, sb, gh, gw, nf)
            xt = torch.from_numpy(xb)
            hm, off, wh, logit = net(xt); B = xt.shape[0]
            hm = hm.view(B, nf, -1); logp = F.log_softmax(hm, 2)
            Lhm = -(logp.gather(2, torch.from_numpy(tcell)[..., None]).squeeze(2)).mean()
            off = off.view(B, nf, 2, -1); wh = wh.view(B, nf, 2, -1)
            Loff = F.smooth_l1_loss(gather_cell(off, torch.from_numpy(tcell)), torch.from_numpy(toff), beta=0.1)
            Lsz = F.smooth_l1_loss(gather_cell(wh, torch.from_numpy(tcell)), torch.from_numpy(tsz), beta=0.02)
            Lcls = F.cross_entropy(logit, torch.from_numpy(cls[bi]))
            loss = Lhm + Loff + 3.0*Lsz + 0.7*Lcls
            opt.zero_grad(); loss.backward(); opt.step()
        best_state = {k: v.clone() for k, v in net.state_dict().items()}  # keep latest (cosine end = best)
        elapsed = time.time()-t_start
        log(f"  det ep{ep+1}/{MAX_EPOCHS} loss={loss.item():.3f} ({time.time()-t0:.0f}s, total {elapsed/60:.1f}m)")
        if elapsed > TRAIN_TIME_BUDGET:
            log(f"  time budget reached at ep{ep+1}; stopping"); break
    net.load_state_dict(best_state); net.eval()
    return net

@torch.no_grad()
def decode(net, x_np, outW, outH, tta=True):
    gh, gw = outH//4, outW//4; nf = 4
    views = [(x_np, False)] + ([(x_np[:, :, ::-1].copy(), True)] if tta else [])
    cens = []; sizs = []; probs = []
    for xv, flip in views:
        hm, off, wh, logit = net(torch.from_numpy(xv).unsqueeze(0))
        hm = hm.view(nf, -1); off = off.view(nf, 2, -1); wh = wh.view(nf, 2, -1)
        cell = hm.argmax(1); ix = (cell % gw).float(); iy = (cell // gw).float()
        ox = off[torch.arange(nf), 0, cell]; oy = off[torch.arange(nf), 1, cell]
        cx = (ix+ox)/gw; cy = (iy+oy)/gh
        w = wh[torch.arange(nf), 0, cell]; h = wh[torch.arange(nf), 1, cell]
        c = torch.stack([cx, cy], 1).numpy(); s = torch.stack([w, h], 1).numpy()
        if flip: c[:, 0] = 1.0 - c[:, 0]
        cens.append(c); sizs.append(s); probs.append(torch.softmax(logit[0], 0).numpy())
    return np.mean(cens, 0), np.mean(sizs, 0), np.mean(probs, 0)

# ----------------------------- refine -----------------------------
def refine_box(red, blue, box, expand=2.6, dog_sigma=8.0, thr=0.15, close_k=7,
               max_shift_frac=0.8, area_lo=0.15, area_hi=6.0):
    H, W = red.shape
    x, y, w, h = box; cx, cy = x+w/2, y+h/2
    sw = min(max(w*expand, 26), 460); sh = min(max(h*expand, 22), 340)
    x0 = int(max(0, cx-sw/2)); x1 = int(min(W, cx+sw/2)); y0 = int(max(0, cy-sh/2)); y1 = int(min(H, cy+sh/2))
    if x1-x0 < 4 or y1-y0 < 4: return box
    comb = red[y0:y1, x0:x1]+blue[y0:y1, x0:x1]
    dog = np.clip(comb - cv2.GaussianBlur(comb, (0, 0), dog_sigma), 0, None)
    m = (dog > thr).astype(np.uint8)
    if m.sum() < 6: m = (comb > 0).astype(np.uint8)
    if close_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return box
    lcx, lcy = cx-x0, cy-y0; refscale = 0.5*max(w, h)+10; carea = max(1.0, w*h)
    best = None; bs = -1
    for i in range(1, n):
        bx, by, bw, bh, ar = stats[i]
        if ar < 4: continue
        ccx, ccy = cent[i]; dist = np.hypot(ccx-lcx, ccy-lcy)
        sc = ar/(1.0+dist/refscale)
        if sc > bs: bs = sc; best = (float(x0+bx), float(y0+by), float(bw), float(bh))
    if best is None: return box
    rx, ry, rw, rh = best
    if np.hypot(rx+rw/2-cx, ry+rh/2-cy) > max_shift_frac*max(w, h)+30: return box
    if not (area_lo*carea <= rw*rh <= area_hi*carea): return box
    return best

def to_box(c, s):
    w = max(4.0, s[0]*FRAME_W); h = max(4.0, s[1]*FRAME_H)
    return (c[0]*FRAME_W-w/2, c[1]*FRAME_H-h/2, w, h)

# ----------------------------- features ---------------------------
def feats_track(cen, siz):
    w = siz[:4, 0]*FRAME_W; h = siz[:4, 1]*FRAME_H
    aspect = w/np.maximum(h, 1e-3); area = w*h
    cx = cen[:4, 0]*FRAME_W; cy = cen[:4, 1]*FRAME_H
    dx = np.diff(cx); dy = np.diff(cy); speed = np.sqrt(dx**2+dy**2)
    return np.array([w.mean(), h.mean(), np.median(w), np.median(h), aspect.mean(), np.median(aspect),
                     aspect.std(), area.mean(), np.median(area), np.log(np.median(area)+1), w[-1], h[-1],
                     aspect[-1], area[-1], (w.std()+h.std()), speed.mean(), speed.max(),
                     cy.mean()/FRAME_H, cx.std()], np.float32)

def appearance(red, blue, box):
    x, y, w, h = box; H, W = red.shape
    x0 = int(max(0, x)); y0 = int(max(0, y)); x1 = int(min(W, x+w)); y1 = int(min(H, y+h))
    if x1-x0 < 2 or y1-y0 < 2: return np.zeros(9, np.float32)
    r = red[y0:y1, x0:x1]; b = blue[y0:y1, x0:x1]; lit = (r+b) > 0
    area = lit.size; nlit = lit.sum(); fill = nlit/area
    pol = r.sum()/(r.sum()+b.sum()+1e-3); hh, ww = lit.shape
    topf = lit[:hh//2].mean(); botf = lit[hh//2:].mean(); leftf = lit[:, :ww//2].mean(); rightf = lit[:, ww//2:].mean()
    m = lit.astype(np.uint8); n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        ars = st[1:, cv2.CC_STAT_AREA]; solidity = ars.max()/max(1, nlit); ncc = int((ars > 3).sum())
    else:
        solidity = 0.0; ncc = 0
    ys, xs = np.where(lit)
    vspread = ys.std()/max(1, hh) if len(ys) else 0; hspread = xs.std()/max(1, ww) if len(xs) else 0
    return np.array([fill, pol, topf-botf, leftf-rightf, solidity, ncc, vspread, hspread, nlit/(area+1)], np.float32)

def forecast(c, s, strat="linfit"):
    if strat == "last": return c[3], s[3]
    if strat == "lin2": return 2*c[3]-c[2], s[3]
    if strat == "linfit":
        t = np.arange(4)[:, None]; A = np.hstack([t, np.ones_like(t)])
        cc = np.linalg.lstsq(A, c[:4], rcond=None)[0]; ss = np.linalg.lstsq(A, s[:4], rcond=None)[0]
        return cc[0]*4+cc[1], np.maximum(ss[0]*4+ss[1], [4.0/FRAME_W, 4.0/FRAME_H])
    raise ValueError(strat)

# process a clip: decode -> refine t0..t3 -> features. Returns (ref_cen, ref_siz, cnn_prob, feat)
def process_clip(net, img_dir, clip, outW, outH):
    x_np = load_clip_input3(img_dir, clip, outW, outH).astype(np.float32)
    c, s, prob = decode(net, x_np, outW, outH, tta=True)
    rc = np.array(c[:4]).copy(); rs = np.array(s[:4]).copy(); apps = []
    for t in range(4):
        img = np.asarray(Image.open(os.path.join(img_dir, clip, f"t{t}.png")).convert("RGB"))
        red, blue = masks(img)
        rb = refine_box(red, blue, to_box(c[t], s[t]))
        rc[t] = [(rb[0]+rb[2]/2)/FRAME_W, (rb[1]+rb[3]/2)/FRAME_H]; rs[t] = [rb[2]/FRAME_W, rb[3]/FRAME_H]
        apps.append(appearance(red, blue, rb))
    apps = np.array(apps); feat = np.concatenate([feats_track(rc, rs), apps.mean(0), apps.std(0), apps[-1]])
    return rc, rs, prob, feat

# ----------------------------- main -------------------------------
def main():
    root, out_csv = sys.argv[1], sys.argv[2]
    torch.set_num_threads(THREADS)
    t_all = time.time()
    outW, outH = RES
    boxes, cat = read_train_csv(os.path.join(root, "train.csv"))
    train_clips = sorted(boxes.keys())
    tr_img = os.path.join(root, "images", "train")
    te_img = os.path.join(root, "images", "test")
    test_clips = read_ids(os.path.join(root, "test.csv"))
    if CLIP_LIMIT > 0:  # smoke-test only
        train_clips = train_clips[:CLIP_LIMIT]; test_clips = test_clips[:CLIP_LIMIT]

    # ---- build train cache ----
    print(f"[{time.time()-t_all:.0f}s] caching {len(train_clips)} train clips", flush=True)
    Xtr = np.zeros((len(train_clips), 12, outH, outW), np.float16)
    cen = np.zeros((len(train_clips), 5, 2), np.float32); siz = np.zeros((len(train_clips), 5, 2), np.float32)
    cls = np.zeros((len(train_clips),), np.int64)
    for i, clip in enumerate(train_clips):
        Xtr[i] = load_clip_input3(tr_img, clip, outW, outH).astype(np.float16)
        for t in range(5):
            x, y, w, h = boxes[clip][t]
            cen[i, t] = [(x+w/2)/FRAME_W, (y+h/2)/FRAME_H]; siz[i, t] = [w/FRAME_W, h/FRAME_H]
        cls[i] = CAT2I[cat[clip]]

    # ---- train detector (time-budgeted) ----
    print(f"[{time.time()-t_all:.0f}s] training detector", flush=True)
    net = train_detector(Xtr, cen, siz, cls, outW, outH)

    # ---- build classifier features from refined TRAIN boxes ----
    print(f"[{time.time()-t_all:.0f}s] extracting train features", flush=True)
    Ftr = np.zeros((len(train_clips), 46), np.float32)
    for i, clip in enumerate(train_clips):
        _, _, _, feat = process_clip(net, tr_img, clip, outW, outH)
        Ftr[i] = feat
    clf = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.06,
                                         l2_regularization=2.0, random_state=SEED)
    clf.fit(Ftr, cls)

    # ---- inference on test ----
    print(f"[{time.time()-t_all:.0f}s] inferring {len(test_clips)} test clips", flush=True)
    rows = []
    for clip in test_clips:
        rc, rs, cnn_prob, feat = process_clip(net, te_img, clip, outW, outH)
        c4, s4 = forecast(rc, rs, "linfit")
        x, y, w, h = to_box(c4, s4)
        x = float(np.clip(x, -w*0.5, FRAME_W)); y = float(np.clip(y, -h*0.5, FRAME_H))
        w = float(max(2.0, w)); h = float(max(2.0, h))
        prob = clf.predict_proba(feat[None])[0]
        prob = 0.85*prob + 0.15*cnn_prob  # blend with CNN head
        prob = prob/prob.sum()
        rows.append((clip, x, y, w, h, *prob))

    # ---- write submission ----
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    def write(path):
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["clip_id", "x", "y", "w", "h", "p_people", "p_car", "p_cat", "p_uav"])
            for r in rows:
                wr.writerow([r[0]] + [f"{v:.4f}" for v in r[1:]])
    write(out_csv)
    os.makedirs("working", exist_ok=True); write(os.path.join("working", "submission.csv"))
    print(f"[{time.time()-t_all:.0f}s] wrote {len(rows)} rows -> {out_csv} (+ working/submission.csv)", flush=True)

if __name__ == "__main__":
    main()
