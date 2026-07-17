"""proto2: grid cell-classification detector (t0-t4) + track extrapolation for t4.
Localization = softmax-CE over a stride-4 grid (1 positive cell/frame) -> trains fast.
Usage: python proto2.py <root> [--epochs N] [--res 320x180]"""
import sys, os, time, argparse, collections
import numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CATS, CAT2I, FRAME_W, FRAME_H, read_train_csv, load_clip_input,
                    iou_xywh, macro_mcfs, stratified_split)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--res", default="320x180")
    ap.add_argument("--cache", default="/mnt/work/eris/cache")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--out", default="/mnt/work/eris/proto2_best.pt")
    return ap.parse_args()

def build_cache(root, outW, outH, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{outW}x{outH}"
    xp = os.path.join(cache_dir, f"trainX_{tag}.npy"); yp = os.path.join(cache_dir, f"trainY_{tag}.npz")
    boxes, cat = read_train_csv(os.path.join(root, "train.csv")); clips = sorted(boxes.keys())
    if os.path.exists(xp) and os.path.exists(yp):
        Y = np.load(yp, allow_pickle=True)
        return clips, np.load(xp), Y["cen"], Y["siz"], Y["cls"], cat
    img_dir = os.path.join(root, "images", "train")
    X = np.zeros((len(clips), 8, outH, outW), np.float16)
    cen = np.zeros((len(clips), 5, 2), np.float32); siz = np.zeros((len(clips), 5, 2), np.float32)
    cls = np.zeros((len(clips),), np.int64); t0 = time.time()
    for i, clip in enumerate(clips):
        X[i] = load_clip_input(img_dir, clip, outW, outH).astype(np.float16)
        for t in range(5):
            x, y, w, h = boxes[clip][t]
            cen[i, t] = [(x+w/2)/FRAME_W, (y+h/2)/FRAME_H]; siz[i, t] = [w/FRAME_W, h/FRAME_H]
        cls[i] = CAT2I[cat[clip]]
        if (i+1) % 200 == 0: print(f"  cache {i+1}/{len(clips)} ({time.time()-t0:.0f}s)", flush=True)
    np.save(xp, X); np.savez(yp, cen=cen, siz=siz, cls=cls)
    return clips, X, cen, siz, cls, cat

def augment(x, cen, siz, rng, outW, outH):
    s = rng.uniform(0.75, 1.3)
    tx = rng.uniform(-0.18, 0.18)*outW; ty = rng.uniform(-0.18, 0.18)*outH
    flip = rng.random() < 0.5
    cxp, cyp = outW/2, outH/2; a = s*(-1 if flip else 1)
    M = np.array([[a, 0, cxp-a*cxp+tx], [0, s, cyp-s*cyp+ty]], np.float32)
    xr = np.empty_like(x)
    for c in range(x.shape[0]):
        xr[c] = cv2.warpAffine(x[c], M, (outW, outH), flags=cv2.INTER_LINEAR, borderValue=0.0)
    cpx = cen[:, 0]*outW; cpy = cen[:, 1]*outH
    ncx = (M[0, 0]*cpx+M[0, 1]*cpy+M[0, 2])/outW; ncy = (M[1, 0]*cpx+M[1, 1]*cpy+M[1, 2])/outH
    return xr, np.stack([ncx, ncy], 1).astype(np.float32), (siz*s).astype(np.float32)

class Net(nn.Module):
    def __init__(self, nframes=4):
        super().__init__()
        def cbr(i, o, s=1): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.stem = cbr(2*nframes, 32, 2)   # /2
        self.b1 = cbr(32, 64, 2)            # /4
        self.b2 = nn.Sequential(cbr(64, 96), cbr(96, 96))
        self.b3 = nn.Sequential(cbr(96, 128, 2), cbr(128, 128))   # /8
        self.up = nn.Sequential(nn.Conv2d(128, 96, 1), nn.ReLU(inplace=True))
        self.shared = nn.Sequential(cbr(96, 96), cbr(96, 96))
        self.hm = nn.Conv2d(96, 5, 1)
        self.off = nn.Conv2d(96, 10, 1)
        self.wh = nn.Conv2d(96, 10, 1)
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(128, 96), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(96, 4))

    def forward(self, x):
        s = self.stem(x); f1 = self.b1(s); f2 = self.b2(f1); f3 = self.b3(f2)
        feat = self.shared(f2 + F.interpolate(self.up(f3), size=f2.shape[-2:], mode="nearest"))
        return self.hm(feat), self.off(feat), torch.sigmoid(self.wh(feat)), self.cls(f3)

def targets_from(cen, siz, gh, gw):
    """cen,siz: (B,5,2) numpy -> tcell(B,5) long, toff(B,5,2), tsz(B,5,2)."""
    B = cen.shape[0]
    gx = np.clip(cen[..., 0]*gw, 0, gw-1e-3); gy = np.clip(cen[..., 1]*gh, 0, gh-1e-3)
    ix = gx.astype(np.int64); iy = gy.astype(np.int64)
    tcell = iy*gw + ix
    toff = np.stack([gx-ix, gy-iy], -1).astype(np.float32)
    return tcell, toff, siz.astype(np.float32)

def gather_cell(mp, cell):
    """mp:(B,5,2,gh*gw), cell:(B,5) -> (B,5,2)."""
    B, F5, C, N = mp.shape
    idx = cell.view(B, F5, 1, 1).expand(B, F5, C, 1)
    return mp.gather(3, idx).squeeze(3)

def main():
    args = parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    outW, outH = map(int, args.res.split("x")); gh, gw = outH//4, outW//4
    clips, X, cen, siz, cls, cat = build_cache(args.root, outW, outH, args.cache)
    print(f"cache X={X.shape} grid={gh}x{gw}", flush=True)
    tr_clips, va_clips = stratified_split(clips, cat, 0.15, args.seed)
    idx = {c: i for i, c in enumerate(clips)}
    tr = np.array([idx[c] for c in tr_clips]); va = np.array([idx[c] for c in va_clips])
    print(f"train={len(tr)} val={len(va)}", flush=True)

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.default_rng(args.seed)
    fw = torch.tensor([1., 1., 1., 1., 1.])

    def run_batch(bi, train=True):
        xb = np.empty((len(bi), 8, outH, outW), np.float32)
        cb = np.empty((len(bi), 5, 2), np.float32); sb = np.empty((len(bi), 5, 2), np.float32)
        for k, i in enumerate(bi):
            xi = X[i].astype(np.float32)
            if train: xi, ci, si = augment(xi, cen[i], siz[i], rng, outW, outH)
            else: ci, si = cen[i], siz[i]
            xb[k] = xi; cb[k] = ci; sb[k] = si
        tcell, toff, tsz = targets_from(cb, sb, gh, gw)
        xt = torch.from_numpy(xb); tcell = torch.from_numpy(tcell); toff = torch.from_numpy(toff); tsz = torch.from_numpy(tsz)
        yt = torch.from_numpy(cls[bi])
        hm, off, wh, logit = net(xt)
        B = xt.shape[0]
        hm = hm.view(B, 5, -1); logp = F.log_softmax(hm, 2)
        Lhm = (-(logp.gather(2, tcell[..., None]).squeeze(2)) * fw).mean()
        off = off.view(B, 5, 2, -1); wh = wh.view(B, 5, 2, -1)
        poff = gather_cell(off, tcell); psz = gather_cell(wh, tcell)
        Loff = (F.smooth_l1_loss(poff, toff, reduction="none", beta=0.1).sum(-1) * fw).mean()
        Lsz = (F.smooth_l1_loss(psz, tsz, reduction="none", beta=0.02).sum(-1) * fw).mean()
        Lcls = F.cross_entropy(logit, yt)
        loss = Lhm*1.0 + Loff*1.0 + Lsz*5.0 + Lcls*0.5
        return loss, (Lhm.item(), Loff.item(), Lsz.item(), Lcls.item())

    best = -1; best_state = None
    for ep in range(args.epochs):
        net.train(); rng.shuffle(tr); t0 = time.time(); tot = 0; nb = 0
        for b in range(0, len(tr), args.bs):
            loss, parts = run_batch(tr[b:b+args.bs], True)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        sched.step()
        if (ep+1) % 5 == 0 or ep == args.epochs-1:
            m = evaluate(net, X, cen, siz, cls, va, outW, outH, gh, gw)
            if m["macro"] > best: best = m["macro"]; best_state = {k: v.clone() for k, v in net.state_dict().items()}
            print(f"ep{ep+1:3d} loss={tot/nb:.3f} parts={parts[0]:.2f}/{parts[1]:.2f}/{parts[2]:.3f}/{parts[3]:.2f} "
                  f"| detIoU(t0-3)={m['det_iou']:.3f} || t4: MCFS={m['macro']:.4f} best={best:.4f} "
                  f"hit={m['hit']:.3f} clsAcc={m['clsacc']:.3f} [{m['strat']}] ({time.time()-t0:.0f}s)", flush=True)
    print(f"BEST valMCFS={best:.4f}")
    if best_state: torch.save({"state": best_state, "args": vars(args)}, args.out)

@torch.no_grad()
def decode(net, X, i, outW, outH, gh, gw):
    xt = torch.from_numpy(X[i].astype(np.float32)).unsqueeze(0)
    hm, off, wh, logit = net(xt)
    hm = hm.view(5, -1); off = off.view(5, 2, -1); wh = wh.view(5, 2, -1)
    cell = hm.argmax(1)  # (5,)
    ix = (cell % gw).float(); iy = (cell // gw).float()
    ox = off[torch.arange(5), 0, cell]; oy = off[torch.arange(5), 1, cell]
    cx = (ix+ox)/gw; cy = (iy+oy)/gh
    w = wh[torch.arange(5), 0, cell]; h = wh[torch.arange(5), 1, cell]
    cen = torch.stack([cx, cy], 1).numpy(); siz = torch.stack([w, h], 1).numpy()
    prob = torch.softmax(logit[0], 0).numpy()
    return cen, siz, prob

def forecast(cen, siz, strat, blend=0.5):
    c = cen[:4]; s = siz[:4]
    if strat == "direct": return cen[4], siz[4]
    if strat == "lin2": return 2*c[3]-c[2], 2*s[3]-s[2]
    if strat == "linfit":
        t = np.arange(4)[:, None]
        A = np.hstack([t, np.ones_like(t)])
        cc = np.linalg.lstsq(A, c, rcond=None)[0]; ss = np.linalg.lstsq(A, s, rcond=None)[0]
        return (cc[0]*4+cc[1]), (ss[0]*4+ss[1])
    if strat == "blend":
        cd, sd = cen[4], siz[4]; cl, sl = 2*c[3]-c[2], 2*s[3]-s[2]
        return blend*cd+(1-blend)*cl, blend*sd+(1-blend)*sl
    if strat == "blend_last":  # blend direct center with lin2, size from last detected
        cd = cen[4]; cl = 2*c[3]-c[2]
        return blend*cd+(1-blend)*cl, s[3]
    raise ValueError(strat)

def to_box(c, s):
    w = max(4, s[0]*FRAME_W); h = max(4, s[1]*FRAME_H)
    return (c[0]*FRAME_W-w/2, c[1]*FRAME_H-h/2, w, h)

@torch.no_grad()
def evaluate(net, X, cen, siz, cls, va, outW, outH, gh, gw):
    net.eval()
    dec = {i: decode(net, X, i, outW, outH, gh, gw) for i in va}
    # detection quality on t0-3
    det = []
    for i in va:
        c, s, _ = dec[i]
        for t in range(4):
            det.append(iou_xywh(to_box(c[t], s[t]), to_box(cen[i, t], siz[i, t])))
    det_iou = float(np.mean(det))
    gtb = [to_box(cen[i, 4], siz[i, 4]) for i in va]; gtc = [int(cls[i]) for i in va]
    best = None
    for strat in ["direct", "lin2", "linfit", "blend", "blend_last"]:
        blends = [0.5] if strat not in ("blend", "blend_last") else [0.3, 0.5, 0.7]
        for bl in blends:
            pb = []; pr = []
            for i in va:
                c, s, prob = dec[i]
                cc, ss = forecast(c, s, strat, bl)
                pb.append(to_box(cc, ss)); pr.append(prob)
            macro, catsc, hit, mIoU, mcls = macro_mcfs(pb, pr, gtb, gtc)
            tag = f"{strat}{bl if strat.startswith('blend') else ''}"
            if best is None or macro > best["macro"]:
                clsacc = float(np.mean([np.argmax(p) == g for p, g in zip(pr, gtc)]))
                best = {"macro": macro, "hit": hit, "mIoU": mIoU, "clsacc": clsacc, "strat": tag, "det_iou": det_iou, "catsc": catsc}
    return best

if __name__ == "__main__":
    main()
