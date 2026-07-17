"""Prototype: multi-frame soft-argmax CNN -> t4 box + category. Validates on macro-MCFS.
Usage: python proto_cnn.py <root> [--epochs N] [--res 320x180] [--cache path]"""
import sys, os, time, argparse, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CATS, CAT2I, FRAME_W, FRAME_H, read_train_csv, load_clip_input,
                    iou_xywh, macro_mcfs, stratified_split)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--res", default="320x180")
    ap.add_argument("--cache", default="/mnt/work/eris/cache")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    return ap.parse_args()

# ---------------- data ----------------
def build_cache(root, outW, outH, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{outW}x{outH}"
    xp = os.path.join(cache_dir, f"trainX_{tag}.npy")
    yp = os.path.join(cache_dir, f"trainY_{tag}.npz")
    boxes, cat = read_train_csv(os.path.join(root, "train.csv"))
    clips = sorted(boxes.keys())
    if os.path.exists(xp) and os.path.exists(yp):
        X = np.load(xp)
        Y = np.load(yp, allow_pickle=True)
        return clips, X, Y["cen"], Y["siz"], Y["cls"], cat
    img_dir = os.path.join(root, "images", "train")
    X = np.zeros((len(clips), 8, outH, outW), np.float16)
    cen = np.zeros((len(clips), 5, 2), np.float32)  # normalized cx,cy for t0..t4
    siz = np.zeros((len(clips), 5, 2), np.float32)  # normalized w,h
    cls = np.zeros((len(clips),), np.int64)
    t0 = time.time()
    for i, clip in enumerate(clips):
        X[i] = load_clip_input(img_dir, clip, outW, outH).astype(np.float16)
        for t in range(5):
            x, y, w, h = boxes[clip][t]
            cen[i, t] = [(x + w/2)/FRAME_W, (y + h/2)/FRAME_H]
            siz[i, t] = [w/FRAME_W, h/FRAME_H]
        cls[i] = CAT2I[cat[clip]]
        if (i+1) % 200 == 0: print(f"  cache {i+1}/{len(clips)} ({time.time()-t0:.0f}s)", flush=True)
    np.save(xp, X); np.savez(yp, cen=cen, siz=siz, cls=cls)
    return clips, X, cen, siz, cls, cat

# ---------------- augmentation ----------------
def augment(x, cen, siz, rng, outW, outH):
    """x:(8,H,W) float32; cen:(5,2),siz:(5,2) normalized. Returns augmented copies."""
    s = rng.uniform(0.75, 1.3)          # scale
    max_tx = 0.18; max_ty = 0.18
    tx = rng.uniform(-max_tx, max_tx) * outW
    ty = rng.uniform(-max_ty, max_ty) * outH
    flip = rng.random() < 0.5
    # affine in output-pixel space, about image center
    cxp, cyp = outW/2, outH/2
    a = s * (-1 if flip else 1)
    M = np.array([[a, 0, cxp - a*cxp + tx],
                  [0, s, cyp - s*cyp + ty]], np.float32)
    xr = np.empty_like(x)
    for c in range(x.shape[0]):
        xr[c] = cv2_warp(x[c], M, outW, outH)
    # transform centers (normalized -> pixel -> M -> normalized)
    cpx = cen[:, 0]*outW; cpy = cen[:, 1]*outH
    ncx = (M[0, 0]*cpx + M[0, 1]*cpy + M[0, 2]) / outW
    ncy = (M[1, 0]*cpx + M[1, 1]*cpy + M[1, 2]) / outH
    ncen = np.stack([ncx, ncy], 1).astype(np.float32)
    nsiz = (siz * s).astype(np.float32)
    return xr, ncen, nsiz

import cv2
def cv2_warp(ch, M, outW, outH):
    return cv2.warpAffine(ch, M, (outW, outH), flags=cv2.INTER_LINEAR, borderValue=0.0)

# ---------------- model ----------------
class Net(nn.Module):
    def __init__(self, nframes=4):
        super().__init__()
        cin = 2*nframes
        def cbr(i, o, s=1): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.stem = cbr(cin, 32, 2)     # /2
        self.b1 = cbr(32, 64, 2)        # /4
        self.b2 = nn.Sequential(cbr(64, 96), cbr(96, 96))   # /4
        self.b3 = nn.Sequential(cbr(96, 128, 2), cbr(128, 128))  # /8
        self.up = nn.Sequential(nn.Conv2d(128, 96, 1), nn.ReLU(inplace=True))
        self.heat = nn.Sequential(nn.Conv2d(96, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 5, 1))
        self.size = nn.Sequential(nn.Conv2d(96, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 10, 1))
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(128, 96), nn.ReLU(inplace=True), nn.Linear(96, 4))

    def forward(self, x):
        s = self.stem(x); f1 = self.b1(s); f2 = self.b2(f1); f3 = self.b3(f2)
        up = F.interpolate(self.up(f3), size=f2.shape[-2:], mode="nearest")  # /8 -> /4
        feat = f2 + up
        heat = self.heat(feat)          # (B,5,h,w)
        size = self.size(feat)          # (B,10,h,w)
        logits = self.cls(f3)           # (B,4)
        B, _, h, w = heat.shape
        # soft-argmax per frame
        p = F.softmax(heat.view(B, 5, -1), dim=2).view(B, 5, h, w)
        xs = (torch.arange(w, device=x.device).float() + 0.5) / w
        ys = (torch.arange(h, device=x.device).float() + 0.5) / h
        cx = (p.sum(2) * xs).sum(2)     # (B,5)
        cy = (p.sum(3) * ys).sum(2)     # (B,5)
        cen = torch.stack([cx, cy], -1)  # (B,5,2)
        sz = size.view(B, 5, 2, h, w)
        w_ = (p.unsqueeze(2) * sz).sum(-1).sum(-1)  # (B,5,2) attention-weighted size
        return cen, w_, logits, p

def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    outW, outH = map(int, args.res.split("x"))
    clips, X, cen, siz, cls, cat = build_cache(args.root, outW, outH, args.cache)
    print(f"cache ready X={X.shape} ({X.nbytes/1e6:.0f}MB)", flush=True)
    tr_clips, va_clips = stratified_split(clips, cat, 0.15, args.seed)
    idx = {c: i for i, c in enumerate(clips)}
    tr = np.array([idx[c] for c in tr_clips]); va = np.array([idx[c] for c in va_clips])
    print(f"train={len(tr)} val={len(va)}", flush=True)

    dev = "cpu"
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.default_rng(args.seed)
    Wf = np.array([FRAME_W, FRAME_H], np.float32)
    fw = torch.tensor([0.3, 0.3, 0.3, 0.6, 1.0])  # per-frame loss weight (t4 highest)

    def make_gauss_target(cen_b, h, w, sigma=1.5):
        B, F5, _ = cen_b.shape
        yy = torch.arange(h).view(1, 1, h, 1).float(); xx = torch.arange(w).view(1, 1, 1, w).float()
        gx = cen_b[..., 0].view(B, F5, 1, 1) * w; gy = cen_b[..., 1].view(B, F5, 1, 1) * h
        g = torch.exp(-((xx-gx)**2 + (yy-gy)**2) / (2*sigma**2))
        return g

    def run_batch(bi, train=True):
        xb = np.empty((len(bi), 8, outH, outW), np.float32)
        cb = np.empty((len(bi), 5, 2), np.float32); sb = np.empty((len(bi), 5, 2), np.float32)
        for k, i in enumerate(bi):
            xi = X[i].astype(np.float32)
            if train:
                xi, ci, si = augment(xi, cen[i], siz[i], rng, outW, outH)
            else:
                ci, si = cen[i], siz[i]
            xb[k] = xi; cb[k] = ci; sb[k] = si
        xt = torch.from_numpy(xb); ct = torch.from_numpy(cb); st = torch.from_numpy(sb)
        yt = torch.from_numpy(cls[bi])
        pcen, psz, logits, pmap = net(xt)
        h, w = pmap.shape[-2:]
        Lc = (F.smooth_l1_loss(pcen, ct, reduction="none", beta=0.02).sum(-1) * fw).mean()
        Ls = (F.smooth_l1_loss(psz, st, reduction="none", beta=0.02).sum(-1) * fw).mean()
        gt = make_gauss_target(ct, h, w)
        Lh = (F.binary_cross_entropy_with_logits(pmap, gt.clamp(0, 1), reduction="none").mean(dim=(2, 3)) * fw).mean()
        Lcls = F.cross_entropy(logits, yt)
        loss = Lc*5.0 + Ls*5.0 + Lh*1.0 + Lcls*0.5
        return loss, (Lc.item(), Ls.item(), Lh.item(), Lcls.item())

    best = -1; best_state = None
    for ep in range(args.epochs):
        net.train(); rng.shuffle(tr); t0 = time.time(); tot = 0
        for b in range(0, len(tr), args.bs):
            bi = tr[b:b+args.bs]
            loss, parts = run_batch(bi, True)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        if (ep+1) % 5 == 0 or ep == args.epochs-1:
            macro, hit, mIoU, clsacc = evaluate(net, X, cen, siz, cls, va, outW, outH)
            if macro > best: best = macro; best_state = {k: v.clone() for k, v in net.state_dict().items()}
            print(f"ep{ep+1:3d} loss={tot/max(1,len(tr)//args.bs):.3f} "
                  f"valMCFS={macro:.4f} best={best:.4f} | IoU>=.5={hit:.3f} mIoU={mIoU:.3f} clsAcc={clsacc:.3f} "
                  f"({time.time()-t0:.0f}s/ep)", flush=True)
    print(f"BEST valMCFS={best:.4f}")
    if best_state: torch.save(best_state, "/mnt/work/eris/proto_best.pt")

@torch.no_grad()
def evaluate(net, X, cen, siz, cls, va, outW, outH, blend=0.5):
    net.eval()
    pred_boxes = []; probs = []; gtb = []; gtc = []
    ious = []
    for i in va:
        xt = torch.from_numpy(X[i].astype(np.float32)).unsqueeze(0)
        pcen, psz, logits, _ = net(xt)
        pcen = pcen[0].numpy(); psz = psz[0].numpy()
        prob = torch.softmax(logits[0], 0).numpy()
        # direct t4
        c4 = pcen[4]; s4 = psz[4]
        # linear extrapolation of predicted track t2,t3 -> t4
        c4e = 2*pcen[3] - pcen[2]; s4e = 2*psz[3] - psz[2]
        cB = blend*c4 + (1-blend)*c4e; sB = blend*s4 + (1-blend)*s4e
        w = max(4, sB[0]*FRAME_W); h = max(4, sB[1]*FRAME_H)
        x = cB[0]*FRAME_W - w/2; y = cB[1]*FRAME_H - h/2
        pred_boxes.append((x, y, w, h)); probs.append(prob)
        gx, gy = cen[i, 4]; gw, gh = siz[i, 4]
        gtb.append((gx*FRAME_W-gw*FRAME_W/2, gy*FRAME_H-gh*FRAME_H/2, gw*FRAME_W, gh*FRAME_H))
        gtc.append(int(cls[i]))
        ious.append(iou_xywh(pred_boxes[-1], gtb[-1]))
    macro, catsc, hit, mIoU, mcls = macro_mcfs(pred_boxes, probs, gtb, gtc)
    clsacc = float(np.mean([np.argmax(p) == g for p, g in zip(probs, gtc)]))
    return macro, hit, mIoU, clsacc

if __name__ == "__main__":
    main()
