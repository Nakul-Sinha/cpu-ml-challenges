"""proto3: residual backbone + DoG channels, t0-t3 detection only, t4 by extrapolation.
Input 12ch (red,blue,dog x4 frames). Localization = grid CE. Reports per-category det IoU.
Usage: python proto3.py <root> [--epochs N] [--res 256x144]"""
import sys, os, time, argparse, collections
import numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CATS, CAT2I, FRAME_W, FRAME_H, read_train_csv, load_clip_input3,
                    iou_xywh, macro_mcfs, stratified_split)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--res", default="256x144"); ap.add_argument("--cache", default="/mnt/work/data/cache")
    ap.add_argument("--bs", type=int, default=48); ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--out", default="/mnt/work/data/proto3_best.pt")
    ap.add_argument("--balance", type=int, default=1)  # class-balanced sampling (macro metric weights cats equally)
    return ap.parse_args()

def build_cache(root, outW, outH, cache_dir):
    os.makedirs(cache_dir, exist_ok=True); tag = f"{outW}x{outH}_c3"
    xp = os.path.join(cache_dir, f"trainX_{tag}.npy"); yp = os.path.join(cache_dir, f"trainY_{tag}.npz")
    boxes, cat = read_train_csv(os.path.join(root, "train.csv")); clips = sorted(boxes.keys())
    if os.path.exists(xp) and os.path.exists(yp):
        Y = np.load(yp, allow_pickle=True); return clips, np.load(xp), Y["cen"], Y["siz"], Y["cls"], cat
    img_dir = os.path.join(root, "images", "train")
    X = np.zeros((len(clips), 12, outH, outW), np.float16)
    cen = np.zeros((len(clips), 5, 2), np.float32); siz = np.zeros((len(clips), 5, 2), np.float32)
    cls = np.zeros((len(clips),), np.int64); t0 = time.time()
    for i, clip in enumerate(clips):
        X[i] = load_clip_input3(img_dir, clip, outW, outH).astype(np.float16)
        for t in range(5):
            x, y, w, h = boxes[clip][t]
            cen[i, t] = [(x+w/2)/FRAME_W, (y+h/2)/FRAME_H]; siz[i, t] = [w/FRAME_W, h/FRAME_H]
        cls[i] = CAT2I[cat[clip]]
        if (i+1) % 200 == 0: print(f"  cache {i+1}/{len(clips)} ({time.time()-t0:.0f}s)", flush=True)
    np.save(xp, X); np.savez(yp, cen=cen, siz=siz, cls=cls)
    return clips, X, cen, siz, cls, cat

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

class Res(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, s, 1, bias=False); self.n1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.n2 = nn.BatchNorm2d(o)
        self.sk = None
        if s != 1 or i != o:
            self.sk = nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o))
    def forward(self, x):
        y = F.relu(self.n1(self.c1(x)), True); y = self.n2(self.c2(y))
        return F.relu(y + (x if self.sk is None else self.sk(x)), True)

class Net(nn.Module):
    def __init__(self, cin=12, nf=4):
        super().__init__()
        self.nf = nf
        self.stem = nn.Sequential(nn.Conv2d(cin, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))  # /2
        self.l1 = Res(32, 64, 2)     # /4
        self.l2 = Res(64, 96)        # /4 (trimmed 1 block for speed)
        self.l3 = nn.Sequential(Res(96, 128, 2), Res(128, 128))  # /8
        self.up = nn.Sequential(nn.Conv2d(128, 96, 1, bias=False), nn.BatchNorm2d(96), nn.ReLU(True))
        self.fuse = Res(96, 96)      # trimmed 1 block for speed
        self.hm = nn.Conv2d(96, nf, 1)
        self.off = nn.Conv2d(96, 2*nf, 1)
        self.wh = nn.Conv2d(96, 2*nf, 1)
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 96), nn.ReLU(True), nn.Linear(96, 4))
    def forward(self, x):
        s = self.stem(x); f1 = self.l1(s); f2 = self.l2(f1); f3 = self.l3(f2)
        feat = self.fuse(f2 + F.interpolate(self.up(f3), size=f2.shape[-2:], mode="nearest"))
        return self.hm(feat), self.off(feat), torch.sigmoid(self.wh(feat)), self.cls(f3)

def targets_from(cen, siz, gh, gw, nf):
    gx = np.clip(cen[:, :nf, 0]*gw, 0, gw-1e-3); gy = np.clip(cen[:, :nf, 1]*gh, 0, gh-1e-3)
    ix = gx.astype(np.int64); iy = gy.astype(np.int64)
    return iy*gw+ix, np.stack([gx-ix, gy-iy], -1).astype(np.float32), siz[:, :nf].astype(np.float32)

def gather_cell(mp, cell):
    B, F5, C, N = mp.shape
    return mp.gather(3, cell.view(B, F5, 1, 1).expand(B, F5, C, 1)).squeeze(3)

def main():
    args = parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    outW, outH = map(int, args.res.split("x")); gh, gw = outH//4, outW//4; nf = 4
    clips, X, cen, siz, cls, cat = build_cache(args.root, outW, outH, args.cache)
    print(f"cache X={X.shape} grid={gh}x{gw}", flush=True)
    tr_clips, va_clips = stratified_split(clips, cat, 0.15, args.seed)
    idx = {c: i for i, c in enumerate(clips)}
    tr = np.array([idx[c] for c in tr_clips]); va = np.array([idx[c] for c in va_clips])
    print(f"train={len(tr)} val={len(va)}", flush=True)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    def lr_at(ep):
        if ep < args.warmup: return (ep+1)/args.warmup
        p = (ep-args.warmup)/max(1, args.epochs-args.warmup); return 0.5*(1+np.cos(np.pi*p))
    rng = np.random.default_rng(args.seed)

    def run_batch(bi, train=True):
        xb = np.empty((len(bi), 12, outH, outW), np.float32)
        cb = np.empty((len(bi), 5, 2), np.float32); sb = np.empty((len(bi), 5, 2), np.float32)
        for k, i in enumerate(bi):
            xi = X[i].astype(np.float32)
            if train: xi, ci, si = augment(xi, cen[i], siz[i], rng, outW, outH)
            else: ci, si = cen[i], siz[i]
            xb[k] = xi; cb[k] = ci; sb[k] = si
        tcell, toff, tsz = targets_from(cb, sb, gh, gw, nf)
        xt = torch.from_numpy(xb); tcell = torch.from_numpy(tcell); toff = torch.from_numpy(toff); tsz = torch.from_numpy(tsz)
        yt = torch.from_numpy(cls[bi])
        hm, off, wh, logit = net(xt); B = xt.shape[0]
        hm = hm.view(B, nf, -1); logp = F.log_softmax(hm, 2)
        Lhm = -(logp.gather(2, tcell[..., None]).squeeze(2)).mean()
        off = off.view(B, nf, 2, -1); wh = wh.view(B, nf, 2, -1)
        Loff = F.smooth_l1_loss(gather_cell(off, tcell), toff, beta=0.1)
        Lsz = F.smooth_l1_loss(gather_cell(wh, tcell), tsz, beta=0.02)
        Lcls = F.cross_entropy(logit, yt)
        loss = Lhm*1.0 + Loff*1.0 + Lsz*3.0 + Lcls*0.7  # size downweighted; refinement provides final size
        return loss, (Lhm.item(), Loff.item(), Lsz.item(), Lcls.item())

    # class-balanced sampling weights (oversample scarce car/people to match macro objective)
    cls_tr = cls[tr]; counts = np.bincount(cls_tr, minlength=4)
    sw = 1.0/np.maximum(counts[cls_tr], 1); sw = sw/sw.sum()
    best = -1; best_state = None
    for ep in range(args.epochs):
        for g in opt.param_groups: g["lr"] = args.lr*lr_at(ep)
        net.train(); t0 = time.time(); tot = 0; nb = 0
        order = rng.choice(tr, size=len(tr), replace=True, p=sw) if args.balance else tr.copy()
        if not args.balance: rng.shuffle(order)
        for b in range(0, len(order), args.bs):
            loss, parts = run_batch(order[b:b+args.bs], True)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if (ep+1) % 5 == 0 or ep == args.epochs-1:
            m = evaluate(net, X, cen, siz, cls, va, cat, clips, outW, outH, gh, gw, nf)
            # select by CENTER accuracy (refinement provides final size, so center is what matters)
            score = m["cen_acc"]
            if score > best:
                best = score; best_state = {k: v.clone() for k, v in net.state_dict().items()}
                torch.save({"state": best_state, "args": vars(args)}, args.out)  # save on improvement
            cc = m["cencat"]
            print(f"ep{ep+1:3d} loss={tot/nb:.3f} | cen@30={m['cen_acc']:.3f} best={best:.3f} "
                  f"[ppl {cc['people']:.2f} car {cc['car']:.2f} cat {cc['cat']:.2f} uav {cc['uav']:.2f}] "
                  f"detIoU={m['det_iou']:.3f} clsAcc={m['clsacc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"BEST cen@30={best:.4f} (saved to {args.out})")

@torch.no_grad()
def decode(net, X, i, outW, outH, gh, gw, nf):
    xt = torch.from_numpy(X[i].astype(np.float32)).unsqueeze(0)
    hm, off, wh, logit = net(xt)
    hm = hm.view(nf, -1); off = off.view(nf, 2, -1); wh = wh.view(nf, 2, -1)
    cell = hm.argmax(1); ix = (cell % gw).float(); iy = (cell // gw).float()
    ox = off[torch.arange(nf), 0, cell]; oy = off[torch.arange(nf), 1, cell]
    cx = (ix+ox)/gw; cy = (iy+oy)/gh
    w = wh[torch.arange(nf), 0, cell]; h = wh[torch.arange(nf), 1, cell]
    return torch.stack([cx, cy], 1).numpy(), torch.stack([w, h], 1).numpy(), torch.softmax(logit[0], 0).numpy()

def to_box(c, s):
    w = max(4, s[0]*FRAME_W); h = max(4, s[1]*FRAME_H)
    return (c[0]*FRAME_W-w/2, c[1]*FRAME_H-h/2, w, h)

def forecast(c4, s4, strat, bl=0.5):
    if strat == "lin2": return 2*c4[3]-c4[2], 2*s4[3]-s4[2]
    if strat == "last": return c4[3], s4[3]
    if strat == "linfit":
        t = np.arange(4)[:, None]; A = np.hstack([t, np.ones_like(t)])
        cc = np.linalg.lstsq(A, c4, rcond=None)[0]; ss = np.linalg.lstsq(A, s4, rcond=None)[0]
        return cc[0]*4+cc[1], ss[0]*4+ss[1]
    if strat == "lin_half":  # half-velocity (damped)
        return c4[3]+0.5*(c4[3]-c4[2]), s4[3]
    raise ValueError

@torch.no_grad()
def evaluate(net, X, cen, siz, cls, va, cat, clips, outW, outH, gh, gw, nf):
    net.eval()
    dec = {i: decode(net, X, i, outW, outH, gh, gw, nf) for i in va}
    detcat = collections.defaultdict(list); det = []; cenhit = []; cenhit_cat = collections.defaultdict(list)
    for i in va:
        c, s, _ = dec[i]
        for t in range(4):
            v = iou_xywh(to_box(c[t], s[t]), to_box(cen[i, t], siz[i, t])); det.append(v); detcat[cat[clips[i]]].append(v)
            d = np.hypot((c[t, 0]-cen[i, t, 0])*FRAME_W, (c[t, 1]-cen[i, t, 1])*FRAME_H)
            cenhit.append(d < 30); cenhit_cat[cat[clips[i]]].append(d < 30)
    gtb = [to_box(cen[i, 4], siz[i, 4]) for i in va]; gtc = [int(cls[i]) for i in va]
    best = None
    for strat in ["lin2", "last", "linfit", "lin_half"]:
        pb = []; pr = []
        for i in va:
            c, s, prob = dec[i]; cc, ss = forecast(c, s, strat); pb.append(to_box(cc, ss)); pr.append(prob)
        macro, catsc, hit, mIoU, mcls = macro_mcfs(pb, pr, gtb, gtc)
        if best is None or macro > best["macro"]:
            clsacc = float(np.mean([np.argmax(p) == g for p, g in zip(pr, gtc)]))
            best = {"macro": macro, "hit": hit, "mIoU": mIoU, "clsacc": clsacc, "strat": strat,
                    "det_iou": float(np.mean(det)), "detcat": {k: float(np.mean(v)) for k, v in detcat.items()},
                    "cen_acc": float(np.mean(cenhit)),
                    "cencat": {k: float(np.mean(v)) for k, v in cenhit_cat.items()}}
    return best

if __name__ == "__main__":
    main()
