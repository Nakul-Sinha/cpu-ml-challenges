"""Realistic eval: load proto3 checkpoint(s), flip-TTA decode, robust t4 forecast,
geo+appearance calibrated classifier on DETECTED boxes -> macro-MCFS + diagnostics.
Usage: python eval_ckpt.py <root> <ckpt1> [ckpt2 ...] [--res 256x144] [--tta]"""
import sys, os, argparse, collections, time
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CATS, CAT2I, FRAME_W, FRAME_H, iou_xywh, macro_mcfs, stratified_split, masks)
import proto3
from cls_ceiling import feats_from_track, brier_macro
from cls_appearance import appearance

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--res", default="256x144"); ap.add_argument("--cache", default="/mnt/work/eris/cache")
    ap.add_argument("--tta", action="store_true")
    return ap.parse_args()

@torch.no_grad()
def decode_tta(nets, X, i, outW, outH, gh, gw, nf, tta):
    """Average decoded centers/sizes across nets and (optionally) hflip; average class probs."""
    cens = []; sizs = []; probs = []
    base = X[i].astype(np.float32)
    views = [(base, False)]
    if tta:
        xf = base[:, :, ::-1].copy(); views.append((xf, True))
    for net in nets:
        for xv, flipped in views:
            xt = torch.from_numpy(xv).unsqueeze(0)
            hm, off, wh, logit = net(xt)
            hm = hm.view(nf, -1); off = off.view(nf, 2, -1); wh = wh.view(nf, 2, -1)
            cell = hm.argmax(1); ix = (cell % gw).float(); iy = (cell // gw).float()
            ox = off[torch.arange(nf), 0, cell]; oy = off[torch.arange(nf), 1, cell]
            cx = (ix+ox)/gw; cy = (iy+oy)/gh
            w = wh[torch.arange(nf), 0, cell]; h = wh[torch.arange(nf), 1, cell]
            c = torch.stack([cx, cy], 1).numpy(); s = torch.stack([w, h], 1).numpy()
            if flipped: c[:, 0] = 1.0 - c[:, 0]
            cens.append(c); sizs.append(s); probs.append(torch.softmax(logit[0], 0).numpy())
    return np.mean(cens, 0), np.mean(sizs, 0), np.mean(probs, 0)

def to_box(c, s):
    w = max(4, s[0]*FRAME_W); h = max(4, s[1]*FRAME_H)
    return (c[0]*FRAME_W-w/2, c[1]*FRAME_H-h/2, w, h)

def forecast(c, s, strat):
    if strat == "last": return c[3], s[3]
    if strat == "lin2": return 2*c[3]-c[2], s[3]
    if strat == "lin2s": return 2*c[3]-c[2], 2*s[3]-s[2]
    if strat == "linfit":
        t = np.arange(4)[:, None]; A = np.hstack([t, np.ones_like(t)])
        cc = np.linalg.lstsq(A, c[:4], rcond=None)[0]; ss = np.linalg.lstsq(A, s[:4], rcond=None)[0]
        return cc[0]*4+cc[1], ss[0]*4+ss[1]
    if strat == "damp": return c[3]+0.6*(c[3]-c[2]), s[3]

def main():
    args = parse()
    outW, outH = map(int, args.res.split("x")); gh, gw = outH//4, outW//4; nf = 4
    torch.set_num_threads(4)
    clips, X, cen, siz, cls, cat = proto3.build_cache(args.root, outW, outH, args.cache)
    nets = []
    for cp in args.ckpts:
        n = proto3.Net(); n.load_state_dict(torch.load(cp, map_location="cpu")["state"]); n.eval(); nets.append(n)
    tr_clips, va_clips = stratified_split(clips, cat, 0.15, 0)
    idx = {c: i for i, c in enumerate(clips)}
    tr = [idx[c] for c in tr_clips]; va = [idx[c] for c in va_clips]

    # decode all clips
    t0 = time.time(); det = {}
    for i in range(len(clips)):
        det[i] = decode_tta(nets, X, i, outW, outH, gh, gw, nf, args.tta)
    print(f"decoded {len(clips)} clips ({time.time()-t0:.0f}s), tta={args.tta}", flush=True)

    # diagnostics: t0-3 center/size error + hit rate
    cerr = []; serr = []; hit03 = []
    for i in va:
        c, s, _ = det[i]
        for t in range(4):
            pb = to_box(c[t], s[t]); gb = to_box(cen[i, t], siz[i, t])
            hit03.append(iou_xywh(pb, gb) >= 0.5)
            pc = (c[t]*[FRAME_W, FRAME_H]); gc = (cen[i, t]*[FRAME_W, FRAME_H])
            cerr.append(np.hypot(*(pc-gc)))
            serr.append(abs(s[t, 0]-siz[i, t, 0])*FRAME_W + abs(s[t, 1]-siz[i, t, 1])*FRAME_H)
    cerr = np.array(cerr)
    print(f"t0-3 DETECTION: hit(IoU>=.5)={np.mean(hit03):.3f} centerErr med={np.median(cerr):.1f} "
          f"p75={np.percentile(cerr,75):.1f} within30px={np.mean(cerr<30):.3f} sizeErr med={np.median(serr):.1f}")

    # appearance features need raw masks per clip (from detected boxes)
    img_dir = os.path.join(args.root, "images", "train")
    def clip_features(i, use_gt=False):
        c, s, _ = det[i]
        if use_gt: c = cen[i]; s = siz[i]
        geo = feats_from_track(c, s)
        apps = []
        for t in range(4):
            img = np.asarray(Image.open(os.path.join(img_dir, clips[i], f"t{t}.png")).convert("RGB"))
            red, blue = masks(img); apps.append(appearance(red, blue, to_box(c[t], s[t])))
        apps = np.array(apps); app = np.concatenate([apps.mean(0), apps.std(0), apps[-1]])
        return np.concatenate([geo, app])

    t0 = time.time()
    feat = {i: clip_features(i) for i in range(len(clips))}
    print(f"features ({time.time()-t0:.0f}s)", flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr = np.array([feat[i] for i in tr]); ytr = np.array([int(cls[i]) for i in tr])
    Xva = np.array([feat[i] for i in va]); yva = np.array([int(cls[i]) for i in va])
    clf = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.06, l2_regularization=2.0, random_state=0)
    clf.fit(Xtr, ytr); pva = clf.predict_proba(Xva)
    cnnp = np.array([det[i][2] for i in va])
    from sklearn.metrics import accuracy_score
    print(f"CLASSIFIER (detected-box geo+app): clsAcc={accuracy_score(yva,pva.argmax(1)):.3f} macroBrier={brier_macro(pva,yva,cat):.4f}")
    print(f"CNN class head               : clsAcc={accuracy_score(yva,cnnp.argmax(1)):.3f} macroBrier={brier_macro(cnnp,yva,cat):.4f}")

    gtb = [to_box(cen[i, 4], siz[i, 4]) for i in va]; gtc = [int(cls[i]) for i in va]
    print("\n=== macro-MCFS by forecast strategy (with GBM geo+app probs) ===")
    best = None
    for strat in ["last", "lin2", "lin2s", "linfit", "damp"]:
        pb = [to_box(*forecast(det[i][0], det[i][1], strat)) for i in va]
        macro, catsc, hit, mIoU, mcls = macro_mcfs(pb, list(pva), gtb, gtc)
        cs = {CATS[k]: round(v, 3) for k, v in catsc.items()}
        print(f"  {strat:7s} MCFS={macro:.4f} hit={hit:.3f} mIoU={mIoU:.3f} per-cat={cs}")
        if best is None or macro > best[1]: best = (strat, macro)
    print(f"BEST: {best[0]} MCFS={best[1]:.4f}")

if __name__ == "__main__":
    main()
