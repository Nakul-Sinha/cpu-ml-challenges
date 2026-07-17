"""Test local mask-based box refinement: coarse CNN center -> local DoG connected-component
box (fixes the size bottleneck). Compares no-refine vs refine on macro-MCFS.
Usage: python refine_eval.py <root> <ckpt> [--res 256x144] [--tta]"""
import sys, os, argparse, time, collections
import numpy as np
from PIL import Image
import cv2
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CAT2I, FRAME_W, FRAME_H, iou_xywh, macro_mcfs, stratified_split, masks
import proto3
from cls_ceiling import feats_from_track, brier_macro
from cls_appearance import appearance
from eval_ckpt import decode_tta, to_box, forecast
from refine import refine_box
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("ckpt")
    ap.add_argument("--res", default="256x144"); ap.add_argument("--cache", default="/mnt/work/eris/cache")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    outW, outH = map(int, args.res.split("x")); gh, gw = outH//4, outW//4; nf = 4
    torch.set_num_threads(4)
    clips, X, cen, siz, cls, cat = proto3.build_cache(args.root, outW, outH, args.cache)
    net = proto3.Net(); net.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state"]); net.eval()
    tr_clips, va_clips = stratified_split(clips, cat, 0.15, 0)
    idx = {c: i for i, c in enumerate(clips)}
    tr = [idx[c] for c in tr_clips]; va = [idx[c] for c in va_clips]
    img_dir = os.path.join(args.root, "images", "train")

    t0 = time.time()
    coarse = {}
    for i in range(len(clips)):
        coarse[i] = decode_tta([net], X, i, outW, outH, gh, gw, nf, args.tta)  # (cen4or5, siz, prob)
    print(f"decoded ({time.time()-t0:.0f}s)", flush=True)

    # per-clip: load masks, refine t0-3 boxes, extract appearance from refined boxes
    t0 = time.time()
    ref_cen = {}; ref_siz = {}; feat_ref = {}; feat_raw = {}
    for i in range(len(clips)):
        c, s, _ = coarse[i]
        rc = np.array(c[:4], float).copy(); rs = np.array(s[:4], float).copy()
        apps_ref = []; apps_raw = []
        for t in range(4):
            img = np.asarray(Image.open(os.path.join(img_dir, clips[i], f"t{t}.png")).convert("RGB"))
            red, blue = masks(img)
            cb = to_box(c[t], s[t])
            rb = refine_box(red, blue, cb)
            rc[t] = [(rb[0]+rb[2]/2)/FRAME_W, (rb[1]+rb[3]/2)/FRAME_H]; rs[t] = [rb[2]/FRAME_W, rb[3]/FRAME_H]
            apps_ref.append(appearance(red, blue, rb)); apps_raw.append(appearance(red, blue, cb))
        ref_cen[i] = rc; ref_siz[i] = rs
        ar = np.array(apps_ref); araw = np.array(apps_raw)
        feat_ref[i] = np.concatenate([feats_from_track(rc, rs), ar.mean(0), ar.std(0), ar[-1]])
        feat_raw[i] = np.concatenate([feats_from_track(np.array(c[:4]), np.array(s[:4])), araw.mean(0), araw.std(0), araw[-1]])
    print(f"refined+features ({time.time()-t0:.0f}s)", flush=True)

    # detection hit rate t0-3: raw vs refined
    def det_hit(cend, sizd):
        hits = []
        for i in va:
            for t in range(4):
                hits.append(iou_xywh(to_box(cend[i][t], sizd[i][t]), to_box(cen[i, t], siz[i, t])) >= 0.5)
        return np.mean(hits)
    raw_cen = {i: np.array(coarse[i][0][:4]) for i in range(len(clips))}
    raw_siz = {i: np.array(coarse[i][1][:4]) for i in range(len(clips))}
    print(f"t0-3 hit: RAW={det_hit(raw_cen, raw_siz):.3f}  REFINED={det_hit(ref_cen, ref_siz):.3f}")

    gtb = [to_box(cen[i, 4], siz[i, 4]) for i in va]; gtc = [int(cls[i]) for i in va]
    for tag, fcen, fsiz, feat in [("RAW", raw_cen, raw_siz, feat_raw), ("REFINED", ref_cen, ref_siz, feat_ref)]:
        Xtr = np.array([feat[i] for i in tr]); ytr = np.array([int(cls[i]) for i in tr])
        Xva = np.array([feat[i] for i in va])
        clf = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.06, l2_regularization=2.0, random_state=0)
        clf.fit(Xtr, ytr); pva = list(clf.predict_proba(Xva))
        best = None
        for strat in ["last", "lin2", "linfit", "damp"]:
            pb = [to_box(*forecast(np.array(list(fcen[i])+[fcen[i][-1]]), np.array(list(fsiz[i])+[fsiz[i][-1]]), strat)) for i in va]
            macro, catsc, hit, mIoU, mcls = macro_mcfs(pb, pva, gtb, gtc)
            if best is None or macro > best[1]: best = (strat, macro, hit, {CATS[k]: round(v, 2) for k, v in catsc.items()})
        yva = np.array([int(cls[i]) for i in va])
        print(f"{tag:8s} clsAcc={accuracy_score(yva, np.argmax(pva,1)):.3f} | BEST {best[0]} MCFS={best[1]:.4f} hit={best[2]:.3f} per-cat={best[3]}")

if __name__ == "__main__":
    main()
