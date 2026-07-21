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
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--res", default="256x144"); ap.add_argument("--cache", default="/mnt/work/data/cache")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    outW, outH = map(int, args.res.split("x")); gh, gw = outH//4, outW//4; nf = 4
    torch.set_num_threads(6)
    clips, X, cen, siz, cls, cat = proto3.build_cache(args.root, outW, outH, args.cache)
    nets = []
    for cp in args.ckpts:
        n = proto3.Net(); n.load_state_dict(torch.load(cp, map_location="cpu")["state"]); n.eval(); nets.append(n)
    print(f"ensemble of {len(nets)} checkpoint(s)")
    tr_clips, va_clips = stratified_split(clips, cat, 0.15, 0)
    idx = {c: i for i, c in enumerate(clips)}
    tr = [idx[c] for c in tr_clips]; va = [idx[c] for c in va_clips]
    img_dir = os.path.join(args.root, "images", "train")

    t0 = time.time()
    coarse = {}
    for i in range(len(clips)):
        coarse[i] = decode_tta(nets, X, i, outW, outH, gh, gw, nf, args.tta)  # (cen4or5, siz, prob)
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
    # per-category refined t0-3 hit + size ratio diagnostics
    pc_hit = collections.defaultdict(list); pc_sr = collections.defaultdict(list)
    for i in va:
        for t in range(4):
            pc_hit[cat[clips[i]]].append(iou_xywh(to_box(ref_cen[i][t], ref_siz[i][t]), to_box(cen[i, t], siz[i, t])) >= 0.5)
            gw_, gh_ = siz[i, t]; rw_, rh_ = ref_siz[i][t]
            pc_sr[cat[clips[i]]].append((rw_/max(gw_, 1e-3), rh_/max(gh_, 1e-3)))
    print("refined t0-3 hit per-cat:", {k: round(float(np.mean(v)), 2) for k, v in pc_hit.items()})
    for k in ["people", "car", "cat", "uav"]:
        sr = np.array(pc_sr[k]); print(f"  {k:7s} size-ratio(pred/gt) w~{np.median(sr[:,0]):.2f} h~{np.median(sr[:,1]):.2f}")

    gtb = [to_box(cen[i, 4], siz[i, 4]) for i in va]; gtc = [int(cls[i]) for i in va]
    # classifier on REFINED-box features (best)
    clf = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.06, l2_regularization=2.0, random_state=0)
    clf.fit(np.array([feat_ref[i] for i in tr]), np.array([int(cls[i]) for i in tr]))
    pva = list(clf.predict_proba(np.array([feat_ref[i] for i in va])))
    yva = np.array([int(cls[i]) for i in va])
    print(f"clsAcc(refined)={accuracy_score(yva, np.argmax(pva,1)):.3f}")

    def fc(c4, s4, cfg):
        c = np.array(c4); s = np.array(s4)  # (4,2) center source, (4,2) refined size
        if cfg == "last": cc = c[3]
        elif cfg.startswith("damp"): cc = c[3] + float(cfg[4:])*(c[3]-c[2])
        elif cfg == "linfit":
            t = np.arange(4)[:, None]; A = np.hstack([t, np.ones_like(t)]); cc = (np.linalg.lstsq(A, c, rcond=None)[0]*[[4], [1]]).sum(0)
        return to_box(cc, s[3])

    print("=== forecast sweep (size from REFINED t3; center source x strategy) ===")
    best = None
    for csrc, cend in [("coarse", raw_cen), ("refined", ref_cen)]:
        for cfg in ["last", "damp0.3", "damp0.5", "damp0.8", "linfit"]:
            pb = [fc(cend[i], ref_siz[i], cfg) for i in va]
            macro, catsc, hit, mIoU, mcls = macro_mcfs(pb, pva, gtb, gtc)
            pc = {CATS[k]: round(v, 2) for k, v in catsc.items()}
            print(f"  {csrc:7s} {cfg:8s} MCFS={macro:.4f} hit={hit:.3f} per-cat={pc}")
            if best is None or macro > best[0]: best = (macro, csrc, cfg, hit, pc)
    # learned forecaster: predict t4 center offset from track features (size from refined t3)
    from sklearn.linear_model import Ridge
    def track_feats(c, s):
        c = np.array(c); s = np.array(s); v = np.diff(c, axis=0)
        return np.concatenate([c[3], s[3], v.flatten(), v.mean(0), v[-1]-v[-2], s.mean(0), s.std(0), s[3]-s[0]])
    for csrc, cend in [("coarse", raw_cen), ("refined", ref_cen)]:
        Xf = {i: track_feats(cend[i], ref_siz[i]) for i in range(len(clips))}
        Xtr_f = np.array([Xf[i] for i in tr]); Xva_f = np.array([Xf[i] for i in va])
        ydc = np.array([cen[i, 4]-np.array(cend[i])[3] for i in tr])
        for alpha in [0.3, 1.0, 3.0]:
            rg = Ridge(alpha=alpha).fit(Xtr_f, ydc); dcp = rg.predict(Xva_f)
            pb = [to_box(np.array(cend[i])[3]+dcp[k], ref_siz[i][3]) for k, i in enumerate(va)]
            macro, catsc, hit, mIoU, mcls = macro_mcfs(pb, pva, gtb, gtc)
            pc = {CATS[k]: round(v, 2) for k, v in catsc.items()}
            print(f"  {csrc:7s} learned(a={alpha}) MCFS={macro:.4f} hit={hit:.3f} per-cat={pc}")
            if macro > best[0]: best = (macro, csrc, f"learned_a{alpha}", hit, pc)
    print(f"BEST: MCFS={best[0]:.4f} [{best[1]} center, {best[2]}] hit={best[3]} per-cat={best[4]}")

if __name__ == "__main__":
    main()
