"""
Shared training harness for the Traffic Sign Proposal Coupling challenge.

Importable module. Provides:
  - ROUTES + comp_score (EXACT metric from the brief)
  - robust, path-portable data loading (RGB, resize, ImageNet-normalize)
  - StratifiedKFold(5, shuffle, random_state=42) helper
  - near-duplicate estimate (16x16 grayscale downsample -> NN L2)
  - build_resnet18 (pretrained w/ scratch fallback) + configurable train_model
  - label-geometry-preserving augmentation: HORIZONTAL FLIP ONLY (baseline)

All caches must live under /mnt/work (root disk is tiny); callers set TORCH_HOME.
"""
import os
import time
import math
import copy
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Metric (EXACT copy of the brief snippet) + label space
# ----------------------------------------------------------------------------
ROUTES = [
    "JOINT_ALIGNED",
    "JOINT_OFFSET",
    "SEPARATION_ALIGNED",
    "SEPARATION_OFFSET",
    "INDEPENDENT_PROPOSALS",
]
ROUTE_TO_IDX = {r: i for i, r in enumerate(ROUTES)}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def comp_score(y_true, y_pred):
    """Return (overall_score, dict(macro_f1, bal_acc, min_rec)).

    Operates on route-STRING labels (labels=ROUTES), exactly as the brief.
    """
    mf1 = f1_score(y_true, y_pred, labels=ROUTES, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=ROUTES, average=None, zero_division=0)
    return 0.60 * mf1 + 0.25 * rec.mean() + 0.15 * rec.min(), dict(
        macro_f1=mf1, bal_acc=rec.mean(), min_rec=rec.min()
    )


def per_class_recall(y_true, y_pred):
    """Per-class recall aligned to ROUTES order (list of floats)."""
    rec = recall_score(y_true, y_pred, labels=ROUTES, average=None, zero_division=0)
    return {r: float(v) for r, v in zip(ROUTES, rec)}


# ----------------------------------------------------------------------------
# Path-portable data loading
# ----------------------------------------------------------------------------
def find_dataset_root(hint=None):
    """Auto-detect the dataset root by probing common layouts.

    Tries, in order: hint, ./dataset/public, ./dataset, ., and a couple of
    parents. A valid root contains train.csv (+ a train/ dir).
    """
    cands = []
    if hint:
        cands += [hint, os.path.join(hint, "public"), os.path.join(hint, "dataset"),
                  os.path.join(hint, "dataset", "public")]
    here = os.getcwd()
    cands += [
        os.path.join(here, "dataset", "public"),
        os.path.join(here, "dataset"),
        here,
        os.path.join(here, "public"),
        os.path.join(here, "..", "dataset", "public"),
        os.path.join(here, "..", "dataset"),
    ]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "train.csv")):
            return os.path.abspath(c)
    # last resort: return hint or cwd
    return os.path.abspath(hint or here)


def resolve_image_path(root, image_path):
    """Resolve one image path by trying [root, root/public, dirname(root)] bases
    plus a basename fallback under root/{train,test}. Returns first existing."""
    ip = str(image_path)
    bases = [
        os.path.join(root, ip),
        os.path.join(root, "public", ip),
        os.path.join(os.path.dirname(root), ip),
        ip,
    ]
    for b in bases:
        if os.path.isfile(b):
            return b
    # basename fallbacks (in case CSV prefix differs from on-disk layout)
    bn = os.path.basename(ip)
    for sub in ("", "train", "test", "public/train", "public/test"):
        cand = os.path.join(root, sub, bn)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"Could not resolve image path: {image_path} under root {root}")


def load_rgb_uint8(path, res):
    """Open a JPEG as RGB uint8 HWC, resized to (res,res).

    cv2 reads BGR; we convert to RGB so channel order is
    (gradient-mag, horizontal-deriv-mag, vertical-deriv-mag) per the brief.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != res or rgb.shape[1] != res:
        rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_LINEAR)
    return rgb


def load_split(root, csv_name, res, with_labels):
    """Load a CSV split fully into memory.

    Returns dict(ids, paths, imgs_uint8[N,res,res,3], y_idx or None, df).
    """
    df = pd.read_csv(os.path.join(root, csv_name))
    ids = df["id"].astype(str).tolist()
    paths = [resolve_image_path(root, p) for p in df["image_path"].tolist()]
    imgs = np.empty((len(paths), res, res, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        imgs[i] = load_rgb_uint8(p, res)
    y_idx = None
    if with_labels:
        y_idx = np.array([ROUTE_TO_IDX[r] for r in df["screening_route"].tolist()], dtype=np.int64)
    return dict(ids=ids, paths=paths, imgs=imgs, y_idx=y_idx, df=df)


def normalize_to_tensor(imgs_uint8):
    """[N,H,W,3] uint8 RGB -> [N,3,H,W] float32, /255 then ImageNet-normalized."""
    x = imgs_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD           # HWC broadcast
    x = np.transpose(x, (0, 3, 1, 2))                # NCHW
    return torch.from_numpy(np.ascontiguousarray(x))


# ----------------------------------------------------------------------------
# CV splitter
# ----------------------------------------------------------------------------
def make_folds(y_idx, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y_idx)), y_idx))


# ----------------------------------------------------------------------------
# Near-duplicate leakage estimate
# ----------------------------------------------------------------------------
def near_duplicate_report(imgs_uint8, ids=None, thresholds=(20.0, 40.0, 80.0, 160.0),
                          topk=12, log=print):
    """For each image compute a 16x16 grayscale downsample vector, find each
    image's nearest neighbour (L2) among the others, and report near-identical
    pairs. Estimates grouped-CV leakage risk. Returns a dict of stats."""
    n = len(imgs_uint8)
    feats = np.empty((n, 256), dtype=np.float32)
    for i in range(n):
        g = cv2.cvtColor(imgs_uint8[i], cv2.COLOR_RGB2GRAY)
        d = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA)
        feats[i] = d.astype(np.float32).ravel()

    # full pairwise squared L2 via Gram trick (n~1052 -> trivial)
    sq = (feats * feats).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (feats @ feats.T)
    np.fill_diagonal(d2, np.inf)
    d2 = np.maximum(d2, 0.0)
    nn_idx = d2.argmin(1)
    nn_dist = np.sqrt(d2[np.arange(n), nn_idx])

    # unique-pair counts under thresholds (i<j)
    iu, ju = np.triu_indices(n, k=1)
    pair_dist = np.sqrt(d2[iu, ju])
    pair_counts = {float(t): int((pair_dist < t).sum()) for t in thresholds}
    img_counts = {float(t): int((nn_dist < t).sum()) for t in thresholds}

    pcts = {p: float(np.percentile(nn_dist, p)) for p in (0.5, 1, 2, 5, 25, 50)}

    # smallest unique-pair distances for eyeballing
    order = np.argsort(pair_dist)[:topk]
    examples = []
    for k in order:
        i, j = int(iu[k]), int(ju[k])
        ex = dict(i=i, j=j, dist=float(pair_dist[k]))
        if ids is not None:
            ex["id_i"], ex["id_j"] = ids[i], ids[j]
        examples.append(ex)

    log("[near-dup] n=%d  16x16 grayscale L2 nearest-neighbour analysis" % n)
    log("[near-dup] NN-dist percentiles: " +
        ", ".join("p%s=%.1f" % (k, v) for k, v in pcts.items()))
    log("[near-dup] unique-pairs below threshold: " +
        ", ".join("<%g:%d" % (t, c) for t, c in pair_counts.items()))
    log("[near-dup] images with a NN below threshold: " +
        ", ".join("<%g:%d" % (t, c) for t, c in img_counts.items()))
    log("[near-dup] smallest %d pair distances:" % topk)
    for ex in examples:
        tag = (" %s~%s" % (ex.get("id_i", ""), ex.get("id_j", ""))).rstrip()
        log("[near-dup]   d=%.2f (i=%d,j=%d)%s" % (ex["dist"], ex["i"], ex["j"], tag))

    return dict(n=n, nn_dist=nn_dist, nn_idx=nn_idx, pair_counts=pair_counts,
                img_counts=img_counts, percentiles=pcts, examples=examples)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_resnet18(pretrained=True, num_classes=5, log=print):
    """resnet18; pretrained IMAGENET1K_V1 with scratch (weights=None) fallback.
    Returns (model, weights_tag)."""
    from torchvision.models import resnet18, ResNet18_Weights
    tag = None
    if pretrained:
        try:
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            tag = "IMAGENET1K_V1"
        except Exception as e:  # any download / io error -> scratch
            m = resnet18(weights=None)
            tag = "scratch_fallback(%s)" % type(e).__name__
            log("[build] pretrained load FAILED (%s) -> scratch" % type(e).__name__)
    else:
        m = resnet18(weights=None)
        tag = "scratch"
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m, tag


def inverse_freq_weights(y_idx, num_classes=5):
    """Class-weighted CE weights = balanced inverse frequency (mean ~1)."""
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)   # balanced heuristic
    return (w / w.mean()).astype(np.float32)


def _infer_probs(model, X, idx, bs, device):
    model.eval()
    idx = np.asarray(idx)
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            b = torch.as_tensor(idx[s:s + bs], dtype=torch.long)
            xb = X[b].to(device)
            logits = model(xb)
            out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, len(ROUTES)), np.float32)


def train_model(X, y_idx, tr_idx, va_idx, *, epochs=6, bs=64, lr=5e-4,
                weight_decay=1e-4, pretrained=True, class_weights=None,
                hflip=True, seed=42, device="cpu", test_X=None, log=print,
                fold_tag=""):
    """Fine-tune resnet18 on one fold. Validates every epoch by decoding the
    val fold and keeps the best epoch by comp_score. HORIZONTAL FLIP is the
    only augmentation (label-geometry preserving).

    Returns dict(best_score, best_metrics, best_epoch, val_probs, val_idx,
                 test_probs, weights_tag, epoch_time_last, epoch_times).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, tag = build_resnet18(pretrained, num_classes=len(ROUTES), log=log)
    model.to(device)

    cw = None
    if class_weights is not None:
        cw = torch.as_tensor(np.asarray(class_weights, dtype=np.float32), device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_idx = np.asarray(tr_idx)
    va_idx = np.asarray(va_idx)
    steps_per_epoch = max(1, math.ceil(len(tr_idx) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    yt = torch.as_tensor(y_idx, dtype=torch.long)
    val_true = [ROUTES[int(y_idx[i])] for i in va_idx]

    best = dict(best_score=-1.0, best_metrics=None, best_epoch=-1,
                val_probs=None, state=None)
    epoch_times = []
    rng = np.random.RandomState(seed)

    for ep in range(epochs):
        model.train()
        order = tr_idx[rng.permutation(len(tr_idx))]
        t0 = time.time()
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            bt = torch.as_tensor(b, dtype=torch.long)
            xb = X[bt].to(device)   # advanced-index -> fresh copy; safe for in-place flip
            if hflip:
                fm = torch.rand(xb.shape[0]) < 0.5
                if fm.any():
                    xb[fm] = torch.flip(xb[fm], dims=[-1])   # horizontal flip only
            yb = yt[bt].to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
        et = time.time() - t0
        epoch_times.append(et)

        val_probs = _infer_probs(model, X, va_idx, bs, device)
        val_pred = [ROUTES[int(k)] for k in val_probs.argmax(1)]
        sc, md = comp_score(val_true, val_pred)
        log("[%s] ep%d/%d %.1fs val_comp=%.4f mf1=%.4f bal=%.4f minrec=%.4f"
            % (fold_tag, ep + 1, epochs, et, sc, md["macro_f1"], md["bal_acc"], md["min_rec"]))
        if sc > best["best_score"]:
            best.update(best_score=sc, best_metrics=md, best_epoch=ep + 1,
                        val_probs=val_probs,
                        state={k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()})

    # test inference with the best epoch's weights
    test_probs = None
    if test_X is not None and best["state"] is not None:
        model.load_state_dict(best["state"])
        test_probs = _infer_probs(model, test_X, np.arange(len(test_X)), bs, device)

    return dict(best_score=best["best_score"], best_metrics=best["best_metrics"],
                best_epoch=best["best_epoch"], val_probs=best["val_probs"],
                val_idx=va_idx, test_probs=test_probs, weights_tag=tag,
                epoch_time_last=epoch_times[-1] if epoch_times else float("nan"),
                epoch_times=epoch_times)
