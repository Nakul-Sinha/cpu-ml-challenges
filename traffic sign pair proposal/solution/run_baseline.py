"""
Baseline driver for the Traffic Sign Proposal Coupling challenge.

Pretrained resnet18, res=192, horizontal-flip-only aug, class-weighted CE,
StratifiedKFold(5). Produces OOF comp_score + per-class recall and a 5-fold
softmax-ensemble test submission. Fully self-contained; run under nohup.
"""
import os
import sys
import time

# --- caches + threads MUST be set before importing torch-heavy harness bits ---
os.environ.setdefault("TORCH_HOME", "/mnt/work/torch_cache")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/mnt/work/traffic")
import harness as H

torch.set_num_threads(6)

# ------------------------------- config -------------------------------------
DATA_ROOT = "/mnt/work/traffic/dataset"
WORK = "/mnt/work/traffic/exp_baseline"
SUB_PATH = os.path.join(WORK, "submission.csv")
RES = 192
BS = 64
LR = 5e-4
WD = 1e-4
N_SPLITS = 5
SEED = 42
PRETRAINED = True
TARGET_TRAIN_SEC = 12.5 * 60      # target for the 5-fold portion (~12-15 min)
MIN_EPOCHS, MAX_EPOCHS = 3, 14    # ~50s/epoch x5 folds => 3 epochs ~= 12.6 min
DEVICE = "cpu"


def log(*a):
    print(*a, flush=True)


def main():
    os.makedirs(WORK, exist_ok=True)
    t_start = time.time()
    log("=== BASELINE START ===  torch=%s threads=%d res=%d bs=%d"
        % (torch.__version__, torch.get_num_threads(), RES, BS))

    root = H.find_dataset_root(DATA_ROOT)
    log("[data] dataset root:", root)
    tr = H.load_split(root, "train.csv", RES, with_labels=True)
    te = H.load_split(root, "test.csv", RES, with_labels=False)
    log("[data] train=%d test=%d  img shape=%s" % (len(tr["ids"]), len(te["ids"]), tr["imgs"].shape[1:]))

    # class distribution
    counts = np.bincount(tr["y_idx"], minlength=len(H.ROUTES))
    log("[data] class counts: " + ", ".join("%s=%d" % (r, c) for r, c in zip(H.ROUTES, counts)))
    cw = H.inverse_freq_weights(tr["y_idx"])
    log("[data] class-weights (inv-freq): " + ", ".join("%s=%.3f" % (r, w) for r, w in zip(H.ROUTES, cw)))

    # near-duplicate leakage estimate (train only)
    H.near_duplicate_report(tr["imgs"], ids=tr["ids"], log=log)

    # normalized tensors (hflip commutes with per-channel normalization)
    Xtr = H.normalize_to_tensor(tr["imgs"])
    Xte = H.normalize_to_tensor(te["imgs"])
    y = tr["y_idx"]

    folds = H.make_folds(y, n_splits=N_SPLITS, seed=SEED)

    # ---- timed probe on fold 0 (2 epochs; use warmed steady-state 2nd epoch;
    #      also warms MKL kernels + weight cache) ----
    log("[probe] timing on fold 0 (2 epochs, use warmed steady-state) ...")
    ptr, pva = folds[0]
    probe = H.train_model(Xtr, y, ptr, pva, epochs=2, bs=BS, lr=LR, weight_decay=WD,
                          pretrained=PRETRAINED, class_weights=cw, hflip=True,
                          seed=SEED, device=DEVICE, test_X=None, log=log, fold_tag="probe")
    ep_sec = probe["epoch_times"][-1]   # steady-state (post-warmup)
    log("[probe] weights=%s  cold=%.1fs steady=%.1fs" %
        (probe["weights_tag"], probe["epoch_times"][0], ep_sec))
    epochs = int(round(TARGET_TRAIN_SEC / (N_SPLITS * max(ep_sec, 1.0))))
    epochs = max(MIN_EPOCHS, min(MAX_EPOCHS, epochs))
    est_min = N_SPLITS * epochs * ep_sec / 60.0
    log("[probe] -> epochs=%d/fold  (est 5-fold train ~%.1f min)" % (epochs, est_min))

    # ---- full 5-fold run ----
    oof_probs = np.zeros((len(y), len(H.ROUTES)), dtype=np.float64)
    test_prob_sum = np.zeros((len(te["ids"]), len(H.ROUTES)), dtype=np.float64)
    fold_scores, weights_tags, all_epoch_times = [], [], []

    for f, (tri, vai) in enumerate(folds):
        log("[fold %d] train=%d val=%d" % (f, len(tri), len(vai)))
        r = H.train_model(Xtr, y, tri, vai, epochs=epochs, bs=BS, lr=LR, weight_decay=WD,
                          pretrained=PRETRAINED, class_weights=cw, hflip=True,
                          seed=SEED + f, device=DEVICE, test_X=Xte, log=log,
                          fold_tag="fold%d" % f)
        oof_probs[vai] = r["val_probs"]
        test_prob_sum += r["test_probs"]
        fold_scores.append(r["best_score"])
        weights_tags.append(r["weights_tag"])
        all_epoch_times.extend(r["epoch_times"])
        log("[fold %d] best_epoch=%d best_val_comp=%.4f (mf1=%.4f bal=%.4f minrec=%.4f)"
            % (f, r["best_epoch"], r["best_score"], r["best_metrics"]["macro_f1"],
               r["best_metrics"]["bal_acc"], r["best_metrics"]["min_rec"]))

    # ---- OOF scoring ----
    oof_pred_idx = oof_probs.argmax(1)
    oof_pred = [H.ROUTES[int(k)] for k in oof_pred_idx]
    oof_true = [H.ROUTES[int(k)] for k in y]
    oof_score, oof_md = H.comp_score(oof_true, oof_pred)
    pcr = H.per_class_recall(oof_true, oof_pred)

    log("")
    log("===== OOF RESULTS (5-fold) =====")
    log("OOF comp_score = %.4f" % oof_score)
    log("  macro_f1 = %.4f | bal_acc = %.4f | min_rec = %.4f"
        % (oof_md["macro_f1"], oof_md["bal_acc"], oof_md["min_rec"]))
    log("  per-fold best val comp: " + ", ".join("%.4f" % s for s in fold_scores)
        + "  mean=%.4f" % float(np.mean(fold_scores)))
    log("  per-class recall (OOF):")
    for r in H.ROUTES:
        log("    %-22s %.4f  (n=%d)" % (r, pcr[r], int(counts[H.ROUTE_TO_IDX[r]])))
    # OOF confusion (rows=true, cols=pred) for diagnostics
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(oof_true, oof_pred, labels=H.ROUTES)
    log("  OOF confusion (rows=true, cols=pred; order=ROUTES):")
    for i, r in enumerate(H.ROUTES):
        log("    %-22s %s" % (r, " ".join("%3d" % v for v in cm[i])))

    # ---- test ensemble submission ----
    test_prob = test_prob_sum / N_SPLITS
    test_pred = [H.ROUTES[int(k)] for k in test_prob.argmax(1)]
    sub = pd.DataFrame({"id": te["ids"], "screening_route": test_pred})
    sub.to_csv(SUB_PATH, index=False)
    log("")
    log("[submission] wrote %s (%d rows)" % (SUB_PATH, len(sub)))
    log("[submission] pred distribution: " +
        ", ".join("%s=%d" % (r, int((np.array(test_pred) == r).sum())) for r in H.ROUTES))

    # ---- validate submission ----
    validate_submission(SUB_PATH, te)

    log("[weights] fold tags: " + ", ".join(weights_tags))
    log("[timing] mean epoch=%.1fs total-epochs=%d wall=%.1f min"
        % (float(np.mean(all_epoch_times)), len(all_epoch_times), (time.time() - t_start) / 60.0))
    log("=== BASELINE DONE ===")


def validate_submission(path, te):
    df = pd.read_csv(path)
    ok = True
    cols = list(df.columns)
    if cols != ["id", "screening_route"]:
        ok = False; log("[VALIDATE][FAIL] header is %s" % cols)
    if len(df) != 342:
        ok = False; log("[VALIDATE][FAIL] row count = %d (expected 342)" % len(df))
    exp_ids = set(te["ids"])
    got_ids = set(df["id"].astype(str))
    if got_ids != exp_ids:
        ok = False
        log("[VALIDATE][FAIL] id mismatch: missing=%d extra=%d"
            % (len(exp_ids - got_ids), len(got_ids - exp_ids)))
    if df["id"].duplicated().any():
        ok = False; log("[VALIDATE][FAIL] duplicate ids present")
    bad = set(df["screening_route"].astype(str)) - set(H.ROUTES)
    if bad:
        ok = False; log("[VALIDATE][FAIL] illegal labels: %s" % bad)
    if df.isna().any().any() or (df["screening_route"].astype(str).str.len() == 0).any():
        ok = False; log("[VALIDATE][FAIL] NaN/empty cells present")
    log("[VALIDATE] %s  rows=%d uniq_ids=%d labels_ok=%s"
        % ("PASS" if ok else "FAIL", len(df), df["id"].nunique(), not bad))
    return ok


if __name__ == "__main__":
    main()
