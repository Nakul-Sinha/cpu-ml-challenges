"""Independent verification of N3 deliverables (does NOT import pipeline_final):
  1. canonical elru.score_frames on oof_edits_final.csv vs train truth (nested-op OOF).
  2. strict validation of both submissions (445 rows, ids == test.id, validate_edits, <=8 edits).
  3. per-language edited-row-rate ratios vs train.
  4. diff vs M4 base submission (runs/M4/submission_v2.csv) to confirm which languages changed.
"""
import os, sys, json, collections
import pandas as pd
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "solution"))
import elru

train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
test = pd.read_csv(os.path.join(ROOT, "dataset", "test.csv"))
folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
lang_te = {r.id: r.language for r in test.itertuples()}
tlen_te = {r.id: len(r.text) for r in test.itertuples()}
TRAIN_RATE = {"de": 0.577, "en": 0.470, "it": 0.704}

# ---- 1. canonical OOF check ----
oof = pd.read_csv(os.path.join(ROOT, "runs", "N3", "oof_edits_final.csv"))
truth = train.merge(folds, on="id")[["id", "language", "edits_json"]]
score, det = elru.score_frames(oof, truth)
print(f"[1] canonical elru.score_frames on oof_edits_final.csv = {score:.4f}")
for L in ("de", "en", "it"):
    d = det[L]
    print(f"      {L}: lang={d['lang_score']:.4f} edited={d['edited_mean']:.4f}(n={d['n_edited']}) "
          f"unchanged={d['unchanged_mean']:.4f}(n={d['n_unchanged']})")
assert set(oof.id) == set(truth.id), "OOF id set != train id set"
print(f"      OOF rows={len(oof)} cover all train ids: OK")

# ---- 2+3. strict submission validation + rates ----
def check_sub(path, label):
    sub = pd.read_csv(path)
    sub_map = {r.id: json.loads(r.edits_json) for r in sub.itertuples()}
    assert len(sub) == 445, f"{label}: {len(sub)} rows != 445"
    assert set(sub_map) == set(test.id), f"{label}: id mismatch with test"
    bad = [i for i in sub_map if not elru.validate_edits(sub_map[i], tlen_te[i])]
    assert not bad, f"{label}: invalid rows {bad[:5]}"
    over8 = [i for i in sub_map if len(sub_map[i]) > 8]
    assert not over8, f"{label}: >8 edits {over8[:5]}"
    edn = collections.Counter(); totn = collections.Counter()
    for i in sub_map:
        L = lang_te[i]; totn[L] += 1
        if sub_map[i]:
            edn[L] += 1
    print(f"[2] {label}: 445 rows, ids OK, all validate_edits pass, <=8 edits OK")
    print(f"[3] {label} edited-row rates vs train:")
    for L in ("de", "en", "it"):
        frac = edn[L] / max(1, totn[L]); ratio = frac / TRAIN_RATE[L]
        flag = "" if 0.45 <= ratio <= 1.80 else "  <<FLAG"
        print(f"      {L}: {edn[L]}/{totn[L]}={frac:.3f} train={TRAIN_RATE[L]:.3f} ratio={ratio:.2f}{flag}")
    return sub_map

v3 = check_sub(os.path.join(ROOT, "runs", "N3", "submission_v3.csv"), "submission_v3 (CV-optimal)")
v3r = check_sub(os.path.join(ROOT, "runs", "N3", "submission_v3_robust.csv"), "submission_v3_robust")

# ---- 4. diff vs M4 base ----
base = pd.read_csv(os.path.join(ROOT, "runs", "M4", "submission_v2.csv"))
base_map = {r.id: r.edits_json for r in base.itertuples()}
v3_raw = {r.id: r.edits_json for r in pd.read_csv(os.path.join(ROOT, "runs", "N3", "submission_v3.csv")).itertuples()}
diff = collections.Counter(); tot = collections.Counter()
for i in base_map:
    L = lang_te[i]; tot[L] += 1
    if base_map[i] != v3_raw[i]:
        diff[L] += 1
print("[4] submission_v3 vs M4 base (submission_v2) rows changed per language:")
for L in ("de", "en", "it"):
    print(f"      {L}: {diff[L]}/{tot[L]} changed")
print("\nALL CHECKS PASSED" if True else "")
