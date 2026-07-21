"""D3 submission validator: independent of solution_v3's internal asserts.
Checks: row count == test, id set+ORDER identical to test, zero empty/NaN predictions.
Usage: python validate_submission.py <submission.csv> <dataset_dir>"""
import sys, os
import pandas as pd

sub_path = sys.argv[1]
dd = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/docgap/dataset")
sub = pd.read_csv(sub_path, keep_default_na=False)
test = pd.read_csv(os.path.join(dd, "test.csv"), keep_default_na=False)

ok = True
def check(name, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    ok = ok and cond

print(f"[validate] {sub_path}", flush=True)
check("columns == [id, prediction]", list(sub.columns) == ["id", "prediction"], str(list(sub.columns)))
check("row count == test", len(sub) == len(test), f"{len(sub)} vs {len(test)}")
check("row count == 50000", len(sub) == 50000, f"{len(sub)}")
check("id order == test id order EXACT", sub["id"].tolist() == test["id"].tolist())
check("id set == test id set", set(sub["id"]) == set(test["id"]))
check("ids unique", sub["id"].is_unique, f"dupes={len(sub)-sub['id'].nunique()}")
preds = sub["prediction"].astype(str)
n_empty = int((preds.str.len() == 0).sum())
check("zero empty predictions", n_empty == 0, f"empty={n_empty}")
n_nan = int(sub["prediction"].isna().sum())
check("zero NaN predictions", n_nan == 0, f"nan={n_nan}")
print(f"[stats] mean_pred_len={preds.str.len().mean():.2f} "
      f"unique_preds={preds.nunique()} p50_len={int(preds.str.len().median())}", flush=True)
print(f"[sample] {preds.head(3).tolist()}", flush=True)
print(f"[RESULT] {'ALL CHECKS PASS' if ok else 'VALIDATION FAILED'}", flush=True)
sys.exit(0 if ok else 1)
