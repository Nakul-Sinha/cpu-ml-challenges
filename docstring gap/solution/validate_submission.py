"""Strict validator for docstring-gap submission.csv. Usage:
   python validate_submission.py <submission.csv> <test.csv>"""
import sys
import pandas as pd


def validate(sub_path, test_path):
    sub = pd.read_csv(sub_path, keep_default_na=False, dtype=str)
    test = pd.read_csv(test_path, keep_default_na=False)
    errs = []
    if list(sub.columns) != ["id", "prediction"]:
        errs.append(f"columns {list(sub.columns)} != ['id','prediction']")
    if len(sub) != len(test):
        errs.append(f"row count {len(sub)} != {len(test)}")
    if sub.id.duplicated().any():
        errs.append("duplicate ids")
    if set(sub.id) != set(test.id.astype(str)):
        errs.append("id set mismatch vs test")
    empty = (sub.prediction.str.len() == 0).sum()
    lens = sub.prediction.str.len()
    print(f"rows={len(sub)} empty_preds={empty} len(mean/median/max)={lens.mean():.1f}/{lens.median():.0f}/{lens.max()}")
    # Not a hard platform rule, but empty predictions score 0 on their rows.
    if empty > 0:
        errs.append(f"{empty} empty predictions (each scores 0)")
    if errs:
        print("FAIL:")
        for e in errs[:30]:
            print("  -", e)
        return 1
    print("submission VALID")
    return 0


if __name__ == "__main__":
    sys.exit(validate(sys.argv[1], sys.argv[2]))
