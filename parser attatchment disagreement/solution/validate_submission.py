"""Strict validator for ParseRift submission.csv.
Usage: python validate_submission.py <submission.csv> <test.parquet>"""
import sys, csv
import pandas as pd

def main():
    sub_path, test_path = sys.argv[1], sys.argv[2]
    te = pd.read_parquet(test_path)
    want = list(te.token_id.astype(int))
    want_set = set(want)
    errs = []
    seen = set(); n = 0
    with open(sub_path, newline="") as f:
        rd = csv.reader(f); header = next(rd)
        if header != ["token_id", "contested"]:
            errs.append(f"HEADER must be ['token_id','contested'], got {header}")
        for ln, row in enumerate(rd, 2):
            n += 1
            if len(row) != 2:
                errs.append(f"line {ln}: {len(row)} cols"); continue
            try:
                tid = int(row[0]); c = int(row[1])
            except ValueError:
                errs.append(f"line {ln}: non-integer"); continue
            if c not in (0, 1): errs.append(f"line {ln}: contested={c} not in {{0,1}}")
            if tid in seen: errs.append(f"line {ln}: duplicate token_id {tid}")
            seen.add(tid)
            if tid not in want_set: errs.append(f"line {ln}: unknown token_id {tid}")
    missing = want_set - seen
    if missing: errs.append(f"MISSING {len(missing)} token_ids e.g. {list(missing)[:3]}")
    if n != len(want): errs.append(f"row count {n} != {len(want)}")
    if errs:
        print("INVALID:"); [print("  -", e) for e in errs[:20]]; sys.exit(1)
    print(f"VALID: {n} rows, all {len(want)} test token_ids covered, contested in {{0,1}}.")

if __name__ == "__main__":
    main()
