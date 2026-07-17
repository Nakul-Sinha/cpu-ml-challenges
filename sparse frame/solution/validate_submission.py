"""Strict validator for Sparse-Frame submission.csv.
Usage: python validate_submission.py <submission.csv> <test.csv>"""
import sys, csv
import numpy as np

EXPECT = ["clip_id", "x", "y", "w", "h", "p_people", "p_car", "p_cat", "p_uav"]

def main():
    sub_path, test_path = sys.argv[1], sys.argv[2]
    test_ids = []
    with open(test_path) as f:
        for r in csv.DictReader(f):
            test_ids.append(r["clip_id"])
    test_set = set(test_ids)
    errs = []
    seen = set()
    with open(sub_path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        if header != EXPECT:
            errs.append(f"HEADER mismatch: got {header}\n           expected {EXPECT}")
        n = 0
        for ln, row in enumerate(rd, 2):
            n += 1
            if len(row) != 9:
                errs.append(f"line {ln}: has {len(row)} cols"); continue
            cid = row[0]
            if cid in seen: errs.append(f"line {ln}: duplicate {cid}")
            seen.add(cid)
            if cid not in test_set: errs.append(f"line {ln}: unknown clip_id {cid}")
            try:
                vals = [float(v) for v in row[1:]]
            except ValueError:
                errs.append(f"line {ln}: non-numeric value"); continue
            x, y, w, h, pp, pc, pk, pu = vals
            if not (w > 0 and h > 0): errs.append(f"line {ln}: w,h must be >0 (got {w},{h})")
            probs = [pp, pc, pk, pu]
            if any(p < 0 for p in probs): errs.append(f"line {ln}: negative prob")
            if abs(sum(probs) - 1.0) > 0.02: errs.append(f"line {ln}: probs sum {sum(probs):.4f} != 1 (+-0.02)")
            if any(not np.isfinite(v) for v in vals): errs.append(f"line {ln}: NaN/Inf")
    missing = test_set - seen
    if missing: errs.append(f"MISSING {len(missing)} clip_ids, e.g. {list(missing)[:3]}")
    if n != len(test_ids): errs.append(f"row count {n} != test rows {len(test_ids)}")
    if errs:
        print("INVALID:"); [print("  -", e) for e in errs[:25]]
        print(f"... {len(errs)} total errors" if len(errs) > 25 else "")
        sys.exit(1)
    print(f"VALID: {n} rows, all {len(test_ids)} test clips covered, schema/probs/boxes OK.")

if __name__ == "__main__":
    main()
