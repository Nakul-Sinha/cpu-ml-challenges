"""Strict validator for ins submission.csv. Usage:
   python validate_submission.py <submission.csv> <test.csv>
Checks every platform requirement; exits nonzero on any violation."""
import sys, json
import pandas as pd


def validate(sub_path, test_path):
    sub = pd.read_csv(sub_path, keep_default_na=False, dtype=str)
    test = pd.read_csv(test_path, keep_default_na=False)
    errs = []
    if list(sub.columns) != ["id", "edits_json"]:
        errs.append(f"columns {list(sub.columns)} != ['id','edits_json']")
    if len(sub) != len(test):
        errs.append(f"row count {len(sub)} != {len(test)}")
    if sub.id.duplicated().any():
        errs.append("duplicate ids")
    if set(sub.id) != set(test.id.astype(str)):
        errs.append("id set mismatch vs test")
    tlen = dict(zip(test.id.astype(str), test.text.str.len()))
    n_edits_total = 0
    n_rows_with = 0
    for r in sub.itertuples():
        try:
            edits = json.loads(r.edits_json)
        except Exception as e:
            errs.append(f"{r.id}: invalid JSON ({e})")
            continue
        if not isinstance(edits, list):
            errs.append(f"{r.id}: not a list")
            continue
        if len(edits) > 8:
            errs.append(f"{r.id}: {len(edits)} edits > 8")
        prev_end = -1
        for e in edits:
            if not isinstance(e, dict) or set(e.keys()) != {"start", "end", "replacement"}:
                errs.append(f"{r.id}: bad edit keys {e}")
                break
            s, en, rep = e["start"], e["end"], e["replacement"]
            if not (isinstance(s, int) and isinstance(en, int) and not isinstance(s, bool) and not isinstance(en, bool)):
                errs.append(f"{r.id}: non-int offsets")
                break
            if not isinstance(rep, str):
                errs.append(f"{r.id}: non-str replacement")
                break
            if not (0 <= s < en <= tlen[str(r.id)]):
                errs.append(f"{r.id}: offsets ({s},{en}) out of range 0..{tlen[str(r.id)]}")
                break
            if len(rep) > 160:
                errs.append(f"{r.id}: replacement len {len(rep)} > 160")
                break
            if s < prev_end:
                errs.append(f"{r.id}: edits unsorted/overlapping")
                break
            prev_end = en
        n_edits_total += len(edits)
        n_rows_with += bool(edits)
    print(f"rows={len(sub)} rows_with_edits={n_rows_with} total_edits={n_edits_total}")
    if errs:
        print("FAIL:")
        for e in errs[:30]:
            print("  -", e)
        return 1
    print("submission VALID")
    return 0


if __name__ == "__main__":
    sys.exit(validate(sys.argv[1], sys.argv[2]))
