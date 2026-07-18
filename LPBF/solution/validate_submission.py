"""Strict schema/value validation for a LPBF submission CSV against test.csv."""
import os
import sys
import math
import pandas as pd

PUB = sys.argv[2] if len(sys.argv) > 2 else "dataset/public"
sub_path = sys.argv[1] if len(sys.argv) > 1 else "working/submission.csv"


def main():
    test = pd.read_csv(os.path.join(PUB, "test.csv"))
    sub = pd.read_csv(sub_path).fillna("")
    errs = []
    if list(sub.columns) != ["image_id", "prediction_string"]:
        errs.append("columns must be [image_id, prediction_string], got %s" % list(sub.columns))
    if sub["image_id"].duplicated().any():
        errs.append("duplicate image_id values")
    need = set(test["image_id"]); have = set(sub["image_id"])
    if need != have:
        errs.append("id mismatch: missing=%d extra=%d" % (len(need - have), len(have - need)))
    if len(sub) != len(test):
        errs.append("row count %d != test %d" % (len(sub), len(test)))
    sizes = dict(zip(test["image_id"], zip(test["width"], test["height"])))
    n_box = 0; n_empty = 0
    for _, r in sub.iterrows():
        s = str(r["prediction_string"]).strip()
        if s == "":
            n_empty += 1; continue
        toks = s.split()
        if len(toks) % 5 != 0:
            errs.append("%s: token count %d not multiple of 5" % (r["image_id"], len(toks))); continue
        W, H = sizes.get(r["image_id"], (10 ** 9, 10 ** 9))
        cnt = 0
        for i in range(0, len(toks), 5):
            try:
                sc, x0, y0, x1, y1 = (float(toks[i]), float(toks[i + 1]), float(toks[i + 2]),
                                      float(toks[i + 3]), float(toks[i + 4]))
            except ValueError:
                errs.append("%s: non-numeric group %d" % (r["image_id"], i // 5)); continue
            if any(math.isnan(v) or math.isinf(v) for v in (sc, x0, y0, x1, y1)):
                errs.append("%s: NaN/Inf" % r["image_id"])
            if not (0.0 <= sc <= 1.0):
                errs.append("%s: score %.3f out of [0,1]" % (r["image_id"], sc))
            if not (0 <= x0 < x1 and 0 <= y0 < y1):
                errs.append("%s: bad box %s" % (r["image_id"], (x0, y0, x1, y1)))
            cnt += 1; n_box += 1
        if cnt > 25:
            errs.append("%s: %d boxes > 25" % (r["image_id"], cnt))
    print("rows=%d  empty=%d  boxes=%d  avg=%.2f" % (len(sub), n_empty, n_box, n_box / max(1, len(sub))))
    if errs:
        print("INVALID (%d issues):" % len(errs))
        for e in errs[:20]:
            print("  -", e)
        sys.exit(1)
    print("VALID")


if __name__ == "__main__":
    main()
