"""Strict local validator for the Traffic Sign Proposal Coupling submission.

Usage: python validate_submission.py [submission.csv] [test.csv]
Exits non-zero and prints the first failure if anything is off.
"""
import sys, os
import pandas as pd

ROUTES = {
    "JOINT_ALIGNED", "JOINT_OFFSET", "SEPARATION_ALIGNED",
    "SEPARATION_OFFSET", "INDEPENDENT_PROPOSALS",
}


def _find(cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def main():
    sub_path = sys.argv[1] if len(sys.argv) > 1 else "working/submission.csv"
    here = os.path.dirname(os.path.abspath(__file__))
    test_path = sys.argv[2] if len(sys.argv) > 2 else _find([
        "dataset/public/test.csv", "dataset/test.csv", "test.csv",
        os.path.join(here, "dataset", "test.csv"),
    ])
    assert test_path, "could not locate test.csv"

    sub = pd.read_csv(sub_path)
    test = pd.read_csv(test_path)

    assert list(sub.columns) == ["id", "screening_route"], \
        f"columns must be ['id','screening_route'] in order, got {list(sub.columns)}"
    assert len(sub) == len(test), f"row count {len(sub)} != test {len(test)}"

    assert sub["id"].isna().sum() == 0, "null ids"
    assert sub["screening_route"].isna().sum() == 0, "null/empty screening_route"
    assert sub["id"].duplicated().sum() == 0, "duplicate ids"

    sub_ids, test_ids = set(sub["id"]), set(test["id"])
    assert sub_ids == test_ids, (
        f"id mismatch: missing={len(test_ids - sub_ids)} extra={len(sub_ids - test_ids)}"
    )

    bad = set(sub["screening_route"].unique()) - ROUTES
    assert not bad, f"illegal labels: {bad}"

    print("VALID:", sub_path)
    print(sub["screening_route"].value_counts().to_string())


if __name__ == "__main__":
    main()
