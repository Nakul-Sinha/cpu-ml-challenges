"""Verify the self-contained solution.py reproduces the research CV score, to
catch any porting bugs. Uses solution.py's own train/predict code on a held-out
split and scores with the metric harness."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import numpy as np
import pandas as pd
import solution as S
from metric import full_score


def family(h):
    return "gray" if int(h) == 358 else "color"


def make_split(df, val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed); val = []
    for fam in ["gray", "color"]:
        ids = [i for i in df.index if family(df.loc[i, "height"]) == fam]
        rng.shuffle(ids); val += ids[:int(round(len(ids) * val_frac))]
    mask = df.index.isin(val)
    return df[~mask].copy(), df[mask].copy()


def main():
    S.PUB = S.find_public_dir(None)
    df = pd.read_csv(os.path.join(S.PUB, "train.csv")).reset_index(drop=True)
    tr, va = make_split(df)
    anchors = {f: S.build_anchors(tr, f) for f in S.FAMS}
    prior = {f: S.prior_heatmap(tr, f, 448 if f == "color" else 358, 448) for f in S.FAMS}
    bg = {f: S.build_background(tr, f, S.PUB) for f in S.FAMS}
    cls, regs, shp = S.train(tr, anchors, prior, bg)
    print("trained dim=%d rows=%d" % (shp[1], shp[0]))
    tm, pm = {}, {}
    for _, r in va.iterrows():
        fam = family(r["height"]); bgr = S.load_bgr(S.PUB, r["image_path"])
        fe = S.FeatureExtractor(bgr, fam, prior[fam], bg[fam])
        out = S.predict_image(fe, anchors, cls, regs, fam)
        tm[r["image_id"]] = S.parse_boxes(r["boxes"])
        pm[r["image_id"]] = [(s, b[0], b[1], b[2], b[3]) for (s, b) in out]
    sc, d = full_score(pm, tm)
    print("SELF-CONTAINED solution.py val score = %.4f  @.50=%.3f @.75=%.3f @.85=%.3f"
          % (sc, d["m50"], d["m75"], d["m85"]))


if __name__ == "__main__":
    main()
