"""Draw GT boxes on t0-t3 (+t4) frames, save a horizontal montage per clip.
Usage: python overlay.py <root> <out_dir> <clip1> [clip2 ...]"""
import sys, os, csv, collections
import numpy as np
from PIL import Image, ImageDraw

root, out_dir = sys.argv[1], sys.argv[2]
clips = sys.argv[3:]
os.makedirs(out_dir, exist_ok=True)
boxes = collections.defaultdict(dict); cat = {}
with open(os.path.join(root, "train.csv")) as f:
    for r in csv.DictReader(f):
        boxes[r["clip_id"]][int(r["frame_index"])] = (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
        cat[r["clip_id"]] = r["category"]
for clip in clips:
    ims = []
    for t in range(5):
        im = Image.open(os.path.join(root, "images", "train", clip, f"t{t}.png")).convert("RGB")
        d = ImageDraw.Draw(im)
        x, y, w, h = boxes[clip][t]
        d.rectangle([x, y, x+w, y+h], outline=(0, 200, 0), width=3)
        d.text((5, 5), f"t{t} {cat[clip]}", fill=(0, 150, 0))
        ims.append(np.asarray(im))
    montage = np.concatenate(ims, axis=1)
    Image.fromarray(montage).save(os.path.join(out_dir, f"{clip}.png"))
    print("saved", clip, cat[clip])
