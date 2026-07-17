"""Draw GT(green)/coarse(red)/refined(blue) boxes on t0-t3 for given clips -> montage.
Usage: python diag_boxes.py <root> <ckpt> <out_dir> <clip1> [clip2 ...] [--res 256x144]"""
import sys, os, argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FRAME_W, FRAME_H, read_train_csv, load_clip_input3, masks
import proto3
from refine import refine_box

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("ckpt"); ap.add_argument("out_dir"); ap.add_argument("clips", nargs="+")
ap.add_argument("--res", default="256x144")
args = ap.parse_args()
outW, outH = map(int, args.res.split("x")); gh, gw = outH//4, outW//4
os.makedirs(args.out_dir, exist_ok=True)
boxes, cat = read_train_csv(os.path.join(args.root, "train.csv"))
net = proto3.Net(); net.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state"]); net.eval()
img_dir = os.path.join(args.root, "images", "train")

def to_box(c, s):
    w = max(4, s[0]*FRAME_W); h = max(4, s[1]*FRAME_H)
    return (c[0]*FRAME_W-w/2, c[1]*FRAME_H-h/2, w, h)

@torch.no_grad()
def decode_np(x_np):
    import torch.nn.functional as F
    hm, off, wh, logit = net(torch.from_numpy(x_np).unsqueeze(0))
    hm = hm.view(4, -1); off = off.view(4, 2, -1); wh = wh.view(4, 2, -1)
    cell = hm.argmax(1); ix = (cell % gw).float(); iy = (cell // gw).float()
    ox = off[torch.arange(4), 0, cell]; oy = off[torch.arange(4), 1, cell]
    cx = (ix+ox)/gw; cy = (iy+oy)/gh
    w = wh[torch.arange(4), 0, cell]; h = wh[torch.arange(4), 1, cell]
    return torch.stack([cx, cy], 1).numpy(), torch.stack([w, h], 1).numpy()

for clip in args.clips:
    x_np = load_clip_input3(img_dir, clip, outW, outH).astype(np.float32)
    c, s = decode_np(x_np)
    ims = []
    for t in range(4):
        img = np.asarray(Image.open(os.path.join(img_dir, clip, f"t{t}.png")).convert("RGB"))
        red, blue = masks(img)
        cb = to_box(c[t], s[t]); rb = refine_box(red, blue, cb)
        gx, gy, gw_, gh_ = boxes[clip][t]
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        d.rectangle([gx, gy, gx+gw_, gy+gh_], outline=(0, 220, 0), width=2)      # GT green
        d.rectangle([cb[0], cb[1], cb[0]+cb[2], cb[1]+cb[3]], outline=(255, 140, 0), width=1)  # coarse orange
        d.rectangle([rb[0], rb[1], rb[0]+rb[2], rb[1]+rb[3]], outline=(0, 200, 255), width=2)  # refined cyan
        ims.append(np.asarray(im))
    Image.fromarray(np.concatenate(ims, 1)).save(os.path.join(args.out_dir, f"{clip}_{cat[clip]}.png"))
    print("saved", clip, cat[clip], "GTgreen coarse-orange refined-cyan")
