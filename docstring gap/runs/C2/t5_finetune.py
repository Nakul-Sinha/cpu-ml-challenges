"""C2 T5 fine-tune probe: in-script fine-tune of t5-small on span-corruption over
docstring gaps (buckets 2-19), with a chrF learning curve measured on a FIXED
1500-row bucket-0 subset. Training-only wall time on the x-axis (eval time excluded).

Fits ONLY on buckets 2-19: bucket 0 = validation, bucket 1 = LOCKED holdout (never touched).
Runtime-fair with sibling: torch threads 7, run under nice -n 10.

Usage:
  python runs/C2/t5_finetune.py --lr 3e-4 --mode plain --budget 1800 --eval_every 300 \
      --tag main --save_ckpt runs/C2/ft_main
  python runs/C2/t5_finetune.py --lr 1e-4 --mode plain --budget 600  --eval_every 300 --tag lr1e4
  python runs/C2/t5_finetune.py --lr 3e-4 --mode code_first --budget 720 --eval_every 240 --tag codehint
"""
import sys, os, time, re, hashlib, json, random, argparse
os.environ.setdefault("OMP_NUM_THREADS", "7")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(7)
sys.path.insert(0, "solution")
sys.path.insert(0, os.path.join("runs", "C2"))
from chrf import f_pooled, score_lists
from t5_probe import build_input, extract, bucket, load_val, GAP
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from torch.optim import AdamW

HERE = os.path.join("runs", "C2")


def eval_chrf(model, tok, val, mode, bs=64, max_new=16):
    texts = [build_input(r.masked_docstring, r.code_context, tok, mode) for r in val.itertuples()]
    refs = [str(r.target_span) for r in val.itertuples()]
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=128)
            out = model.generate(**enc, max_new_tokens=max_new, num_beams=1, do_sample=False)
            preds.extend([extract(x) for x in tok.batch_decode(out, skip_special_tokens=False)])
    return score_lists(preds, refs)


def train_run(lr, mode, budget_s, eval_every, tag, n_train=100000, eval_n=1500,
              bs=24, save_ckpt=None):
    print(f"\n#### RUN tag={tag} lr={lr} mode={mode} budget={budget_s}s bs={bs} ####", flush=True)
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    opt = AdamW(model.parameters(), lr=lr)

    d = pd.read_csv("dataset/train.csv", keep_default_na=False)
    b = d.masked_docstring.map(bucket)
    trn = d[b >= 2].reset_index(drop=True)  # buckets 2-19 only
    trn = trn.sample(min(n_train, len(trn)), random_state=42).reset_index(drop=True)
    val = load_val(eval_n, seed=1)  # FIXED eval subset (bucket 0)
    print(f"train pool={len(trn)} (buckets2-19)  eval={len(val)} (bucket0)", flush=True)

    inputs = [build_input(r.masked_docstring, r.code_context, tok, mode) for r in trn.itertuples()]
    targets = [f"<extra_id_0> {str(r.target_span)} <extra_id_1>" for r in trn.itertuples()]
    idx = list(range(len(inputs)))
    random.seed(0); random.shuffle(idx)

    curve = []
    zs = eval_chrf(model, tok, val, mode)
    curve.append({"train_min": 0.0, "seen": 0, "chrf": zs})
    print(f"  t=0 (zero-shot) chrF={zs:.4f}", flush=True)

    model.train()
    train_elapsed = 0.0
    next_eval = eval_every
    step = seen = ptr = 0
    loss_acc = 0.0
    last = time.time()
    while train_elapsed < budget_s:
        batch_idx = [idx[(ptr + k) % len(idx)] for k in range(bs)]
        ptr += bs
        bt = [inputs[j] for j in batch_idx]
        tg = [targets[j] for j in batch_idx]
        enc = tok(bt, return_tensors="pt", padding=True, truncation=True, max_length=128)
        lab = tok(tg, return_tensors="pt", padding=True, truncation=True, max_length=24).input_ids
        lab[lab == tok.pad_token_id] = -100
        out = model(**enc, labels=lab)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        step += 1; seen += bs; loss_acc += out.loss.item()

        now = time.time()
        train_elapsed += now - last
        last = now
        if train_elapsed >= next_eval:
            c = eval_chrf(model, tok, val, mode)
            sps = seen / train_elapsed
            curve.append({"train_min": train_elapsed/60, "seen": seen, "chrf": c})
            print(f"  t={train_elapsed/60:5.1f}min seen={seen} sps={sps:.1f} "
                  f"loss={loss_acc/step:.3f} chrF={c:.4f}", flush=True)
            model.train()
            next_eval += eval_every
            last = time.time()  # exclude eval time from train clock

    final_sps = seen / train_elapsed
    print(f"  DONE {tag}: {step} steps, {seen} samples, {final_sps:.1f} samples/s train-only", flush=True)

    df = pd.DataFrame(curve)
    df["tag"] = tag; df["lr"] = lr; df["mode"] = mode; df["samples_per_s"] = final_sps
    path = os.path.join(HERE, f"curve_{tag}.csv")
    df.to_csv(path, index=False)
    print(f"  curve -> {path}", flush=True)

    if save_ckpt:
        model.save_pretrained(save_ckpt)
        tok.save_pretrained(save_ckpt)
        print(f"  ckpt -> {save_ckpt}", flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mode", default="plain")
    ap.add_argument("--budget", type=int, default=1800)
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--save_ckpt", default=None)
    a = ap.parse_args()
    t0 = time.time()
    train_run(a.lr, a.mode, a.budget, a.eval_every, a.tag, bs=a.bs, save_ckpt=a.save_ckpt)
    print(f"\nwall {time.time()-t0:.0f}s", flush=True)
