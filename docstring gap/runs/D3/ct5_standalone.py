"""D3: measure codet5-small ZERO-SHOT standalone chrF on bucket-0, apples-to-apples with C1 (0.3189/0.321).
Uses solution_v3's exact build_t5_input / t5_extract / f_pooled helpers so it measures the shipped code path.
Also dumps per-row pred/ref/f/seq_logprob for gate + hybrid-oracle analysis."""
import os, sys, time, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "7")
sys.path.insert(0, os.path.expanduser("~/docgap/runs/D1"))
import numpy as np, pandas as pd, torch
from solution_v3 import bucket, build_t5_input, t5_extract, f_pooled, T5_MODE
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2500)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--quant", type=int, default=1)
ap.add_argument("--bucket", type=int, default=0)
ap.add_argument("--threads", type=int, default=7)
ap.add_argument("--maxnew", type=int, default=16)
ap.add_argument("--model", default="Salesforce/codet5-small")
ap.add_argument("--out", default="")
args = ap.parse_args()

torch.set_num_threads(args.threads)
tr = pd.read_csv(os.path.expanduser("~/docgap/dataset/train.csv"), keep_default_na=False)
tr["_bkt"] = tr.masked_docstring.map(bucket)
val = tr[tr._bkt == args.bucket]
if 0 < args.n < len(val):
    val = val.sample(args.n, random_state=1).reset_index(drop=True)
else:
    val = val.reset_index(drop=True)
print(f"[data] bucket{args.bucket} n={len(val)} model={args.model} mode={T5_MODE}", flush=True)

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
if args.quant:
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()
    print("[q] int8 dynamic", flush=True)

masked = val.masked_docstring.values; codes = val.code_context.values
texts = [build_t5_input(masked[i], codes[i], tok, T5_MODE) for i in range(len(val))]
preds, logps = [], []
t0 = time.time(); bs = args.bs
for i in range(0, len(texts), bs):
    chunk = texts[i:i + bs]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.maxnew, num_beams=1, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    dec = tok.batch_decode(out.sequences, skip_special_tokens=False)
    preds.extend([t5_extract(x) for x in dec])
    try:
        ts = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        gen = out.sequences[:, 1:]; mask = (gen != tok.pad_token_id)
        lp = (ts * mask).sum(dim=1); ln = mask.sum(dim=1).clamp(min=1)
        logps.extend((lp / ln).tolist())
    except Exception:
        logps.extend([-1.0] * len(chunk))
    if (i // bs) % 15 == 0:
        print(f"  {i + len(chunk)}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
dt = time.time() - t0
refs = val.target_span.astype(str).values
fs = [f_pooled(p, r) for p, r in zip(preds, refs)]
chrf = sum(fs) / len(fs)
lp = np.array(logps); fa = np.array(fs)
print(f"[RESULT] {args.model} zero-shot bucket{args.bucket} n={len(val)} chrF={chrf:.4f} "
      f"rows/s={len(val) / dt:.1f} quant={args.quant} bs={bs}", flush=True)
print(f"[corr] seq_logprob vs f = {np.corrcoef(lp, fa)[0, 1]:.4f}", flush=True)
print(f"[logp] mean={lp.mean():.3f} p10={np.quantile(lp, .1):.3f} p50={np.quantile(lp, .5):.3f} "
      f"p90={np.quantile(lp, .9):.3f}", flush=True)
print(f"[exact] hit={np.mean([p==r for p,r in zip(preds,refs)]):.4f} "
      f"empty_pred={np.mean([str(p)=='' for p in preds]):.4f}", flush=True)
if args.out:
    idcol = val.id.values if "id" in val.columns else np.arange(len(val))
    pd.DataFrame({"id": idcol, "masked": masked, "pred": preds, "ref": refs, "f": fs,
                  "logp": logps}).to_csv(args.out, index=False)
    print(f"[save] {args.out}", flush=True)
