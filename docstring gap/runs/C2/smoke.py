"""Quick T5 smoke test: verify decode format, sentinel extraction, rough zero-shot chrF."""
import sys, os, time, re, hashlib
os.environ.setdefault("OMP_NUM_THREADS", "7")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import pandas as pd, torch
torch.set_num_threads(7)
sys.path.insert(0, "solution")
from chrf import f_pooled
from transformers import T5TokenizerFast, T5ForConditionalGeneration

GAP = "[GAP]"
def bucket(s):
    return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 20

t0 = time.time()
tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").eval()
print("load", f"{time.time()-t0:.1f}s")

d = pd.read_csv("dataset/train.csv", keep_default_na=False)
b = d.masked_docstring.map(bucket)
val = d[b == 0].head(40)

texts = [r.masked_docstring.replace(GAP, "<extra_id_0>") for r in val.itertuples()]
refs = [str(r.target_span) for r in val.itertuples()]

enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
t1 = time.time()
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=16, num_beams=1, do_sample=False)
print("gen 40 rows", f"{time.time()-t1:.1f}s")
dec = tok.batch_decode(out, skip_special_tokens=False)

def extract(text):
    m = re.search(r"<extra_id_0>(.*?)(?:<extra_id_1>|<extra_id_|</s>|<pad>|$)", text, re.DOTALL)
    return (m.group(1).strip() if m else text.strip())

preds = [extract(x) for x in dec]
fs = [f_pooled(p, r) for p, r in zip(preds, refs)]
for i in range(12):
    print(f"--- raw: {dec[i]!r}")
    print(f"    pred={preds[i]!r} ref={refs[i]!r} f={fs[i]:.3f}")
print("mean chrF (plain zero-shot, 40 rows):", sum(fs)/len(fs))
