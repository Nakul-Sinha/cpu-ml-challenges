"""D2 CodeT5 feasibility probe: mirrors C2 t5_probe.py exactly, swapping in
Salesforce/codet5-small (60.5M, same T5 arch as t5-small; RobertaTokenizer BPE,
pretrained on code+docstrings with T5 span-denoising -> <extra_id_N> sentinels).

Reuses C2's build_input/extract/code_hint/load_val/bucket so the eval subsets and
input formats are IDENTICAL to C2 -> curves are directly comparable.
Runtime-fair: torch threads 7, nice -n 10, keep_default_na=False, bucket 0 only
(never touches bucket 1 = locked holdout).

Usage:
  python runs/D2/codet5_probe.py zeroshot
  python runs/D2/codet5_probe.py speed
  python runs/D2/codet5_probe.py dump [ckpt] [mode]
"""
import sys, os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "7")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(7)
sys.path.insert(0, "solution")
sys.path.insert(0, os.path.join("runs", "C2"))
from chrf import f_pooled, score_lists
# identical formatting + eval subsets as C2:
from t5_probe import build_input, extract, code_hint, load_val, bucket, GAP
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "Salesforce/codet5-small"
HERE = os.path.join("runs", "D2")


def load_tok_model(src=MODEL):
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSeq2SeqLM.from_pretrained(src).eval()
    return tok, model


def gen_batch(model, tok, texts, max_new=16, in_len=128):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=in_len)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, num_beams=1, do_sample=False)
    dec = tok.batch_decode(out, skip_special_tokens=False)
    n_out_tok = int((out[:, 1:] != tok.pad_token_id).sum().item())
    return [extract(x) for x in dec], n_out_tok


def eval_format(model, tok, val, mode, bs=32, max_new=16):
    texts = [build_input(r.masked_docstring, r.code_context, tok, mode) for r in val.itertuples()]
    refs = [str(r.target_span) for r in val.itertuples()]
    preds = []
    t0 = time.time()
    for i in range(0, len(texts), bs):
        p, _ = gen_batch(model, tok, texts[i:i + bs], max_new=max_new)
        preds.extend(p)
    dt = time.time() - t0
    return score_lists(preds, refs), dt, preds, refs


def run_zeroshot():
    print("=== ZERO-SHOT FORMAT SWEEP (codet5-small) ===", flush=True)
    val = load_val(1500, seed=1)  # SAME subset C2 used
    print(f"eval rows: {len(val)} (bucket-0 sample, seed=1 == C2)", flush=True)
    tok, model = load_tok_model()
    results = {}
    for mode in ["plain", "code_first", "doc_first"]:
        chrf, dt, preds, refs = eval_format(model, tok, val, mode)
        results[mode] = chrf
        print(f"  {mode:12s} chrF={chrf:.4f}  {dt:.1f}s  {len(val)/dt:.1f} rows/s", flush=True)
        if mode == "doc_first":
            samp = pd.DataFrame({"masked": val.masked_docstring, "pred": preds, "ref": refs})
            samp["f"] = [f_pooled(p, r) for p, r in zip(preds, refs)]
            samp.head(60).to_csv(os.path.join(HERE, "zeroshot_samples_docfirst.csv"), index=False)
    with open(os.path.join(HERE, "zeroshot_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("zeroshot json saved", flush=True)
    return results


def run_speed():
    print("=== INFERENCE SPEED (codet5-small, greedy, threads=7) ===", flush=True)
    val = load_val(1024, seed=3)  # SAME subset C2 used
    tok, model = load_tok_model()
    texts = [build_input(r.masked_docstring, r.code_context, tok, "code_first") for r in val.itertuples()]
    refs = [str(r.target_span) for r in val.itertuples()]
    gen_batch(model, tok, texts[:16])  # warmup
    speed_rows = {}
    preds = []
    for bs in [16, 32, 64]:
        t0 = time.time(); tot_tok = 0; preds = []
        for i in range(0, len(texts), bs):
            p, nt = gen_batch(model, tok, texts[i:i + bs], max_new=16, in_len=128)
            preds.extend(p); tot_tok += nt
        dt = time.time() - t0; rps = len(texts) / dt
        speed_rows[bs] = {"rows_per_s": rps, "tok_per_s": tot_tok / dt, "min_50k": 50000 / rps / 60}
        print(f"  fp32 bs={bs:3d}: {rps:6.1f} rows/s  {tot_tok/dt:7.1f} tok/s  50k->{50000/rps/60:5.1f} min", flush=True)
    fp32_chrf = score_lists(preds, refs)
    print(f"  fp32 chrF (zero-shot code_first, {len(refs)} rows): {fp32_chrf:.4f}", flush=True)

    print("\n  --- dynamic int8 quantization ---", flush=True)
    qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()
    gen_batch(qmodel, tok, texts[:16])
    q_speed = {}; qpreds = []
    for bs in [16, 32, 64]:
        t0 = time.time(); tot_tok = 0; qpreds = []
        for i in range(0, len(texts), bs):
            p, nt = gen_batch(qmodel, tok, texts[i:i + bs], max_new=16, in_len=128)
            qpreds.extend(p); tot_tok += nt
        dt = time.time() - t0; rps = len(texts) / dt
        q_speed[bs] = {"rows_per_s": rps, "min_50k": 50000 / rps / 60}
        print(f"  int8 bs={bs:3d}: {rps:6.1f} rows/s  {tot_tok/dt:7.1f} tok/s  50k->{50000/rps/60:5.1f} min", flush=True)
    q_chrf = score_lists(qpreds, refs)
    print(f"  int8 chrF (same subset): {q_chrf:.4f}  (delta {q_chrf-fp32_chrf:+.4f})", flush=True)
    out = {"fp32": speed_rows, "fp32_chrf": fp32_chrf, "int8": q_speed, "int8_chrf": q_chrf}
    with open(os.path.join(HERE, "speed_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def run_dump(n=500, ckpt=None, mode="doc_first"):
    """Dump codet5 preds + seq logprob for union-oracle analysis.
    Uses seed=7 (SAME 500 rows as C2 t5_dump) so joins are apples-to-apples."""
    print(f"=== INTEGRATION DUMP ({n} rows, mode={mode}, ckpt={ckpt}) ===", flush=True)
    val = load_val(n, seed=7)
    src = ckpt if ckpt else MODEL
    tok, model = load_tok_model(src)
    texts = [build_input(r.masked_docstring, r.code_context, tok, mode) for r in val.itertuples()]
    refs = [str(r.target_span) for r in val.itertuples()]
    preds, seq_scores = [], []
    bs = 32
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=16, num_beams=1, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        dec = tok.batch_decode(out.sequences, skip_special_tokens=False)
        preds.extend([extract(x) for x in dec])
        ts = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        gen = out.sequences[:, 1:]
        mask = (gen != tok.pad_token_id)
        lp = (ts * mask).sum(dim=1); ln = mask.sum(dim=1).clamp(min=1)
        seq_scores.extend((lp / ln).tolist())
    fs = [f_pooled(p, r) for p, r in zip(preds, refs)]
    df = pd.DataFrame({"id": val.id, "masked_docstring": val.masked_docstring,
                       "ct5_pred": preds, "target_span": refs,
                       "ct5_seq_logprob": seq_scores, "ct5_f": fs})
    tag = "ft" if ckpt else "zs"
    df.to_csv(os.path.join(HERE, f"ct5_dump_{tag}_{n}.csv"), index=False)
    print(f"  chrF on dump: {np.mean(fs):.4f}", flush=True)
    print(f"  seq_logprob vs f corr: {np.corrcoef(seq_scores, fs)[0,1]:.3f}", flush=True)
    print(f"  saved {HERE}/ct5_dump_{tag}_{n}.csv", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if cmd in ("zeroshot", "all"):
        run_zeroshot()
    if cmd in ("speed", "all"):
        run_speed()
    if cmd == "dump":
        ck = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "none" else None
        md = sys.argv[3] if len(sys.argv) > 3 else "doc_first"
        run_dump(500, ckpt=ck, mode=md)
    print(f"\ntotal {time.time()-t0:.1f}s", flush=True)
