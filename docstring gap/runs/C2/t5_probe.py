"""C2 T5 feasibility probe: zero-shot chrF (format sweep + flan), inference speed
(batch sweep + dynamic int8 quantization), 50k-row extrapolation, and an
integration dump (T5 preds + sequence scores) for union-oracle analysis.

Runtime-fair with sibling agent: torch threads 7, run under nice -n 10.
Reads CSVs with keep_default_na=False. Evaluates only on bucket 0 (validation);
never touches bucket 1 (locked holdout).

Usage:
  python runs/C2/t5_probe.py zeroshot   # format sweep + flan reference
  python runs/C2/t5_probe.py speed      # batch sweep + quantization
  python runs/C2/t5_probe.py dump       # 500-row integration dump
  python runs/C2/t5_probe.py all
"""
import sys, os, time, re, hashlib, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "7")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(7)
sys.path.insert(0, "solution")
from chrf import f_pooled, score_lists
from transformers import T5TokenizerFast, T5ForConditionalGeneration

GAP = "[GAP]"
HERE = os.path.join("runs", "C2")


def bucket(s):
    return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 20


def code_hint(code):
    """def line + last return line, compact."""
    lines = code.split("\n")
    def_line, ret_line = "", ""
    for ln in lines:
        s = ln.strip()
        if not def_line and s.startswith("def "):
            def_line = s
        if s.startswith("return ") or s == "return" or s.startswith("return("):
            ret_line = s
    if not def_line:
        for ln in lines:
            if ln.strip():
                def_line = ln.strip()
                break
    parts = [p for p in [def_line, ret_line] if p]
    return " ".join(parts)


def build_input(masked, code, tok, mode):
    sent = masked.replace(GAP, "<extra_id_0>")
    if mode == "plain":
        return sent
    hint = code_hint(code)
    hid = tok(hint, add_special_tokens=False, truncation=True, max_length=64).input_ids
    hint = tok.decode(hid)
    if mode == "code_first":
        return hint + " . " + sent
    else:  # doc_first
        return sent + " . " + hint


_EXTRACT = re.compile(r"<extra_id_0>(.*?)(?:<extra_id_1>|<extra_id_\d|</s>|<pad>|$)", re.DOTALL)


def extract(text):
    m = _EXTRACT.search(text)
    if m:
        return m.group(1).strip()
    t = re.sub(r"</?s>|<pad>|<extra_id_\d+>", " ", text)
    return re.sub(r"\s+", " ", t).strip()


def load_val(n=None, seed=1):
    d = pd.read_csv("dataset/train.csv", keep_default_na=False)
    b = d.masked_docstring.map(bucket)
    val = d[b == 0].reset_index(drop=True)
    if n:
        val = val.sample(n, random_state=seed).reset_index(drop=True)
    return val


def gen_batch(model, tok, texts, max_new=16, in_len=128, quant=False):
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
    chrf = score_lists(preds, refs)
    return chrf, dt, preds, refs


def run_zeroshot():
    print("=== ZERO-SHOT FORMAT SWEEP (t5-small) ===", flush=True)
    val = load_val(1500, seed=1)
    print(f"eval rows: {len(val)} (bucket-0 sample)", flush=True)
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").eval()
    results = {}
    for mode in ["plain", "code_first", "doc_first"]:
        chrf, dt, preds, refs = eval_format(model, tok, val, mode)
        rate = len(val) / dt
        results[mode] = chrf
        print(f"  {mode:12s} chrF={chrf:.4f}  {dt:.1f}s  {rate:.1f} rows/s", flush=True)
        # save a few samples
        if mode == "plain":
            samp = pd.DataFrame({"masked": val.masked_docstring, "pred": preds, "ref": refs})
            samp["f"] = [f_pooled(p, r) for p, r in zip(preds, refs)]
            samp.head(60).to_csv(os.path.join(HERE, "zeroshot_samples_plain.csv"), index=False)

    print("\n=== FLAN-T5-SMALL zero-shot reference (500 rows) ===", flush=True)
    val500 = load_val(500, seed=2)
    try:
        ftok = T5TokenizerFast.from_pretrained("google/flan-t5-small")
        fmodel = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").eval()
        # flan is instruction-tuned; span-corruption sentinels less native. Try plain + an instruction format.
        for mode in ["plain", "code_first"]:
            chrf, dt, _, _ = eval_format(fmodel, ftok, val500, mode)
            print(f"  flan {mode:12s} chrF={chrf:.4f}  {dt:.1f}s", flush=True)
        # instruction-style prompt for flan
        texts = [f"Fill in the blank marked ___ in this sentence: "
                 + r.masked_docstring.replace(GAP, "___") for r in val500.itertuples()]
        refs = [str(r.target_span) for r in val500.itertuples()]
        preds = []
        for i in range(0, len(texts), 32):
            enc = ftok(texts[i:i+32], return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                out = fmodel.generate(**enc, max_new_tokens=16, num_beams=1, do_sample=False)
            preds.extend(ftok.batch_decode(out, skip_special_tokens=True))
        print(f"  flan instruct    chrF={score_lists([p.strip() for p in preds], refs):.4f}", flush=True)
        results["flan_plain"] = None
    except Exception as e:
        print("  flan failed:", repr(e), flush=True)
    with open(os.path.join(HERE, "zeroshot_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_speed():
    print("=== INFERENCE SPEED (t5-small, greedy, threads=7) ===", flush=True)
    val = load_val(1024, seed=3)
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").eval()
    texts = [build_input(r.masked_docstring, r.code_context, tok, "code_first") for r in val.itertuples()]
    refs = [str(r.target_span) for r in val.itertuples()]

    # warmup
    gen_batch(model, tok, texts[:16])
    speed_rows = {}
    for bs in [16, 32, 64]:
        t0 = time.time()
        tot_tok = 0
        preds = []
        for i in range(0, len(texts), bs):
            p, nt = gen_batch(model, tok, texts[i:i + bs], max_new=16, in_len=128)
            preds.extend(p)
            tot_tok += nt
        dt = time.time() - t0
        rps = len(texts) / dt
        tps = tot_tok / dt
        speed_rows[bs] = {"rows_per_s": rps, "tok_per_s": tps, "min_50k": 50000 / rps / 60}
        print(f"  fp32 bs={bs:3d}: {rps:6.1f} rows/s  {tps:7.1f} tok/s  50k->{50000/rps/60:5.1f} min", flush=True)
    fp32_chrf = score_lists(preds, refs)  # last-batch-size preds (identical greedy)
    print(f"  fp32 chrF (zero-shot, code_first, {len(refs)} rows): {fp32_chrf:.4f}", flush=True)

    # dynamic int8 quantization
    print("\n  --- dynamic int8 quantization ---", flush=True)
    qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()
    gen_batch(qmodel, tok, texts[:16])
    q_speed = {}
    for bs in [16, 32, 64]:
        t0 = time.time()
        tot_tok = 0
        qpreds = []
        for i in range(0, len(texts), bs):
            p, nt = gen_batch(qmodel, tok, texts[i:i + bs], max_new=16, in_len=128)
            qpreds.extend(p)
            tot_tok += nt
        dt = time.time() - t0
        rps = len(texts) / dt
        q_speed[bs] = {"rows_per_s": rps, "min_50k": 50000 / rps / 60}
        print(f"  int8 bs={bs:3d}: {rps:6.1f} rows/s  {tot_tok/dt:7.1f} tok/s  50k->{50000/rps/60:5.1f} min", flush=True)
    q_chrf = score_lists(qpreds, refs)
    print(f"  int8 chrF (same subset): {q_chrf:.4f}  (delta {q_chrf-fp32_chrf:+.4f})", flush=True)

    out = {"fp32": speed_rows, "fp32_chrf": fp32_chrf, "int8": q_speed, "int8_chrf": q_chrf}
    with open(os.path.join(HERE, "speed_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def run_dump(n=500, ckpt=None, mode="code_first"):
    """Dump T5 preds + sequence logprob scores for union-oracle analysis.
    mode MUST match the checkpoint's training input format (ft_main = plain)."""
    print(f"=== INTEGRATION DUMP ({n} rows, mode={mode}, ckpt={ckpt}) ===", flush=True)
    val = load_val(n, seed=7)
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    src = ckpt if ckpt else "google-t5/t5-small"
    model = T5ForConditionalGeneration.from_pretrained(src).eval()
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
        lp = (ts * mask).sum(dim=1)
        ln = mask.sum(dim=1).clamp(min=1)
        seq_scores.extend((lp / ln).tolist())
    fs = [f_pooled(p, r) for p, r in zip(preds, refs)]
    df = pd.DataFrame({"id": val.id, "masked_docstring": val.masked_docstring,
                       "t5_pred": preds, "target_span": refs,
                       "t5_seq_logprob": seq_scores, "t5_f": fs})
    tag = "ft" if ckpt else "zs"
    df.to_csv(os.path.join(HERE, f"t5_dump_{tag}_{n}.csv"), index=False)
    print(f"  chrF on dump: {np.mean(fs):.4f}", flush=True)
    print(f"  seq_logprob vs f corr: {np.corrcoef(seq_scores, fs)[0,1]:.3f}", flush=True)
    print(f"  saved {HERE}/t5_dump_{tag}_{n}.csv", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if mode in ("zeroshot", "all"):
        run_zeroshot()
    if mode in ("speed", "all"):
        run_speed()
    if mode in ("dump", "all"):
        run_dump(500)
    print(f"\ntotal {time.time()-t0:.1f}s", flush=True)
