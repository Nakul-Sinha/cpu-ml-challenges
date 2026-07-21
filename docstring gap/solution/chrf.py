"""Canonical scorer for Docstring Gap Restoration.

Spec: character n-gram F-score, n=1..6: precision and recall over character
n-grams from length 1 through 6, then F = 2PR/(P+R). Final = mean over rows.
Primary implementation pools n-gram counts across n=1..6 (single P, R over the
union of all n-gram multisets), with clipped matches. chrF-style macro variant
included for sensitivity checks. All agents import THIS module.
"""
from collections import Counter


def _ngrams(s, n):
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def f_pooled(pred, ref, nmax=6):
    pred, ref = str(pred), str(ref)
    if len(pred) == 0 and len(ref) == 0:
        return 1.0
    match = 0
    tp = 0
    tr = 0
    for n in range(1, nmax + 1):
        cp = _ngrams(pred, n)
        cr = _ngrams(ref, n)
        tp += sum(cp.values())
        tr += sum(cr.values())
        match += sum(min(v, cr[k]) for k, v in cp.items())
    if tp == 0 or tr == 0 or match == 0:
        return 0.0
    p = match / tp
    r = match / tr
    return 2 * p * r / (p + r)


def f_macro(pred, ref, nmax=6):
    pred, ref = str(pred), str(ref)
    if len(pred) == 0 and len(ref) == 0:
        return 1.0
    vals = []
    for n in range(1, nmax + 1):
        cp = _ngrams(pred, n)
        cr = _ngrams(ref, n)
        tp, tr = sum(cp.values()), sum(cr.values())
        if tp == 0 and tr == 0:
            continue
        if tp == 0 or tr == 0:
            vals.append(0.0)
            continue
        m = sum(min(v, cr[k]) for k, v in cp.items())
        p, r = m / tp, m / tr
        vals.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return sum(vals) / len(vals) if vals else 0.0


def score_lists(preds, refs, mode="pooled"):
    fn = f_pooled if mode == "pooled" else f_macro
    assert len(preds) == len(refs)
    return sum(fn(p, r) for p, r in zip(preds, refs)) / len(refs)
