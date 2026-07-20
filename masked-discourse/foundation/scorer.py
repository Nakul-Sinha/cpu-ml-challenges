"""Exact implementation of the official metric.

Score = 100 * clip(0.45*TypeMacroF1 + 0.25*AnchorMacroF1 + 0.10*TypeScore
                   + 0.15*OrderedScore + 0.05*ParentScore, 0, 1)
"""
import re
import numpy as np

VALID_TYPES = {'type_answer', 'type_elaboration', 'type_question',
               'type_appreciation', 'type_agreement'}
TYPE_CLASSES = ['answer', 'elaboration', 'question', 'appreciation', 'agreement']
PARENT_RE = re.compile(r'^parent_(root|n\d\d)$')
TOKEN_RE = re.compile(r'[a-z0-9_]+')


def _norm_tokens(s):
    return TOKEN_RE.findall(str(s).lower())[:300]


def _parse_pred(s):
    """Return (valid, type_list, parent_list, seq_tokens). Invalid rows get no
    type/parent/ordered credit but we still expose any valid type tokens for
    the anchor read."""
    toks = _norm_tokens(s)
    valid = len(toks) % 2 == 0 and len(toks) > 0
    if valid:
        for i, t in enumerate(toks):
            if i % 2 == 0 and t not in VALID_TYPES:
                valid = False
                break
            if i % 2 == 1 and not PARENT_RE.match(t):
                valid = False
                break
    if len(toks) == 0:
        valid = False
    types_any = [t for t in toks if t in VALID_TYPES]
    if valid:
        return True, toks[0::2], toks[1::2], toks
    return False, [], [], []


def _lcs(a, b):
    if not a or not b:
        return 0
    la, lb = len(a), len(b)
    prev = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur = [0] * (lb + 1)
        ai = a[i - 1]
        for j in range(1, lb + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return prev[lb]


def _macro_f1(tp, fp, fn):
    f1s = []
    for c in TYPE_CLASSES:
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append(2 * tp[c] / denom if denom > 0 else 0.0)
    return float(np.mean(f1s)), {c: (2 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) if (2 * tp[c] + fp[c] + fn[c]) > 0 else 0.0) for c in TYPE_CLASSES}


def score_submission(sub_df, truth_df, verbose=False):
    """sub_df/truth_df: DataFrames with sample_id, target_sequence.
    Returns (score, components dict)."""
    truth = dict(zip(truth_df['sample_id'], truth_df['target_sequence']))
    assert set(sub_df['sample_id']) == set(truth), 'id mismatch'
    assert len(sub_df) == len(sub_df['sample_id'].unique())

    type_pos, par_pos, ordered = [], [], []
    tp = {c: 0 for c in TYPE_CLASSES}; fp = {c: 0 for c in TYPE_CLASSES}; fn = {c: 0 for c in TYPE_CLASSES}
    atp = {c: 0 for c in TYPE_CLASSES}; afp = {c: 0 for c in TYPE_CLASSES}; afn = {c: 0 for c in TYPE_CLASSES}

    for _, row in sub_df.iterrows():
        g_valid, g_t, g_p, g_s = _parse_pred(truth[row['sample_id']])
        assert g_valid, f'ground truth invalid for {row["sample_id"]}'
        p_valid, p_t, p_p, p_s = _parse_pred(row['target_sequence'])

        if p_valid:
            m = sum(1 for a, b in zip(p_t, g_t) if a == b)
            type_pos.append(m / max(len(p_t), len(g_t), 1))
            m = sum(1 for a, b in zip(p_p, g_p) if a == b)
            par_pos.append(m / max(len(p_p), len(g_p), 1))
            l = _lcs(p_s, g_s)
            if len(p_s) == 0 and len(g_s) == 0:
                ordered.append(1.0)
            elif len(p_s) == 0 or len(g_s) == 0:
                ordered.append(0.0)
            else:
                pr, rc = l / len(p_s), l / len(g_s)
                ordered.append(2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0)
        else:
            type_pos.append(0.0); par_pos.append(0.0); ordered.append(0.0)

        # pooled type macro-F1 over true masked positions
        for j, g in enumerate(g_t):
            gc = g[len('type_'):]
            if p_valid and j < len(p_t):
                pc = p_t[j][len('type_'):]
                if pc == gc:
                    tp[gc] += 1
                else:
                    fp[pc] += 1; fn[gc] += 1
            else:
                fn[gc] += 1
        if p_valid:
            for j in range(len(g_t), len(p_t)):
                fp[p_t[j][len('type_'):]] += 1

        # anchor at true position K-1
        K = len(g_t)
        gc = g_t[K - 1][len('type_'):]
        if p_valid and K - 1 < len(p_t):
            pc = p_t[K - 1][len('type_'):]
            if pc == gc:
                atp[gc] += 1
            else:
                afp[pc] += 1; afn[gc] += 1
        else:
            afn[gc] += 1

    type_macro, type_percls = _macro_f1(tp, fp, fn)
    anchor_macro, anchor_percls = _macro_f1(atp, afp, afn)
    comps = dict(
        TypeMacroF1=type_macro, AnchorMacroF1=anchor_macro,
        TypeScore=float(np.mean(type_pos)), OrderedScore=float(np.mean(ordered)),
        ParentScore=float(np.mean(par_pos)),
        type_percls=type_percls, anchor_percls=anchor_percls,
    )
    score = 100 * float(np.clip(
        0.45 * comps['TypeMacroF1'] + 0.25 * comps['AnchorMacroF1'] +
        0.10 * comps['TypeScore'] + 0.15 * comps['OrderedScore'] +
        0.05 * comps['ParentScore'], 0, 1))
    if verbose:
        print(f"Score={score:.4f}  TypeMacroF1={type_macro:.4f} AnchorMacroF1={anchor_macro:.4f} "
              f"TypeScore={comps['TypeScore']:.4f} OrderedScore={comps['OrderedScore']:.4f} ParentScore={comps['ParentScore']:.4f}")
        print('  type per-class F1:', {k: round(v, 3) for k, v in type_percls.items()})
        print('  anchor per-class F1:', {k: round(v, 3) for k, v in anchor_percls.items()})
    return score, comps


if __name__ == '__main__':
    import pandas as pd, sys
    sub = pd.read_csv(sys.argv[1])
    truth = pd.read_csv(sys.argv[2])
    score_submission(sub, truth, verbose=True)
