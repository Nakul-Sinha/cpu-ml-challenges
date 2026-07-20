"""Canonical structured decoding + metric-driven tuning on top of per-node
class probabilities. Builders produce probs; this module turns them into
sequences and tunes class multipliers against the OFFICIAL metric on OOF.
"""
import collections
import numpy as np

from common import TYPES

NT = len(TYPES)


def fit_transitions(targets_by_id, ids, alpha=1.0):
    """Position-conditioned transition tables with global fallback.
    Returns dict with log-prob lookup fn."""
    trans = collections.defaultdict(collections.Counter)
    trans_g = collections.defaultdict(collections.Counter)
    init = collections.defaultdict(collections.Counter)
    for sid in ids:
        ts = targets_by_id[sid]
        L = len(ts)
        init[L][ts[0]] += 1
        for i in range(1, L):
            trans[(ts[i - 1], i, L)][ts[i]] += 1
            trans_g[ts[i - 1]][ts[i]] += 1

    def logp(counter, key):
        tot = sum(counter.values())
        return np.log((counter[key] + alpha) / (tot + alpha * NT))

    def log_trans(prev_t, i, L):
        c = trans[(prev_t, i, L)]
        out = np.empty(NT)
        for ci, ct in enumerate(TYPES):
            if sum(c.values()) >= 8:
                out[ci] = logp(c, ct)
            else:
                out[ci] = 0.5 * logp(c, ct) + 0.5 * logp(trans_g[prev_t], ct)
        return out

    def log_init(L):
        return np.array([logp(init[L], t) for t in TYPES])

    return dict(log_trans=log_trans, log_init=log_init)


def role_of(i, L):
    return 0 if i == 0 else (2 if i == L - 1 else 1)


def decode_row(probs, tables, w_emis=1.0, mults=None, mode='viterbi'):
    """probs: (L, 5) per-node class probabilities (already blended).
    mults: (3, 5) log-space class multipliers per role (first/mid/anchor).
    Returns list of type strings."""
    L = probs.shape[0]
    E = np.log(np.clip(probs, 1e-9, 1.0)) * w_emis
    if mults is not None:
        for i in range(L):
            E[i] += mults[role_of(i, L)]
    lt = [None] + [tables['log_trans_cache'][(tp, i, L)] if 'log_trans_cache' in tables else None
                   for i in range(1, L)]
    if mode == 'viterbi':
        dp = tables['log_init'](L) + E[0]
        bp = np.zeros((L, NT), int)
        for i in range(1, L):
            M = np.empty((NT, NT))
            for pi, pt in enumerate(TYPES):
                M[pi] = dp[pi] + tables['log_trans'](pt, i, L) + E[i]
            bp[i] = M.argmax(0)
            dp = M.max(0)
        seq = [int(dp.argmax())]
        for i in range(L - 1, 0, -1):
            seq.append(int(bp[i][seq[-1]]))
        seq.reverse()
    else:  # posterior max-marginal via forward-backward
        A = np.zeros((L, NT))
        A[0] = tables['log_init'](L) + E[0]
        Tm = [None] * L
        for i in range(1, L):
            Tm[i] = np.stack([tables['log_trans'](pt, i, L) for pt in TYPES])  # (prev, cur)
            A[i] = E[i] + _lse(A[i - 1][:, None] + Tm[i], axis=0)
        B = np.zeros((L, NT))
        for i in range(L - 2, -1, -1):
            B[i] = _lse(Tm[i + 1] + (E[i + 1] + B[i + 1])[None, :], axis=1)
        seq = list((A + B).argmax(1))
    return [TYPES[k] for k in seq]


def _lse(x, axis):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).reshape(
        [s for a, s in enumerate(x.shape) if a != axis])


def decode_all(probs_by_row, tables, mults=None, mode='viterbi', w_emis=1.0):
    return {sid: decode_row(p, tables, w_emis=w_emis, mults=mults, mode=mode)
            for sid, p in probs_by_row.items()}


def tune_multipliers(probs_by_row, ids, pars_by_id, truth_df, score_fn,
                     mode='viterbi', w_emis=1.0, n_passes=2,
                     grid=(-0.6, -0.4, -0.25, -0.12, 0.0, 0.12, 0.25, 0.4, 0.6),
                     targets_by_id=None, fold_of=None, verbose=True):
    """Coordinate ascent on (3 roles x 5 classes) log-multipliers maximizing the
    OFFICIAL score on OOF. Transitions must be fit per-fold outside; here we
    take a fn sid->tables via fold_of to avoid leakage.
    score_fn(type_seqs_by_id) -> float official score.
    """
    mults = np.zeros((3, NT))
    best = score_fn(_decoded(probs_by_row, ids, mults, mode, w_emis, fold_of))
    if verbose:
        print(f'[tune] start score={best:.4f}')
    for p in range(n_passes):
        for r in range(3):
            for c in range(NT):
                base = mults[r, c]
                cand_best, cand_val = base, best
                for g in grid:
                    if g == 0.0 and base == 0.0:
                        continue
                    mults[r, c] = base + g
                    s = score_fn(_decoded(probs_by_row, ids, mults, mode, w_emis, fold_of))
                    if s > cand_val:
                        cand_best, cand_val = mults[r, c], s
                mults[r, c] = cand_best
                best = cand_val
        if verbose:
            print(f'[tune] pass {p + 1}: score={best:.4f}')
    return mults, best


def _decoded(probs_by_row, ids, mults, mode, w_emis, fold_of):
    out = {}
    for sid in ids:
        tables = fold_of(sid)
        out[sid] = decode_row(probs_by_row[sid], tables, w_emis=w_emis,
                              mults=mults, mode=mode)
    return out
