"""Reviewer stage: blend builder OOF probs, pick decode mode, tune class
multipliers against the OFFICIAL metric with an honest nested protocol, freeze
the recipe, and emit a test submission.

Run on the box:  ~/venv/bin/python reviewer_blend.py runs/gbm runs/textlin runs/nnseq
"""
import sys, os, json, itertools, collections
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TYPES, parse_target, make_folds, format_submission, solve_parents
from scorer import score_submission
from decode import fit_transitions
from fast_score import FastScorer

NT = 5
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'public')


def load_ctx():
    train = pd.read_csv(os.path.join(ROOT, 'train.csv'))
    test = pd.read_csv(os.path.join(ROOT, 'test.csv'))
    ids = train['sample_id'].tolist()
    lens = np.array([len(r.split()) for r in train['masked_nodes']])
    targets = {r['sample_id']: parse_target(r['target_sequence'])[0] for _, r in train.iterrows()}
    folds = make_folds(train)
    pars = {r['sample_id']: solve_parents(r) for _, r in train.iterrows()}
    test_ids = test['sample_id'].tolist()
    test_lens = np.array([len(r.split()) for r in test['masked_nodes']])
    test_pars = {r['sample_id']: solve_parents(r) for _, r in test.iterrows()}
    truth = train[['sample_id', 'target_sequence']]
    return train, test, ids, lens, targets, folds, pars, test_ids, test_lens, test_pars, truth


def tables_to_arrays(tables, Lmax=4):
    """T[L][i] = (5,5) prev->cur log-trans; I[L] = (5,) init."""
    T, I = {}, {}
    for L in (3, 4):
        I[L] = tables['log_init'](L)
        T[L] = [None] * L
        for i in range(1, L):
            T[L][i] = np.stack([tables['log_trans'](pt, i, L) for pt in TYPES])
    return T, I


def batch_viterbi(E, T, I):
    """E: (R, L, 5) log-emissions (already multiplied/mult-adjusted). Returns (R, L) int."""
    R, L, _ = E.shape
    dp = I[None, :] + E[:, 0]
    bps = np.zeros((R, L, NT), dtype=np.int8)
    for i in range(1, L):
        M = dp[:, :, None] + T[i][None, :, :] + E[:, i][:, None, :]
        bps[:, i] = M.argmax(1)
        dp = M.max(1)
    out = np.zeros((R, L), dtype=np.int8)
    out[:, L - 1] = dp.argmax(1)
    for i in range(L - 1, 0, -1):
        out[:, i - 1] = bps[np.arange(R), i, out[:, i]]
    return out


def batch_posterior(E, T, I):
    R, L, _ = E.shape
    A = np.zeros((R, L, NT)); A[:, 0] = I[None, :] + E[:, 0]
    for i in range(1, L):
        prev = A[:, i - 1][:, :, None] + T[i][None, :, :]
        m = prev.max(1)
        A[:, i] = E[:, i] + m + np.log(np.exp(prev - m[:, None, :]).sum(1))
    B = np.zeros((R, L, NT))
    for i in range(L - 2, -1, -1):
        nxt = T[i + 1][None, :, :] + (E[:, i + 1] + B[:, i + 1])[:, None, :]
        m = nxt.max(2)
        B[:, i] = m + np.log(np.exp(nxt - m[:, :, None]).sum(2))
    return (A + B).argmax(2)


class Decoder:
    def __init__(self, ids, lens, folds, targets, n_folds=5):
        self.ids, self.lens, self.folds = ids, lens, folds
        self.offs = np.concatenate([[0], np.cumsum(lens)])
        self.TA = {}
        for f in range(n_folds):
            tr = [s for s, fo in zip(ids, folds) if fo != f]
            self.TA[f] = tables_to_arrays(fit_transitions(targets, tr))
        self.TA['full'] = tables_to_arrays(fit_transitions(targets, list(targets.keys())))

    def decode(self, P, mults=None, mode='viterbi', w_emis=1.0, use_fold_tables=True,
               ext_lens=None, ext_folds=None):
        lens = self.lens if ext_lens is None else ext_lens
        folds = self.folds if ext_folds is None else ext_folds
        offs = np.concatenate([[0], np.cumsum(lens)])
        E_all = np.log(np.clip(P, 1e-9, 1.0)) * w_emis
        n = len(lens)
        seqs = [None] * n
        fold_keys = sorted(set(folds)) if use_fold_tables else ['full']
        for fk in fold_keys:
            for L in (3, 4):
                idx = [r for r in range(n) if lens[r] == L and (not use_fold_tables or folds[r] == fk)]
                if not idx:
                    continue
                E = np.stack([E_all[offs[r]:offs[r] + L] for r in idx])
                if mults is not None:
                    E = E + mults_for(L, mults)[None, :, :]
                T, I = self.TA[fk if use_fold_tables else 'full']
                dec = batch_viterbi(E, T[L], I[L]) if mode == 'viterbi' else batch_posterior(E, T[L], I[L])
                for k, r in enumerate(idx):
                    seqs[r] = [TYPES[c] for c in dec[k]]
        return seqs


def mults_for(L, mults):
    """mults: (3,5) role x class -> (L,5)"""
    roles = [0] + [1] * (L - 2) + [2]
    return mults[roles]


def main():
    run_dirs = sys.argv[1:]
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    train, test, ids, lens, targets, folds, pars, test_ids, test_lens, test_pars, truth = load_ctx()
    dec = Decoder(ids, lens, folds, targets)

    P_list, names = [], []
    for rd in run_dirs:
        p = os.path.join(base, rd, 'oof_probs.npy')
        if os.path.exists(p):
            P = np.load(p)
            assert P.shape == (int(lens.sum()), NT), (rd, P.shape)
            P_list.append(np.clip(P, 1e-9, 1)); names.append(rd)
        else:
            print(f'[skip] {p} missing')
    print('members:', names)

    true_types = [[TYPES.index(t) for t in targets[s]] for s in ids]
    true_pars = []
    for _, r in train.iterrows():
        true_pars.append(parse_target(r['target_sequence'])[1])
    pred_pars = [pars[s] for s in ids]
    fs_cache = {}

    def get_fs(key, use_idx):
        if key not in fs_cache:
            fs_cache[key] = (FastScorer([true_types[i] for i in use_idx],
                                        [true_pars[i] for i in use_idx],
                                        [pred_pars[i] for i in use_idx]), use_idx)
        return fs_cache[key]

    def score_seqs(seqs, row_mask=None):
        key = 'full' if row_mask is None else row_mask.tobytes()
        use_idx = list(range(len(ids))) if row_mask is None else list(np.where(row_mask)[0])
        fs, uidx = get_fs(key, use_idx)
        preds = [np.array([TYPES.index(t) for t in seqs[i]]) for i in uidx]
        return fs.score(preds)

    # ---- blend weight search (geometric mean) ----
    logs = [np.log(P) for P in P_list]
    best = None
    m = len(P_list)
    steps = [w / 8 for w in range(9)]
    for ws in itertools.product(steps, repeat=m):
        if abs(sum(ws) - 1.0) > 1e-9:
            continue
        Pb = np.exp(sum(w * l for w, l in zip(ws, logs)))
        Pb /= Pb.sum(1, keepdims=True)
        for mode in ('viterbi', 'posterior'):
            s, _ = score_seqs(dec.decode(Pb, mode=mode))
            if best is None or s > best[0]:
                best = (s, ws, mode)
                print(f'  blend {ws} {mode}: {s:.4f} *')
    s0, ws, mode = best
    print(f'[blend] best {ws} mode={mode} -> {s0:.4f}')
    Pb = np.exp(sum(w * l for w, l in zip(ws, logs)))
    Pb /= Pb.sum(1, keepdims=True)

    # ---- emission temperature ----
    best_w = (s0, 1.0)
    for w_emis in (0.7, 0.85, 1.0, 1.2, 1.5):
        s, _ = score_seqs(dec.decode(Pb, mode=mode, w_emis=w_emis))
        print(f'  w_emis={w_emis}: {s:.4f}')
        if s > best_w[0]:
            best_w = (s, w_emis)
    s0, w_emis = best_w
    print(f'[temp] w_emis={w_emis} -> {s0:.4f}')

    # ---- multiplier coordinate ascent ----
    GRID = (-0.5, -0.35, -0.2, -0.1, 0.1, 0.2, 0.35, 0.5)

    def tune(row_mask, init=None, passes=2):
        mults = np.zeros((3, NT)) if init is None else init.copy()
        cur, _ = score_seqs(dec.decode(Pb, mults=mults, mode=mode, w_emis=w_emis), row_mask)
        for _ in range(passes):
            for r in range(3):
                for c in range(NT):
                    b = mults[r, c]; bb, bv = b, cur
                    for g in GRID:
                        mults[r, c] = b + g
                        s, _ = score_seqs(dec.decode(Pb, mults=mults, mode=mode, w_emis=w_emis), row_mask)
                        if s > bv:
                            bb, bv = mults[r, c], s
                    mults[r, c] = bb; cur = bv
        return mults, cur

    rng = np.random.RandomState(0)
    half = rng.rand(len(ids)) < 0.5
    mA, sA = tune(half); mB, sB = tune(~half)
    sAB, _ = score_seqs(dec.decode(Pb, mults=mA, mode=mode, w_emis=w_emis), ~half)
    sBA, _ = score_seqs(dec.decode(Pb, mults=mB, mode=mode, w_emis=w_emis), half)
    base_A, _ = score_seqs(dec.decode(Pb, mode=mode, w_emis=w_emis), ~half)
    base_B, _ = score_seqs(dec.decode(Pb, mode=mode, w_emis=w_emis), half)
    cross_gain = 0.5 * (sAB - base_A + sBA - base_B)
    print(f'[tune-nested] cross-applied gain: {cross_gain:+.4f} (A->B {sAB - base_A:+.3f}, B->A {sBA - base_B:+.3f})')

    m_full, s_insample = tune(None)
    shrink = 0.7 if cross_gain > 0.15 else (0.4 if cross_gain > 0 else 0.0)
    m_final = m_full * shrink
    s_final, comps = score_seqs(dec.decode(Pb, mults=m_final, mode=mode, w_emis=w_emis))
    print(f'[tune] in-sample tuned={s_insample:.4f}; shrink={shrink}; final OOF={s_final:.4f}')
    print('  comps:', {k: round(v, 4) for k, v in comps.items() if not isinstance(v, dict)})
    print('  type per-class:', {k: round(v, 3) for k, v in comps['type_percls'].items()})
    print('  anchor per-class:', {k: round(v, 3) for k, v in comps['anchor_percls'].items()})

    recipe = dict(members=names, weights=list(ws), mode=mode, w_emis=w_emis,
                  mults=m_final.tolist(), shrink=shrink, oof_score=s_final,
                  cross_gain=cross_gain, blend_score=s0)
    with open(os.path.join(base, 'foundation', 'recipe.json'), 'w') as f:
        json.dump(recipe, f, indent=2)

    # ---- test submission ----
    Pt_list = []
    for rd, w in zip(names, ws):
        Pt = np.clip(np.load(os.path.join(base, rd, 'test_probs.npy')), 1e-9, 1)
        assert Pt.shape == (int(test_lens.sum()), NT), (rd, Pt.shape)
        Pt_list.append((w, np.log(Pt)))
    Pbt = np.exp(sum(w * l for w, l in Pt_list)); Pbt /= Pbt.sum(1, keepdims=True)
    seqs_t = dec.decode(Pbt, mults=m_final, mode=mode, w_emis=w_emis,
                        use_fold_tables=False, ext_lens=test_lens,
                        ext_folds=[0] * len(test_ids))
    sub = format_submission(test_ids, seqs_t, [test_pars[s] for s in test_ids])
    out = os.path.join(base, 'working', 'submission.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sub.to_csv(out, index=False)
    print(f'[test] wrote {out} rows={len(sub)}')
    # strict format sanity
    from scorer import _parse_pred
    assert all(_parse_pred(s)[0] for s in sub['target_sequence'])
    ss = pd.read_csv(os.path.join(ROOT, 'sample_submission.csv'))
    assert list(sub.columns) == list(ss.columns) and set(sub.sample_id) == set(ss.sample_id)
    print('[test] format valid')


if __name__ == '__main__':
    main()
