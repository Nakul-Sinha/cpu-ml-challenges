"""Reviewer v2: blend all members (incl. stack), optional anchor-specialist mix,
then a nested-validated calibration ladder (each knob kept only if honest
cross-applied gain clears a threshold). Emits recipe2.json + submission.

Usage: ~/venv/bin/python reviewer_blend2.py runs/gbm runs/textlin runs/nnseq runs/stack
"""
import sys, os, json, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TYPES, parse_target, make_folds, format_submission, solve_parents
from decode import fit_transitions
from fast_score import FastScorer
from reviewer_blend import Decoder, mults_for, load_ctx, tables_to_arrays

NT = 5
KEEP_GAIN = 0.05


def main():
    run_dirs = sys.argv[1:]
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    train, test, ids, lens, targets, folds, pars, test_ids, test_lens, test_pars, truth = load_ctx()
    dec = Decoder(ids, lens, folds, targets)
    n = len(ids)
    offs = np.concatenate([[0], np.cumsum(lens)])
    anchor_node_idx = offs[1:] - 1                     # last node of each row
    roles_flat = np.zeros(int(lens.sum()), int)        # 0 first, 1 mid, 2 anchor
    for r in range(n):
        roles_flat[offs[r]] = 0
        roles_flat[offs[r] + 1: offs[r + 1] - 1] = 1
        roles_flat[offs[r + 1] - 1] = 2

    true_types = [[TYPES.index(t) for t in targets[s]] for s in ids]
    true_pars = [parse_target(r['target_sequence'])[1] for _, r in train.iterrows()]
    pred_pars = [pars[s] for s in ids]
    fs_cache = {}

    def score_seqs(seqs, row_mask=None):
        key = 'full' if row_mask is None else row_mask.tobytes()
        uidx = list(range(n)) if row_mask is None else list(np.where(row_mask)[0])
        if key not in fs_cache:
            fs_cache[key] = FastScorer([true_types[i] for i in uidx],
                                       [true_pars[i] for i in uidx],
                                       [pred_pars[i] for i in uidx])
        fs = fs_cache[key]
        preds = [np.array([TYPES.index(t) for t in seqs[i]]) for i in uidx]
        return fs.score(preds)

    # ---- load members ----
    P_list, Pt_list, names = [], [], []
    anchor_oof = anchor_test = None
    for rd in run_dirs:
        d = os.path.join(base, rd)
        for oofn, testn in (('oof_probs2.npy', 'test_probs2.npy'), ('oof_probs.npy', 'test_probs.npy')):
            p = os.path.join(d, oofn)
            if os.path.exists(p):
                P_list.append(np.clip(np.load(p), 1e-9, 1))
                Pt_list.append(np.clip(np.load(os.path.join(d, testn)), 1e-9, 1))
                names.append(rd + '/' + oofn)
                break
        ap = os.path.join(d, 'anchor_oof.npy')
        if os.path.exists(ap):
            anchor_oof = np.clip(np.load(ap), 1e-9, 1)
            anchor_test = np.clip(np.load(os.path.join(d, 'anchor_test.npy')), 1e-9, 1)
    print('members:', names, ' anchor_specialist:', anchor_oof is not None)
    logs = [np.log(P) for P in P_list]

    def blend(ws, matrices):
        Pb = np.exp(sum(w * l for w, l in zip(ws, matrices)))
        return Pb / Pb.sum(1, keepdims=True)

    # ---- coarse simplex search (step 1/8) ----
    m = len(P_list)
    best = None
    steps = [w / 8 for w in range(9)]
    for ws in itertools.product(steps, repeat=m):
        if abs(sum(ws) - 1.0) > 1e-9:
            continue
        Pb = blend(ws, logs)
        for mode in ('viterbi', 'posterior'):
            s, _ = score_seqs(dec.decode(Pb, mode=mode))
            if best is None or s > best[0]:
                best = (s, ws, mode)
    s0, ws, mode = best
    print(f'[blend coarse] {ws} {mode} -> {s0:.4f}')
    # local refinement step 1/16
    deltas = [-1 / 16, 0, 1 / 16]
    for _ in range(2):
        improved = False
        for dd in itertools.product(deltas, repeat=m):
            w2 = np.array(ws) + np.array(dd)
            if (w2 < -1e-9).any() or abs(w2.sum() - 1) > 1e-9:
                continue
            Pb = blend(w2, logs)
            s, _ = score_seqs(dec.decode(Pb, mode=mode))
            if s > s0 + 1e-9:
                s0, ws = s, tuple(w2); improved = True
        if not improved:
            break
    print(f'[blend fine] {tuple(round(w, 4) for w in ws)} {mode} -> {s0:.4f}')
    Pb = blend(ws, logs)

    # ---- anchor specialist mix (1 param, nested-checked) ----
    rng = np.random.RandomState(0)
    half = rng.rand(n) < 0.5
    anchor_mix = 0.0
    if anchor_oof is not None:
        def apply_mix(P, A_probs, w):
            P2 = P.copy()
            la = np.log(np.clip(A_probs, 1e-9, 1))
            P2[anchor_node_idx] = np.exp((1 - w) * np.log(P[anchor_node_idx]) + w * la)
            P2 /= P2.sum(1, keepdims=True)
            return P2
        def best_mix(row_mask):
            bw, bs = 0.0, score_seqs(dec.decode(Pb, mode=mode), row_mask)[0]
            for w in (0.25, 0.5, 0.75, 1.0):
                s, _ = score_seqs(dec.decode(apply_mix(Pb, anchor_oof, w), mode=mode), row_mask)
                if s > bs:
                    bw, bs = w, s
            return bw, bs
        wA, _ = best_mix(half); wB, _ = best_mix(~half)
        gA = score_seqs(dec.decode(apply_mix(Pb, anchor_oof, wA), mode=mode), ~half)[0] - \
             score_seqs(dec.decode(Pb, mode=mode), ~half)[0]
        gB = score_seqs(dec.decode(apply_mix(Pb, anchor_oof, wB), mode=mode), half)[0] - \
             score_seqs(dec.decode(Pb, mode=mode), half)[0]
        cross = 0.5 * (gA + gB)
        print(f'[anchor mix] halves w=({wA},{wB}) cross-gain {cross:+.4f}')
        if cross > KEEP_GAIN:
            anchor_mix, s_full = best_mix(None)
            Pb = apply_mix(Pb, anchor_oof, anchor_mix)
            s0, _ = score_seqs(dec.decode(Pb, mode=mode))
            print(f'[anchor mix] kept w={anchor_mix} -> {s0:.4f}')
        else:
            print('[anchor mix] rejected')

    # ---- calibration ladder ----
    # (a) role-wise prior correction: E *= (uniform / OOF-marginal)^beta, single beta
    marg = np.stack([Pb[roles_flat == r].mean(0) for r in range(3)])
    corr = np.log(0.2) - np.log(marg)                  # (3,5)

    def apply_beta(P, beta):
        E = np.log(P) + beta * corr[roles_flat]
        P2 = np.exp(E); P2 /= P2.sum(1, keepdims=True)
        return P2

    def eval_P(P, row_mask=None, mm=None):
        return score_seqs(dec.decode(P, mode=mode, mults=mm), row_mask)[0]

    def best_beta(row_mask):
        bb, bs = 0.0, eval_P(Pb, row_mask)
        for b in (0.25, 0.5, 0.75, 1.0):
            s = eval_P(apply_beta(Pb, b), row_mask)
            if s > bs:
                bb, bs = b, s
        return bb, bs

    bA, _ = best_beta(half); bB, _ = best_beta(~half)
    gA = eval_P(apply_beta(Pb, bA), ~half) - eval_P(Pb, ~half)
    gB = eval_P(apply_beta(Pb, bB), half) - eval_P(Pb, half)
    cross_beta = 0.5 * (gA + gB)
    beta = 0.0
    print(f'[beta] halves=({bA},{bB}) cross-gain {cross_beta:+.4f}')
    if cross_beta > KEEP_GAIN:
        beta, s_new = best_beta(None)
        Pb = apply_beta(Pb, beta)
        s0 = eval_P(Pb)
        print(f'[beta] kept beta={beta} -> {s0:.4f}')
    else:
        print('[beta] rejected')

    # (b) global class multipliers (5 params, small grid)
    GRID = (-0.3, -0.15, 0.15, 0.3)

    def tune_global(row_mask, passes=2):
        v = np.zeros(NT)
        cur = eval_P(Pb, row_mask, np.tile(v, (3, 1)))
        for _ in range(passes):
            for c in range(NT):
                b0, bb, bv = v[c], v[c], cur
                for g in GRID:
                    v[c] = b0 + g
                    s = eval_P(Pb, row_mask, np.tile(v, (3, 1)))
                    if s > bv:
                        bb, bv = v[c], s
                v[c] = bb; cur = bv
        return v, cur

    vA, _ = tune_global(half); vB, _ = tune_global(~half)
    gA = eval_P(Pb, ~half, np.tile(vA, (3, 1))) - eval_P(Pb, ~half)
    gB = eval_P(Pb, half, np.tile(vB, (3, 1))) - eval_P(Pb, half)
    cross_g = 0.5 * (gA + gB)
    gmult = np.zeros(NT)
    print(f'[gmult] cross-gain {cross_g:+.4f} (A {gA:+.3f}, B {gB:+.3f})')
    if cross_g > KEEP_GAIN:
        gmult, _ = tune_global(None)
        gmult *= 0.7
        s0 = eval_P(Pb, None, np.tile(gmult, (3, 1)))
        print(f'[gmult] kept {np.round(gmult, 3)} -> {s0:.4f}')
    else:
        print('[gmult] rejected')

    # (c) anchor-only multipliers on top
    def tune_anchor(row_mask, passes=2):
        mm = np.tile(gmult, (3, 1)).copy()
        cur = eval_P(Pb, row_mask, mm)
        for _ in range(passes):
            for c in range(NT):
                b0, bb, bv = mm[2, c], mm[2, c], cur
                for g in GRID:
                    mm[2, c] = b0 + g
                    s = eval_P(Pb, row_mask, mm)
                    if s > bv:
                        bb, bv = mm[2, c], s
                mm[2, c] = bb; cur = bv
        return mm, cur

    mA, _ = tune_anchor(half); mB, _ = tune_anchor(~half)
    base_mm = np.tile(gmult, (3, 1))
    gA = eval_P(Pb, ~half, mA) - eval_P(Pb, ~half, base_mm)
    gB = eval_P(Pb, half, mB) - eval_P(Pb, half, base_mm)
    cross_a = 0.5 * (gA + gB)
    final_mm = base_mm
    print(f'[amult] cross-gain {cross_a:+.4f}')
    if cross_a > KEEP_GAIN:
        mm_full, _ = tune_anchor(None)
        final_mm = base_mm + 0.7 * (mm_full - base_mm)
        s0 = eval_P(Pb, None, final_mm)
        print(f'[amult] kept anchor row {np.round(final_mm[2], 3)} -> {s0:.4f}')
    else:
        print('[amult] rejected')

    s_final, comps = score_seqs(dec.decode(Pb, mode=mode, mults=final_mm))
    print(f'\n[FINAL OOF] {s_final:.4f}')
    print('  comps:', {k: round(v, 4) for k, v in comps.items() if not isinstance(v, dict)})
    print('  type per-class:', {k: round(v, 3) for k, v in comps['type_percls'].items()})
    print('  anchor per-class:', {k: round(v, 3) for k, v in comps['anchor_percls'].items()})

    recipe = dict(members=names, weights=[float(w) for w in ws], mode=mode,
                  anchor_mix=float(anchor_mix), beta=float(beta),
                  mults=final_mm.tolist(), oof_score=float(s_final),
                  checks=dict(cross_beta=float(cross_beta), cross_gmult=float(cross_g),
                              cross_amult=float(cross_a)))
    with open(os.path.join(base, 'foundation', 'recipe2.json'), 'w') as f:
        json.dump(recipe, f, indent=2)

    # ---- test submission ----
    Pbt = blend(ws, [np.log(P) for P in Pt_list])
    t_offs = np.concatenate([[0], np.cumsum(test_lens)])
    if anchor_mix > 0 and anchor_test is not None:
        t_anchor_idx = t_offs[1:] - 1
        la = np.log(anchor_test)
        Pbt[t_anchor_idx] = np.exp((1 - anchor_mix) * np.log(Pbt[t_anchor_idx]) + anchor_mix * la)
        Pbt /= Pbt.sum(1, keepdims=True)
    if beta > 0:
        t_roles = np.zeros(int(test_lens.sum()), int)
        for r in range(len(test_ids)):
            t_roles[t_offs[r]] = 0
            t_roles[t_offs[r] + 1: t_offs[r + 1] - 1] = 1
            t_roles[t_offs[r + 1] - 1] = 2
        Pbt = np.exp(np.log(Pbt) + beta * corr[t_roles])
        Pbt /= Pbt.sum(1, keepdims=True)
    seqs_t = dec.decode(Pbt, mults=final_mm, mode=mode, use_fold_tables=False,
                        ext_lens=test_lens, ext_folds=[0] * len(test_ids))
    sub = format_submission(test_ids, seqs_t, [test_pars[s] for s in test_ids])
    ss = pd.read_csv(os.path.join(base, 'dataset', 'public', 'sample_submission.csv'))
    sub = sub.set_index('sample_id').loc[ss['sample_id']].reset_index()[['sample_id', 'target_sequence']]
    out = os.path.join(base, 'working', 'submission.csv')
    sub.to_csv(out, index=False)
    from scorer import _parse_pred
    assert all(_parse_pred(x)[0] for x in sub['target_sequence'])
    assert list(sub.columns) == list(ss.columns) and (sub['sample_id'].values == ss['sample_id'].values).all()
    print(f'[test] wrote {out} rows={len(sub)}; format valid')


if __name__ == '__main__':
    main()
