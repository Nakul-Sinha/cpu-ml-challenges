"""Honest sweep under forum-group CV: gbm variant, re-blend (incl nn5 when
ready), transition alpha grid, anchor/global multipliers with nested checks."""
import sys, os, time, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/solcheck'))
sys.path.insert(0, os.path.expanduser('~/discourse/foundation'))
import solution as S
from fast_score import FastScorer
import lightgbm as lgb

t00 = time.time()
ROOT = os.path.expanduser('~/solcheck/dataset/public')
train = pd.read_csv(ROOT + '/train.csv')
ids = train['sample_id'].tolist()
lens = np.array([len(r.split()) for r in train['masked_nodes']])
targets = {r['sample_id']: S.parse_target(r['target_sequence'])[0] for _, r in train.iterrows()}
pars_tr = {r['sample_id']: S.solve_parents(r) for _, r in train.iterrows()}
fold_rows = np.load(os.path.expanduser('~/discourse/runs/group_folds.npy'))
sid_to_fold = dict(zip(ids, fold_rows))
recs_tr = S.extract_nodes(train)
node_fold = np.array([sid_to_fold[r['sample_id']] for r in recs_tr])
y = np.array([S.TYPE_IDX[targets[r['sample_id']][r['pos']]] for r in recs_tr])
offs = np.concatenate([[0], np.cumsum(lens)])
true_types = [[S.TYPE_IDX[t] for t in targets[s]] for s in ids]
true_pars = [S.parse_target(train.iloc[i]['target_sequence'])[1] for i in range(len(ids))]
fs_full = FastScorer(true_types, true_pars, [pars_tr[s] for s in ids])
R = os.path.expanduser('~/discourse/runs')
g_oof = np.load(R + '/group_g_oof.npy'); t_oof = np.load(R + '/group_t_oof.npy')
n_oof = np.load(R + '/group_n_oof.npy')


def make_tables(alpha):
    return {f: S.fit_transitions(targets, [s for s in ids if sid_to_fold[s] != f], alpha=alpha)
            for f in range(5)}


TB = {1.0: make_tables(1.0)}


def decode_all(P, mode, mults=None, alpha=1.0):
    tables = TB.setdefault(alpha, make_tables(alpha))
    out = [None] * len(ids)
    E_all = np.log(np.clip(P, 1e-9, 1.0))
    for f in range(5):
        T, I = tables[f]
        for L in (3, 4):
            idx = [r for r in range(len(ids)) if lens[r] == L and sid_to_fold[ids[r]] == f]
            if not idx:
                continue
            E = np.stack([E_all[offs[r]:offs[r] + L] for r in idx])
            if mults is not None:
                roles = [0] + [1] * (L - 2) + [2]
                E = E + mults[roles][None, :, :]
            dec = S.batch_viterbi(E, T[L], I[L]) if mode == 'viterbi' else S.batch_posterior(E, T[L], I[L])
            for k, r in enumerate(idx):
                out[r] = dec[k]
    return out


def sc(P, mode, mults=None, alpha=1.0, rows=None):
    out = decode_all(P, mode, mults, alpha)
    if rows is None:
        return fs_full.score(out)[0]
    fsx = FastScorer([true_types[i] for i in rows], [true_pars[i] for i in rows],
                     [pars_tr[ids[i]] for i in rows])
    return fsx.score([out[i] for i in rows])[0]


# ---- S3: gbm variant lv31 n300 lr0.03 ----
Xg = S.gbm_features(recs_tr)
Xg_te = None  # oof-only for selection; test refit later in solution
g2_oof = np.zeros((len(recs_tr), 5))
for cw in ('balanced',):
    for f in range(5):
        om = node_fold == f; tm = ~om
        acc = np.zeros((int(om.sum()), 5))
        for s in (42, 7, 123):
            clf = lgb.LGBMClassifier(objective='multiclass', num_class=5, learning_rate=0.03,
                                     num_leaves=31, min_child_samples=20, subsample=0.9,
                                     subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                     n_estimators=300, class_weight=cw, random_state=s,
                                     n_jobs=-1, verbose=-1)
            clf.fit(Xg[tm], y[tm])
            acc += clf.predict_proba(Xg[om]) / 3
        g2_oof[om] = acc
print(f'gbm31 OOF v={sc(g2_oof, "viterbi"):.4f} p={sc(g2_oof, "posterior"):.4f} ({time.time()-t00:.0f}s)', flush=True)
np.save(R + '/group_g2_oof.npy', g2_oof)

# ---- re-blend over members (add nn5 if present) ----
members = {'g': g_oof, 'g2': g2_oof, 't': t_oof, 'n': n_oof}
if os.path.exists(R + '/group_n5_oof.npy'):
    members['n5'] = np.load(R + '/group_n5_oof.npy')
    print('nn5 available', flush=True)
names = list(members)
logs = [np.log(np.clip(members[k], 1e-9, 1)) for k in names]
best = None
steps = [w / 8 for w in range(9)]
import itertools
for ws in itertools.product(steps, repeat=len(names)):
    if abs(sum(ws) - 1) > 1e-9:
        continue
    Pb = np.exp(sum(w * l for w, l in zip(ws, logs))); Pb /= Pb.sum(1, keepdims=True)
    for md in ('viterbi', 'posterior'):
        s = sc(Pb, md)
        if best is None or s > best[0]:
            best = (s, ws, md)
sB, W, mdB = best
print(f'reblend {dict(zip(names, W))} {mdB} -> {sB:.4f} ({time.time()-t00:.0f}s)', flush=True)
Bo = np.exp(sum(w * l for w, l in zip(W, logs))); Bo /= Bo.sum(1, keepdims=True)
np.save(R + '/group_bestblend_oof.npy', Bo)
with open(R + '/group_bestblend.txt', 'w') as fh:
    fh.write(repr((names, W, mdB, sB)))

# ---- S1: transition alpha grid ----
for alpha in (0.5, 2.0, 4.0):
    print(f'alpha={alpha}: v={sc(Bo, "viterbi", alpha=alpha):.4f} p={sc(Bo, "posterior", alpha=alpha):.4f}', flush=True)

# ---- S2: multipliers (global-5 and anchor-5), nested by forum-hash halves ----
gmask = np.array([hash(f) % 2 for f in train['forum']])
GRID = (-0.3, -0.15, 0.15, 0.3)


def tune_vec(kind, rows_mask_val=None, passes=2):
    mults = np.zeros((3, 5))
    rows = None if rows_mask_val is None else list(np.where(gmask == rows_mask_val)[0])
    cur = sc(Bo, mdB, mults, 1.0, rows)
    for _ in range(passes):
        for c in range(5):
            b0 = mults[2, c] if kind == 'anchor' else mults[0, c]
            bb, bv = b0, cur
            for g in GRID:
                if kind == 'anchor':
                    mults[2, c] = b0 + g
                else:
                    mults[:, c] = b0 + g
                v = sc(Bo, mdB, mults, 1.0, rows)
                if v > bv:
                    bb, bv = b0 + g, v
            if kind == 'anchor':
                mults[2, c] = bb
            else:
                mults[:, c] = bb
            cur = bv
    return mults, cur


for kind in ('global', 'anchor'):
    mA, _ = tune_vec(kind, 0)
    mB, _ = tune_vec(kind, 1)
    rows0 = list(np.where(gmask == 0)[0]); rows1 = list(np.where(gmask == 1)[0])
    z = np.zeros((3, 5))
    gA = sc(Bo, mdB, mA, 1.0, rows1) - sc(Bo, mdB, z, 1.0, rows1)
    gB = sc(Bo, mdB, mB, 1.0, rows0) - sc(Bo, mdB, z, 1.0, rows0)
    m_full, s_in = tune_vec(kind, None)
    print(f'mult[{kind}]: nested A->B {gA:+.3f} B->A {gB:+.3f} | in-sample {s_in:.4f} (base {sB:.4f})', flush=True)
    np.save(R + f'/group_mult_{kind}.npy', m_full)
print(f'SWEEP DONE {time.time()-t00:.0f}s', flush=True)
