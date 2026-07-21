"""Forum-group CV: the honest protocol. Rebuild families + blend + meta under
group folds (forum never straddles folds); quantify the text/forum OOF illusion;
tune multipliers honestly. Reuses the parameterized functions in solution.py."""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/solcheck'))
sys.path.insert(0, os.path.expanduser('~/discourse/foundation'))
import solution as S
from fast_score import FastScorer
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

t00 = time.time()
ROOT = os.path.expanduser('~/solcheck/dataset/public')
train = pd.read_csv(ROOT + '/train.csv')
test = pd.read_csv(ROOT + '/test.csv')
ids = train['sample_id'].tolist()
lens = np.array([len(r.split()) for r in train['masked_nodes']])
test_lens = np.array([len(r.split()) for r in test['masked_nodes']])
targets = {r['sample_id']: S.parse_target(r['target_sequence'])[0] for _, r in train.iterrows()}
pars_tr = {r['sample_id']: S.solve_parents(r) for _, r in train.iterrows()}

# ---- forum-group folds: greedy balance by group size, then anchor check ----
rng = np.random.RandomState(42)
groups = {}
for i, f in enumerate(train['forum']):
    groups.setdefault(f, []).append(i)
order = sorted(groups.items(), key=lambda kv: -len(kv[1]))
fold_rows = np.zeros(len(train), int)
loads = [0] * 5
for f, rows in order:
    k = int(np.argmin(loads))
    for r in rows:
        fold_rows[r] = k
    loads[k] += len(rows)
print('group-fold row counts:', np.bincount(fold_rows).tolist(), flush=True)
anch = np.array([targets[s][-1] for s in ids])
for k in range(5):
    vals, cnts = np.unique(anch[fold_rows == k], return_counts=True)
    print(f'  fold {k} anchor dist: {dict(zip(vals.tolist(), cnts.tolist()))}', flush=True)

recs_tr = S.extract_nodes(train)
recs_te = S.extract_nodes(test)
n_tr, n_te = len(recs_tr), len(recs_te)
sid_to_fold = dict(zip(ids, fold_rows))
node_fold = np.array([sid_to_fold[r['sample_id']] for r in recs_tr])
y = np.array([S.TYPE_IDX[targets[r['sample_id']][r['pos']]] for r in recs_tr])
offs = np.concatenate([[0], np.cumsum(lens)])
t_offs = np.concatenate([[0], np.cumsum(test_lens)])
prev_tr = np.array([offs[i] + k - 1 if k > 0 else -1 for i in range(len(ids)) for k in range(lens[i])])
next_tr = np.array([offs[i] + k + 1 if k < lens[i] - 1 else -1 for i in range(len(ids)) for k in range(lens[i])])
prev_te = np.array([t_offs[i] + k - 1 if k > 0 else -1 for i in range(len(test_lens)) for k in range(test_lens[i])])
next_te = np.array([t_offs[i] + k + 1 if k < test_lens[i] - 1 else -1 for i in range(len(test_lens)) for k in range(test_lens[i])])

tables_by_fold = {f: S.fit_transitions(targets, [s for s in ids if sid_to_fold[s] != f]) for f in range(5)}
true_types = [[S.TYPE_IDX[t] for t in targets[s]] for s in ids]
true_pars = [S.parse_target(train.iloc[i]['target_sequence'])[1] for i in range(len(ids))]
fs_full = FastScorer(true_types, true_pars, [pars_tr[s] for s in ids])


def decode_oof(P, mode):
    out = [None] * len(ids)
    E_all = np.log(np.clip(P, 1e-9, 1.0))
    for f in range(5):
        T, I = tables_by_fold[f]
        for L in (3, 4):
            idx = [r for r in range(len(ids)) if lens[r] == L and sid_to_fold[ids[r]] == f]
            if not idx:
                continue
            E = np.stack([E_all[offs[r]:offs[r] + L] for r in idx])
            dec = S.batch_viterbi(E, T[L], I[L]) if mode == 'viterbi' else S.batch_posterior(E, T[L], I[L])
            for k, r in enumerate(idx):
                out[r] = dec[k]
    return out


def sc(P, mode, mults=None):
    out = decode_oof_m(P, mode, mults) if mults is not None else decode_oof(P, mode)
    return fs_full.score(out)[0]


def decode_oof_m(P, mode, mults):
    out = [None] * len(ids)
    E_all = np.log(np.clip(P, 1e-9, 1.0))
    for f in range(5):
        T, I = tables_by_fold[f]
        for L in (3, 4):
            idx = [r for r in range(len(ids)) if lens[r] == L and sid_to_fold[ids[r]] == f]
            if not idx:
                continue
            E = np.stack([E_all[offs[r]:offs[r] + L] for r in idx])
            roles = [0] + [1] * (L - 2) + [2]
            E = E + mults[roles][None, :, :]
            dec = S.batch_viterbi(E, T[L], I[L]) if mode == 'viterbi' else S.batch_posterior(E, T[L], I[L])
            for k, r in enumerate(idx):
                out[r] = dec[k]
    return out


# ---- families under group folds ----
Xg_tr = S.gbm_features(recs_tr); Xg_te = S.gbm_features(recs_te)
t0 = time.time()
gbm_out = S.run_gbm(Xg_tr, Xg_te, y, node_fold)
g_best = None
for tag, (o, t) in gbm_out.items():
    s = max(sc(o, 'viterbi'), sc(o, 'posterior'))
    print(f'gbm[{tag}] group-OOF {s:.4f}', flush=True)
    if g_best is None or s > g_best[0]:
        g_best = (s, tag)
g_oof, g_test = gbm_out[g_best[1]]
print(f'gbm winner {g_best[1]} {g_best[0]:.4f} ({time.time()-t0:.0f}s)', flush=True)

Dl_tr = S.textlin_dense(recs_tr); Dl_te = S.textlin_dense(recs_te)
tit_tr = [str(r['title']) for r in recs_tr]; for_tr = [str(r['forum']) for r in recs_tr]
tit_te = [str(r['title']) for r in recs_te]; for_te = [str(r['forum']) for r in recs_te]
t0 = time.time()
t_oof, t_test = S.run_textlin(Dl_tr, Dl_te, tit_tr, for_tr, tit_te, for_te, y, node_fold)
print(f'textlin group-OOF {max(sc(t_oof, "viterbi"), sc(t_oof, "posterior")):.4f} ({time.time()-t0:.0f}s)', flush=True)

t0 = time.time()
n_oof, n_test, nm = S.run_nnseq(train, test, fold_rows, 3)
print(f'nnseq ({nm} models) group-OOF {max(sc(n_oof, "viterbi"), sc(n_oof, "posterior")):.4f} ({time.time()-t0:.0f}s)', flush=True)

np.save(os.path.expanduser('~/discourse/runs/group_g_oof.npy'), g_oof)
np.save(os.path.expanduser('~/discourse/runs/group_g_test.npy'), g_test)
np.save(os.path.expanduser('~/discourse/runs/group_t_oof.npy'), t_oof)
np.save(os.path.expanduser('~/discourse/runs/group_t_test.npy'), t_test)
np.save(os.path.expanduser('~/discourse/runs/group_n_oof.npy'), n_oof)
np.save(os.path.expanduser('~/discourse/runs/group_n_test.npy'), n_test)
np.save(os.path.expanduser('~/discourse/runs/group_folds.npy'), fold_rows)

# ---- blend ----
logs = [np.log(np.clip(P, 1e-9, 1)) for P in (g_oof, t_oof, n_oof)]
best = None
for wg in range(9):
    for wt in range(9 - wg):
        wn = 8 - wg - wt
        ws = (wg / 8, wt / 8, wn / 8)
        Pb = np.exp(sum(w * l for w, l in zip(ws, logs))); Pb /= Pb.sum(1, keepdims=True)
        for mode in ('viterbi', 'posterior'):
            s = sc(Pb, mode)
            if best is None or s > best[0]:
                best = (s, ws, mode)
sB, W, mdB = best
print(f'blend {W} {mdB} group-OOF {sB:.4f}', flush=True)
Bo = np.exp(sum(w * l for w, l in zip(W, logs))); Bo /= Bo.sum(1, keepdims=True)

# ---- title-only probe (both to document illusion) ----
Ttr = S.hash_text(tit_tr, for_tr); Tte = S.hash_text(tit_te, for_te)
po = np.zeros((n_tr, 5))
for f in range(5):
    tm = node_fold != f
    m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, tol=1e-3)
    m.fit(Ttr[tm], y[tm])
    po[~tm] = S.predict_full(m, Ttr[~tm])
print(f'title+forum-ONLY LR group-OOF: {max(sc(po, "viterbi"), sc(po, "posterior")):.4f}', flush=True)

# ---- meta under group folds: text vs no-text ----
def run_meta_g(C, use_text):
    def neigh(B, pv, nx):
        n = B.shape[0]; p = np.zeros((n, 5)); q = np.zeros((n, 5))
        m = pv >= 0; p[m] = B[pv[m]]; m2 = nx >= 0; q[m2] = B[nx[m2]]
        return p, q
    pvo, nxo = neigh(Bo, prev_tr, next_tr)
    Mtr = np.concatenate([np.log(np.clip(np.concatenate([g_oof, t_oof, n_oof], 1), 1e-9, 1)),
                          np.log(Bo), np.log(np.clip(pvo, 1e-9, 1)), np.log(np.clip(nxo, 1e-9, 1)), Dl_tr], 1)
    oof = np.zeros((n_tr, 5))
    for f in range(5):
        tm = node_fold != f; vm = ~tm
        scl = StandardScaler().fit(Mtr[tm])
        def bx(idx_mask):
            b = sparse.csr_matrix(scl.transform(Mtr[idx_mask]))
            return sparse.hstack([b, Ttr[idx_mask]]).tocsr() if use_text else b
        m = LogisticRegression(C=C, class_weight='balanced', max_iter=4000, tol=1e-3)
        m.fit(bx(tm), y[tm])
        oof[vm] = S.predict_full(m, bx(vm))
    return oof

for use_text in (False, True):
    for C in (1.5, 4.0):
        t0 = time.time()
        mo = run_meta_g(C, use_text)
        sv, sp = sc(mo, 'viterbi'), sc(mo, 'posterior')
        print(f'meta group text={use_text} C={C}: vit {sv:.4f} post {sp:.4f} ({time.time()-t0:.0f}s)', flush=True)
        np.save(os.path.expanduser(f'~/discourse/runs/group_meta_t{int(use_text)}_C{C}.npy'), mo)

# ---- honest multiplier ascent on the best simple emission (blend) ----
GRID = (-0.35, -0.2, -0.1, 0.1, 0.2, 0.35)
gmask = np.array([hash(f) % 2 for f in train['forum']])  # group-respecting halves

def tune(P, mode, mask_val=None, passes=2):
    mults = np.zeros((3, 5))
    def s_of(mm):
        out = decode_oof_m(P, mode, mm)
        if mask_val is None:
            return fs_full.score(out)[0]
        idx = np.where(gmask == mask_val)[0]
        fsx = FastScorer([true_types[i] for i in idx], [true_pars[i] for i in idx],
                         [[pars_tr[ids[i]]][0] for i in idx])
        return fsx.score([out[i] for i in idx])[0]
    cur = s_of(mults)
    for _ in range(passes):
        for r in range(3):
            for c in range(5):
                b0, bb, bv = mults[r, c], mults[r, c], cur
                for g in GRID:
                    mults[r, c] = b0 + g
                    v = s_of(mults)
                    if v > bv:
                        bb, bv = mults[r, c], v
                mults[r, c] = bb; cur = bv
    return mults, cur

mA, _ = tune(Bo, mdB, 0)
mB, _ = tune(Bo, mdB, 1)
def s_half(P, mode, mm, val):
    out = decode_oof_m(P, mode, mm)
    idx = np.where(gmask == val)[0]
    fsx = FastScorer([true_types[i] for i in idx], [true_pars[i] for i in idx],
                     [pars_tr[ids[i]] for i in idx])
    return fsx.score([out[i] for i in idx])[0]
base_1 = s_half(Bo, mdB, np.zeros((3, 5)), 1); base_0 = s_half(Bo, mdB, np.zeros((3, 5)), 0)
gA = s_half(Bo, mdB, mA, 1) - base_1
gB = s_half(Bo, mdB, mB, 0) - base_0
print(f'multiplier nested (group halves): A->B {gA:+.3f}  B->A {gB:+.3f}', flush=True)
m_full, s_tuned = tune(Bo, mdB, None)
print(f'multiplier in-sample tuned blend: {s_tuned:.4f} (base {sB:.4f})', flush=True)
print(f'TOTAL {time.time()-t00:.0f}s DONE', flush=True)
