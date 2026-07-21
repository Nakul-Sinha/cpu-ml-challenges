"""Build 3 candidate test submissions under forum-group CV:
C1 blend(0,.5,.5) viterbi (honest 54.50, no meta)
C2 meta text C4 with BAGGED stacking (per-fold base test probs -> per-fold meta -> average outputs)
C3 meta no-text C4 posterior, bagged the same way
Saves per-fold test probs for all families first."""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/solcheck'))
sys.path.insert(0, os.path.expanduser('~/discourse/foundation'))
import solution as S
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

t00 = time.time()
ROOT = os.path.expanduser('~/solcheck/dataset/public')
train = pd.read_csv(ROOT + '/train.csv')
test = pd.read_csv(ROOT + '/test.csv')
sample = pd.read_csv(ROOT + '/sample_submission.csv')
ids = train['sample_id'].tolist()
test_ids = test['sample_id'].tolist()
lens = np.array([len(r.split()) for r in train['masked_nodes']])
test_lens = np.array([len(r.split()) for r in test['masked_nodes']])
targets = {r['sample_id']: S.parse_target(r['target_sequence'])[0] for _, r in train.iterrows()}
pars_te = {r['sample_id']: S.solve_parents(r) for _, r in test.iterrows()}
fold_rows = np.load(os.path.expanduser('~/discourse/runs/group_folds.npy'))
sid_to_fold = dict(zip(ids, fold_rows))

recs_tr = S.extract_nodes(train); recs_te = S.extract_nodes(test)
n_tr, n_te = len(recs_tr), len(recs_te)
node_fold = np.array([sid_to_fold[r['sample_id']] for r in recs_tr])
y = np.array([S.TYPE_IDX[targets[r['sample_id']][r['pos']]] for r in recs_tr])
offs = np.concatenate([[0], np.cumsum(lens)])
t_offs = np.concatenate([[0], np.cumsum(test_lens)])
prev_tr = np.array([offs[i] + k - 1 if k > 0 else -1 for i in range(len(ids)) for k in range(lens[i])])
next_tr = np.array([offs[i] + k + 1 if k < lens[i] - 1 else -1 for i in range(len(ids)) for k in range(lens[i])])
prev_te = np.array([t_offs[i] + k - 1 if k > 0 else -1 for i in range(len(test_lens)) for k in range(test_lens[i])])
next_te = np.array([t_offs[i] + k + 1 if k < test_lens[i] - 1 else -1 for i in range(len(test_lens)) for k in range(test_lens[i])])

# ---------- per-fold families (train side = OOF as before; test side = PER-FOLD probs) ----------
import lightgbm as lgb
Xg_tr = S.gbm_features(recs_tr); Xg_te = S.gbm_features(recs_te)
g_oof = np.zeros((n_tr, 5)); g_test_f = np.zeros((5, n_te, 5))
for cwi, cw in enumerate(('balanced', None)):
    pass
def gbm_fold(cw, f):
    om = node_fold == f; tm = ~om
    oof_f = np.zeros((int(om.sum()), 5)); te = np.zeros((n_te, 5))
    for s in (42, 7, 123):
        clf = lgb.LGBMClassifier(objective='multiclass', num_class=5, learning_rate=0.05,
                                 num_leaves=15, min_child_samples=20, subsample=0.9,
                                 subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                 n_estimators=170, class_weight=cw, random_state=s,
                                 n_jobs=-1, verbose=-1)
        clf.fit(Xg_tr[tm], y[tm])
        oof_f += clf.predict_proba(Xg_tr[om]) / 3
        te += clf.predict_proba(Xg_te) / 3
    return om, oof_f, te

g_oof_b = np.zeros((n_tr, 5)); g_te_b = np.zeros((5, n_te, 5))
g_oof_n = np.zeros((n_tr, 5)); g_te_n = np.zeros((5, n_te, 5))
for f in range(5):
    om, o, te = gbm_fold('balanced', f); g_oof_b[om] = o; g_te_b[f] = te
    om, o, te = gbm_fold(None, f); g_oof_n[om] = o; g_te_n[f] = te
def geo(a, b, w):
    m = np.exp(w * np.log(np.clip(a, 1e-9, 1)) + (1 - w) * np.log(np.clip(b, 1e-9, 1)))
    return m / m.sum(-1, keepdims=True)
g_oof = geo(g_oof_b, g_oof_n, 0.7)
g_test_f = geo(g_te_b, g_te_n, 0.7)          # (5, n_te, 5) per-fold
print(f'gbm per-fold done {time.time()-t00:.0f}s', flush=True)

Dl_tr = S.textlin_dense(recs_tr); Dl_te = S.textlin_dense(recs_te)
tit_tr = [str(r['title']) for r in recs_tr]; for_tr = [str(r['forum']) for r in recs_tr]
tit_te = [str(r['title']) for r in recs_te]; for_te = [str(r['forum']) for r in recs_te]
Ttr = S.hash_text(tit_tr, for_tr); Tte = S.hash_text(tit_te, for_te)
t_oof = np.zeros((n_tr, 5)); t_test_f = np.zeros((5, n_te, 5))
for f in range(5):
    tm = node_fold != f; vm = ~tm
    scl = StandardScaler().fit(Dl_tr[tm])
    Xtr = sparse.hstack([Ttr[tm], sparse.csr_matrix(scl.transform(Dl_tr[tm]))]).tocsr()
    Xva = sparse.hstack([Ttr[vm], sparse.csr_matrix(scl.transform(Dl_tr[vm]))]).tocsr()
    Xte = sparse.hstack([Tte, sparse.csr_matrix(scl.transform(Dl_te))]).tocsr()
    m = LogisticRegression(C=0.5, class_weight='balanced', solver='lbfgs', max_iter=500, tol=1e-3)
    m.fit(Xtr, y[tm])
    t_oof[vm] = S.predict_full(m, Xva)
    t_test_f[f] = S.predict_full(m, Xte)
print(f'textlin per-fold done {time.time()-t00:.0f}s', flush=True)

# nnseq: reuse the *aggregate* per-fold structure by rerunning (per-fold test probs needed)
import torch
import collections, hashlib
n_oof = np.zeros((n_tr, 5)); n_test_f = np.zeros((5, n_te, 5))
# reuse S.run_nnseq internals is heavy; simpler: call run_nnseq per fold is not supported.
# Instead: quick per-fold torch training mirroring S.run_nnseq but capturing per-fold test preds.
def run_nn_perfold(seeds_per_arch=3):
    # replicate S.run_nnseq but return per-fold test
    import types
    src = S.run_nnseq.__code__
    # simplest: inline copy of the loop calling internal builders via S.run_nnseq is not possible;
    # so re-run S.run_nnseq 5 times is wasteful. We instead accept fold-mean nnseq for C1 and use
    # gbm+textlin per-fold + nnseq FOLD-MEAN for the bagged meta (nn shift is smallest: 30-model avg).
    return None
n_oof_saved = np.load(os.path.expanduser('~/discourse/runs/group_n_oof.npy'))
n_test_mean = np.load(os.path.expanduser('~/discourse/runs/group_n_test.npy'))
n_oof = n_oof_saved
print(f'nnseq reuse (fold-mean test) {time.time()-t00:.0f}s', flush=True)

W = (0.0, 0.5, 0.5)
def blend3(g, t, n):
    L = W[0] * np.log(np.clip(g, 1e-9, 1)) + W[1] * np.log(np.clip(t, 1e-9, 1)) + W[2] * np.log(np.clip(n, 1e-9, 1))
    P = np.exp(L); return P / P.sum(-1, keepdims=True)

# ---------- C1: honest blend viterbi ----------
full_tables = S.fit_transitions(targets, ids)
t_test_mean = t_test_f.mean(0)
g_test_mean = g_test_f.mean(0)
Bt_mean = blend3(g_test_mean, t_test_mean, n_test_mean)
seqs = S.decode_matrix(Bt_mean, test_lens, full_tables, 'viterbi')
def write_sub(seqs_idx, path):
    rows = {}
    for r, sid in enumerate(test_ids):
        toks = []
        for k, ci in enumerate(seqs_idx[r]):
            toks.append('type_' + S.TYPES[int(ci)])
            toks.append('parent_' + pars_te[sid][k])
        rows[sid] = ' '.join(toks)
    sub = pd.DataFrame({'sample_id': sample['sample_id'],
                        'target_sequence': [rows[s] for s in sample['sample_id']]})
    sub.to_csv(path, index=False)
    return sub
write_sub(seqs, os.path.expanduser('~/discourse/runs/C1_blend.csv'))
print('C1 written', flush=True)

# ---------- bagged meta (C2 text / C3 no-text) ----------
Bo = blend3(g_oof, t_oof, n_oof)
def neigh(B, pv, nx):
    n = B.shape[0]; p = np.zeros((n, 5)); q = np.zeros((n, 5))
    m = pv >= 0; p[m] = B[pv[m]]; m2 = nx >= 0; q[m2] = B[nx[m2]]
    return p, q
pvo, nxo = neigh(Bo, prev_tr, next_tr)
Mtr = np.concatenate([np.log(np.clip(np.concatenate([g_oof, t_oof, n_oof], 1), 1e-9, 1)),
                      np.log(Bo), np.log(np.clip(pvo, 1e-9, 1)), np.log(np.clip(nxo, 1e-9, 1)), Dl_tr], 1)

def bagged_meta(C, use_text):
    te_acc = np.zeros((n_te, 5))
    for f in range(5):
        tm = node_fold != f
        scl = StandardScaler().fit(Mtr[tm])
        def bx_tr(mask):
            b = sparse.csr_matrix(scl.transform(Mtr[mask]))
            return sparse.hstack([b, Ttr[mask]]).tocsr() if use_text else b
        m = LogisticRegression(C=C, class_weight='balanced', max_iter=4000, tol=1e-3)
        m.fit(bx_tr(tm), y[tm])
        # test inputs from THIS fold's base models (single-model distribution)
        g_te = g_test_f[f]; t_te = t_test_f[f]; n_te_p = n_test_mean
        Bt = blend3(g_te, t_te, n_te_p)
        pvt, nxt_ = neigh(Bt, prev_te, next_te)
        Mte = np.concatenate([np.log(np.clip(np.concatenate([g_te, t_te, n_te_p], 1), 1e-9, 1)),
                              np.log(Bt), np.log(np.clip(pvt, 1e-9, 1)),
                              np.log(np.clip(nxt_, 1e-9, 1)), Dl_te], 1)
        Xte = sparse.csr_matrix(scl.transform(Mte))
        if use_text:
            Xte = sparse.hstack([Xte, Tte]).tocsr()
        te_acc += S.predict_full(m, Xte) / 5
    return te_acc

for tag, C, ut, mode in (('C2_metatext_bag', 4.0, True, 'posterior'),
                         ('C3_metanotext_bag', 4.0, False, 'posterior')):
    te = bagged_meta(C, ut)
    seqs = S.decode_matrix(te, test_lens, full_tables, mode)
    write_sub(seqs, os.path.expanduser(f'~/discourse/runs/{tag}.csv'))
    print(f'{tag} written ({time.time()-t00:.0f}s)', flush=True)
print('DONE', flush=True)
