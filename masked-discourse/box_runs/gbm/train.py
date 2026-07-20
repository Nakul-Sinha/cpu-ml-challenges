"""GBM family trainer: LightGBM multiclass on dense features, honest 5-fold OOF.
Row-level inner-validation for early stopping (OOF fold never used to pick iters).
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FND = os.path.join(HERE, '..', '..', 'foundation')
sys.path.insert(0, FND); sys.path.insert(0, HERE)
import common
from common import TYPES, TYPE_IDX
import features as F

os.environ.setdefault('OMP_NUM_THREADS', '5')
DATA = os.path.join(HERE, '..', '..', 'dataset', 'public')


def load_all(title_buck=48, forum_buck=32, ngram=3, hashing=True):
    tr = pd.read_csv(os.path.join(DATA, 'train.csv'))
    te = pd.read_csv(os.path.join(DATA, 'test.csv'))
    rtr = common.extract_nodes(tr); rte = common.extract_nodes(te)
    Xtr, names = F.build_features(rtr, title_buck, forum_buck, ngram, hashing)
    Xte, _ = F.build_features(rte, title_buck, forum_buck, ngram, hashing)
    ytr = []
    for _, r in tr.iterrows():
        t, _ = common.parse_target(r['target_sequence']); ytr += t
    ytr = np.array([TYPE_IDX[t] for t in ytr], dtype=int)
    folds = common.make_folds(tr)
    node_row, node_fold, node_pos, node_len = [], [], [], []
    for ri, (_, r) in enumerate(tr.iterrows()):
        L = len(r['masked_nodes'].split())
        for p in range(L):
            node_row.append(ri); node_fold.append(folds[ri])
            node_pos.append(p); node_len.append(L)
    node_row = np.array(node_row); node_fold = np.array(node_fold)
    node_pos = np.array(node_pos); node_len = np.array(node_len)
    assert len(ytr) == Xtr.shape[0] == len(node_fold)
    return dict(Xtr=Xtr, Xte=Xte, ytr=ytr, names=names, folds=folds,
                node_row=node_row, node_fold=node_fold, node_pos=node_pos,
                node_len=node_len, tr=tr, te=te)


def run_cv(D, params, class_weight=None, anchor_w=1.0, inner_frac=0.15,
           es_rounds=60, seed=42, verbose=False):
    import lightgbm as lgb
    Xtr, Xte, y = D['Xtr'], D['Xte'], D['ytr']
    node_fold, node_row, node_pos, node_len = (
        D['node_fold'], D['node_row'], D['node_pos'], D['node_len'])
    NC = 5
    oof = np.zeros((Xtr.shape[0], NC), dtype=np.float64)
    test = np.zeros((Xte.shape[0], NC), dtype=np.float64)
    best_iters = []
    rng = np.random.RandomState(seed)
    sw_all = np.ones(len(y))
    sw_all[node_pos == node_len - 1] *= anchor_w
    for f in range(5):
        oof_mask = node_fold == f
        tr_mask = ~oof_mask
        tr_rows = np.unique(node_row[tr_mask])
        rng.shuffle(tr_rows)
        n_val = max(1, int(len(tr_rows) * inner_frac))
        val_rows = set(tr_rows[:n_val].tolist())
        val_mask = np.array([tr_mask[i] and (node_row[i] in val_rows)
                             for i in range(len(y))])
        fit_mask = tr_mask & ~val_mask
        clf = lgb.LGBMClassifier(
            objective='multiclass', num_class=NC, n_estimators=3000,
            class_weight=class_weight, n_jobs=int(os.environ.get('OMP_NUM_THREADS', 5)),
            verbose=-1, **params)
        clf.fit(Xtr[fit_mask], y[fit_mask], sample_weight=sw_all[fit_mask],
                eval_set=[(Xtr[val_mask], y[val_mask])], eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                           lgb.log_evaluation(0)])
        bi = clf.best_iteration_ or params.get('n_estimators', 300)
        best_iters.append(bi)
        oof[oof_mask] = clf.predict_proba(Xtr[oof_mask])
        test += clf.predict_proba(Xte) / 5.0
        if verbose:
            print('  fold %d: best_iter=%s fit=%d val=%d oof=%d' % (
                f, bi, fit_mask.sum(), val_mask.sum(), oof_mask.sum()))
    return oof, test, best_iters


def quick_score(oof, D, mode='viterbi'):
    sys.path.insert(0, FND)
    from decode import fit_transitions, decode_row
    from scorer import score_submission
    tr = D['tr']
    ids = tr['sample_id'].tolist()
    lens = {r['sample_id']: len(r['masked_nodes'].split()) for _, r in tr.iterrows()}
    targets = {r['sample_id']: common.parse_target(r['target_sequence'])[0] for _, r in tr.iterrows()}
    fold_of = dict(zip(ids, D['folds']))
    pars = {r['sample_id']: common.solve_parents(r) for _, r in tr.iterrows()}
    truth = tr[['sample_id', 'target_sequence']]
    pbr = {}; o = 0
    for sid in ids:
        L = lens[sid]; pbr[sid] = oof[o:o + L]; o += L
    tables = {f: fit_transitions(targets, [s for s in ids if fold_of[s] != f]) for f in range(5)}
    seqs = {sid: decode_row(pbr[sid], tables[fold_of[sid]], mode=mode) for sid in ids}
    sub = common.format_submission(ids, [seqs[s] for s in ids], [pars[s] for s in ids])
    s, comps = score_submission(sub, truth, verbose=False)
    return s, comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--leaves', type=int, default=31)
    ap.add_argument('--minleaf', type=int, default=20)
    ap.add_argument('--ff', type=float, default=0.8)
    ap.add_argument('--bf', type=float, default=0.9)
    ap.add_argument('--l2', type=float, default=1.0)
    ap.add_argument('--cw', default='none')
    ap.add_argument('--anchor_w', type=float, default=1.0)
    ap.add_argument('--no_hash', action='store_true')
    ap.add_argument('--title_buck', type=int, default=48)
    ap.add_argument('--forum_buck', type=int, default=32)
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    t0 = time.time()
    D = load_all(a.title_buck, a.forum_buck, hashing=not a.no_hash)
    params = dict(learning_rate=a.lr, num_leaves=a.leaves, min_child_samples=a.minleaf,
                  subsample=a.bf, subsample_freq=1, colsample_bytree=a.ff,
                  reg_lambda=a.l2, reg_alpha=0.0)
    cw = None if a.cw == 'none' else a.cw
    oof, test, bi = run_cv(D, params, class_weight=cw, anchor_w=a.anchor_w, verbose=True)
    sv, cv = quick_score(oof, D, 'viterbi')
    sp, cp = quick_score(oof, D, 'posterior')
    print('[%s] feats=%d best_iters=%s' % (a.tag, D['Xtr'].shape[1], bi))
    print('  VITERBI   score=%.4f  TypeMF1=%.4f AnchMF1=%.4f' % (sv, cv['TypeMacroF1'], cv['AnchorMacroF1']))
    print('  POSTERIOR score=%.4f  TypeMF1=%.4f AnchMF1=%.4f' % (sp, cp['TypeMacroF1'], cp['AnchorMacroF1']))
    print('  elapsed %.1fs' % (time.time() - t0))
    if a.save:
        np.save(os.path.join(HERE, 'oof_probs.npy'), oof.astype(np.float64))
        np.save(os.path.join(HERE, 'test_probs.npy'), test.astype(np.float64))
        best = ('viterbi', sv, cv) if sv >= sp else ('posterior', sp, cp)
        out = {k: float(v) for k, v in best[2].items() if not isinstance(v, dict)}
        out['score'] = float(best[1]); out['mode'] = best[0]
        out['viterbi_score'] = float(sv); out['posterior_score'] = float(sp)
        out['params'] = params; out['class_weight'] = a.cw; out['anchor_w'] = a.anchor_w
        out['best_iters'] = [int(x) for x in bi]; out['n_features'] = int(D['Xtr'].shape[1])
        with open(os.path.join(HERE, 'score.json'), 'w') as fh:
            json.dump(out, fh, indent=2)
        print('  SAVED oof/test/score.json best=', best[0], best[1])


if __name__ == '__main__':
    main()
