import sys, os, collections, json, time, argparse
sys.path.insert(0, os.path.expanduser('~/discourse/foundation'))
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from common import (TYPES, TYPE_IDX, parse_target, extract_nodes, make_folds,
                    format_submission, solve_parents)
from decode import fit_transitions, decode_row
from scorer import score_submission

ROOT = os.path.expanduser('~/discourse/dataset/public')
VIS6 = ['answer', 'elaboration', 'question', 'appreciation', 'agreement', 'other']
PARTYPES = ['answer', 'elaboration', 'question', 'appreciation', 'agreement', 'other', 'ROOT', 'MASK']


def dense_row(r):
    L = r['route_len']; pos = r['pos']
    is0 = pos == 0; isanc = pos == L - 1; isint = (not is0) and (not isanc)
    f = [float(is0), float(isint), float(isanc),
         float(pos), float(L == 4), float(r['depth']), float(r['par_depth']),
         float(r['gap_prev']), float(r['gap_next']),
         float(r['view_idx']), float(r['view_frac']), float(r['n_nodes']),
         float(r['max_depth']), float(r['n_out']), float(r['has_post'])]
    for k in ['root', 'masked', 'post', 'visible']:
        f.append(float(r['par_kind'] == k))
    for t in PARTYPES:
        f.append(float(r['par_type'] == t))
    kt = r['kid_types']; nk = r['n_kids_vis']
    for t in VIS6:
        f.append(float(kt.get(t, 0)))
    for t in VIS6:
        f.append(kt.get(t, 0) / nk if nk > 0 else 0.0)
    f += [float(nk), float(r['n_masked_kids']), float(nk > 0)]
    dt = r['desc_types']; nd = r['n_desc_vis']
    for t in VIS6:
        f.append(float(dt.get(t, 0)))
    for t in VIS6:
        f.append(dt.get(t, 0) / nd if nd > 0 else 0.0)
    f += [float(nd), float(nd > 0)]
    st = r['sib_types']; ns = r['n_sibs']
    for t in VIS6:
        f.append(float(st.get(t, 0)))
    for t in VIS6:
        f.append(st.get(t, 0) / ns if ns > 0 else 0.0)
    f += [float(ns)]
    bt = collections.Counter(r['between'])
    for t in VIS6:
        f.append(float(bt.get(t, 0)))
    vc = r['vis_counts']; nn = r['n_nodes']
    for t in VIS6:
        f.append(vc.get(t, 0) / nn if nn > 0 else 0.0)
    f.append(float(nd == 0 and r['n_masked_kids'] == 0))  # terminal
    f += [float(r['title_len']), float(r['title_words']), float(r['title_q']),
          float(r['title_excl']), float(r['title_wh'])]
    f += [float(r['prof_wide']), float(r['prof_long']), float(r['prof_self'])]
    return f


def load():
    train = pd.read_csv(os.path.join(ROOT, 'train.csv'))
    test = pd.read_csv(os.path.join(ROOT, 'test.csv'))
    return train, test


def build(df):
    recs = extract_nodes(df)
    D = np.array([dense_row(r) for r in recs], dtype=np.float64)
    titles = [str(r['title']) for r in recs]
    forums = [str(r['forum']) for r in recs]
    return recs, D, titles, forums


def roles(recs):
    out = np.empty(len(recs), dtype=int)
    for i, r in enumerate(recs):
        L = r['route_len']; pos = r['pos']
        out[i] = 0 if pos == 0 else (2 if pos == L - 1 else 1)
    return out


def make_vecs():
    tvecs = [
        TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        lowercase=True, strip_accents='unicode'),
        TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                        lowercase=True),
    ]
    fvec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                           lowercase=True)
    return tvecs, fvec


def fit_text(titles_tr, forums_tr):
    tvecs, fvec = make_vecs()
    for v in tvecs:
        v.fit(titles_tr)
    fvec.fit(forums_tr)
    return tvecs, fvec


def tf_text(tvecs, fvec, titles, forums):
    mats = [v.transform(titles) for v in tvecs] + [fvec.transform(forums)]
    return sparse.hstack(mats).tocsr()


def predict_full(clf, X):
    p = clf.predict_proba(X)
    full = np.full((X.shape[0], 5), 1e-6)
    for j, c in enumerate(clf.classes_):
        full[:, int(c)] = p[:, j]
    full /= full.sum(1, keepdims=True)
    return full


# ---------- official in-process scorer ----------
def ctx(train):
    ids = train['sample_id'].tolist()
    lens = {r['sample_id']: len(r['masked_nodes'].split()) for _, r in train.iterrows()}
    targets = {r['sample_id']: parse_target(r['target_sequence'])[0] for _, r in train.iterrows()}
    folds = make_folds(train); fold_of = dict(zip(ids, folds))
    pars = {r['sample_id']: solve_parents(r) for _, r in train.iterrows()}
    truth = train[['sample_id', 'target_sequence']]
    tbf = {f: fit_transitions(targets, [s for s in ids if fold_of[s] != f]) for f in range(5)}
    return ids, lens, targets, fold_of, pars, truth, tbf


def official(P, ids, lens, fold_of, pars, truth, tbf, mode='viterbi'):
    o = 0; pbr = {}
    for sid in ids:
        L = lens[sid]; pbr[sid] = P[o:o + L]; o += L
    seqs = {sid: decode_row(pbr[sid], tbf[fold_of[sid]], mode=mode) for sid in ids}
    sub = format_submission(ids, [seqs[s] for s in ids], [pars[s] for s in ids])
    s, comps = score_submission(sub, truth)
    return s, comps


def main():
    t0 = time.time()
    train, test = load()
    recs_tr, Dtr, tit_tr, for_tr = build(train)
    recs_te, Dte, tit_te, for_te = build(test)
    role_tr = roles(recs_tr); role_te = roles(recs_te)
    tmap = {r['sample_id']: parse_target(r['target_sequence'])[0] for _, r in train.iterrows()}
    y = np.array([TYPE_IDX[tmap[r['sample_id']][r['pos']]] for r in recs_tr])
    ids, lens, targets, fold_of, pars, truth, tbf = ctx(train)
    node_fold = np.array([fold_of[r['sample_id']] for r in recs_tr])

    cache = {}
    for f in range(5):
        trm = node_fold != f
        idx_tr = np.where(trm)[0]
        tvecs, fvec = fit_text([tit_tr[i] for i in idx_tr], [for_tr[i] for i in idx_tr])
        Xtr_t = tf_text(tvecs, fvec, tit_tr, for_tr)
        Xte_t = tf_text(tvecs, fvec, tit_te, for_te)
        sc = StandardScaler().fit(Dtr[trm])
        Dtr_s = sc.transform(Dtr); Dte_s = sc.transform(Dte)
        cache[f] = dict(trm=trm, vam=(node_fold == f), Xtr_t=Xtr_t, Xte_t=Xte_t,
                        Dtr_s=Dtr_s, Dte_s=Dte_s)
    print('cache built %.1fs' % (time.time() - t0))

    def build_X(block_t, block_d, use_text):
        if use_text:
            return sparse.hstack([block_t, sparse.csr_matrix(block_d)]).tocsr()
        return sparse.csr_matrix(block_d)

    def run_config(cfg):
        role_spec = cfg['role_spec']; C = cfg['C']; cw = cfg['cw']; use_text = cfg['use_text']
        solver = cfg.get('solver', 'lbfgs'); est = cfg.get('est', 'lr')
        oof = np.zeros((len(recs_tr), 5)); test_acc = np.zeros((len(recs_te), 5))
        for f in range(5):
            c = cache[f]; trm = c['trm']; vam = c['vam']
            Xtr = build_X(c['Xtr_t'][trm], c['Dtr_s'][trm], use_text)
            Xva = build_X(c['Xtr_t'][vam], c['Dtr_s'][vam], use_text)
            Xte = build_X(c['Xte_t'], c['Dte_s'], use_text)
            ytr = y[trm]

            def mk():
                if est == 'lr':
                    return LogisticRegression(C=C, class_weight=cw, solver=solver,
                                              max_iter=500, tol=1e-3)
                base = LinearSVC(C=C, class_weight=cw, max_iter=5000)
                return CalibratedClassifierCV(base, method='sigmoid', cv=3)

            if role_spec:
                pv = np.zeros((int(vam.sum()), 5)); pt = np.zeros((len(recs_te), 5))
                rtr = role_tr[trm]; rva = role_tr[vam]
                for rr in [0, 1, 2]:
                    m = mk(); m.fit(Xtr[rtr == rr], ytr[rtr == rr])
                    if (rva == rr).sum() > 0:
                        pv[rva == rr] = predict_full(m, Xva[rva == rr])
                    pt[role_te == rr] = predict_full(m, Xte[role_te == rr])
                oof[vam] = pv; test_acc += pt
            else:
                m = mk(); m.fit(Xtr, ytr)
                oof[vam] = predict_full(m, Xva)
                test_acc += predict_full(m, Xte)
        test_acc /= 5
        sv, cv = official(oof, ids, lens, fold_of, pars, truth, tbf, 'viterbi')
        sp, cp = official(oof, ids, lens, fold_of, pars, truth, tbf, 'posterior')
        return oof, test_acc, sv, cv, sp, cp

    grid = []
    for role_spec in [False, True]:
        for C in [0.3, 1.0, 3.0]:
            for cw in [None, 'balanced']:
                grid.append(dict(role_spec=role_spec, C=C, cw=cw, use_text=True,
                                 est='lr', solver='lbfgs'))
    best = None
    for cfg in grid:
        oof, ta, sv, cv, sp, cp = run_config(cfg)
        tag = "rs=%d C=%s cw=%s" % (int(cfg['role_spec']), cfg['C'], cfg['cw'])
        best_mode = 'viterbi' if sv >= sp else 'posterior'
        bscore = max(sv, sp)
        print("%-30s vit=%.3f post=%.3f  TypeF1=%.3f Anch=%.3f | best=%.3f(%s) [%.0fs]" %
              (tag, sv, sp, cv['TypeMacroF1'], cv['AnchorMacroF1'], bscore, best_mode, time.time() - t0))
        if best is None or bscore > best[0]:
            best = (bscore, best_mode, cfg, oof, ta, sv, sp, cv, cp)
    print('\nBEST:', best[2], 'score=%.4f' % best[0], 'mode=', best[1])
    bscore, bmode, bcfg, oof, ta, sv, sp, cv, cp = best
    np.save(os.path.expanduser('~/discourse/runs/textlin/oof_probs.npy'), oof.astype(np.float32))
    np.save(os.path.expanduser('~/discourse/runs/textlin/test_probs.npy'), ta.astype(np.float32))
    comps = cv if bmode == 'viterbi' else cp
    out = dict(score=bscore, mode=bmode, viterbi=sv, posterior=sp,
               TypeMacroF1=comps['TypeMacroF1'], AnchorMacroF1=comps['AnchorMacroF1'],
               TypeScore=comps['TypeScore'], OrderedScore=comps['OrderedScore'],
               ParentScore=comps['ParentScore'], config=bcfg,
               type_percls={k: round(v, 3) for k, v in comps['type_percls'].items()},
               anchor_percls={k: round(v, 3) for k, v in comps['anchor_percls'].items()})
    json.dump(out, open(os.path.expanduser('~/discourse/runs/textlin/score.json'), 'w'), indent=2)
    print('saved. total %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
