import sys, os, collections, json, time
sys.path.insert(0, os.path.expanduser('~/discourse/foundation'))
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import (TYPES, TYPE_IDX, parse_target, extract_nodes, make_folds,
                    format_submission, solve_parents)
from decode import fit_transitions, decode_row
from scorer import score_submission

ROOT = os.path.expanduser('~/discourse/dataset/public')
VIS6 = ['answer', 'elaboration', 'question', 'appreciation', 'agreement', 'other']
PARTYPES = ['answer', 'elaboration', 'question', 'appreciation', 'agreement', 'other', 'ROOT', 'MASK']


def dense_row(r):  # ORIGINAL feature set (best-performing)
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
    f.append(float(nd == 0 and r['n_masked_kids'] == 0))
    f += [float(r['title_len']), float(r['title_words']), float(r['title_q']),
          float(r['title_excl']), float(r['title_wh'])]
    f += [float(r['prof_wide']), float(r['prof_long']), float(r['prof_self'])]
    return f


def build(df):
    recs = extract_nodes(df)
    D = np.array([dense_row(r) for r in recs], dtype=np.float64)
    return recs, D, [str(r['title']) for r in recs], [str(r['forum']) for r in recs]


def make_vecs():  # ORIGINAL text vectorizers
    return ([TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                             lowercase=True, strip_accents='unicode'),
             TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                             lowercase=True)],
            TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                            lowercase=True))


def fit_text(tit_tr, for_tr):
    tvecs, fvec = make_vecs()
    for v in tvecs:
        v.fit(tit_tr)
    fvec.fit(for_tr)
    return tvecs, fvec


def tf_text(tvecs, fvec, tit, forum):
    return sparse.hstack([v.transform(tit) for v in tvecs] + [fvec.transform(forum)]).tocsr()


def predict_full(clf, X):
    p = clf.predict_proba(X)
    full = np.full((X.shape[0], 5), 1e-6)
    for j, c in enumerate(clf.classes_):
        full[:, int(c)] = p[:, j]
    full /= full.sum(1, keepdims=True)
    return full


def ctx(train):
    ids = train['sample_id'].tolist()
    lens = {r['sample_id']: len(r['masked_nodes'].split()) for _, r in train.iterrows()}
    targets = {r['sample_id']: parse_target(r['target_sequence'])[0] for _, r in train.iterrows()}
    fold_of = dict(zip(ids, make_folds(train)))
    pars = {r['sample_id']: solve_parents(r) for _, r in train.iterrows()}
    truth = train[['sample_id', 'target_sequence']]
    tbf = {f: fit_transitions(targets, [s for s in ids if fold_of[s] != f]) for f in range(5)}
    return ids, lens, fold_of, pars, truth, tbf


def official(P, ids, lens, fold_of, pars, truth, tbf, mode):
    o = 0; pbr = {}
    for sid in ids:
        pbr[sid] = P[o:o + lens[sid]]; o += lens[sid]
    seqs = {sid: decode_row(pbr[sid], tbf[fold_of[sid]], mode=mode) for sid in ids}
    return score_submission(format_submission(ids, [seqs[s] for s in ids], [pars[s] for s in ids]), truth)


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(ROOT, 'train.csv'))
    test = pd.read_csv(os.path.join(ROOT, 'test.csv'))
    recs_tr, Dtr, tit_tr, for_tr = build(train)
    recs_te, Dte, tit_te, for_te = build(test)
    tmap = {r['sample_id']: parse_target(r['target_sequence'])[0] for _, r in train.iterrows()}
    y = np.array([TYPE_IDX[tmap[r['sample_id']][r['pos']]] for r in recs_tr])
    ids, lens, fold_of, pars, truth, tbf = ctx(train)
    node_fold = np.array([fold_of[r['sample_id']] for r in recs_tr])

    cache = {}
    for f in range(5):
        trm = node_fold != f
        idx = np.where(trm)[0]
        tvecs, fvec = fit_text([tit_tr[i] for i in idx], [for_tr[i] for i in idx])
        sc = StandardScaler().fit(Dtr[trm])
        cache[f] = dict(trm=trm, vam=(node_fold == f),
                        Xtr_t=tf_text(tvecs, fvec, tit_tr, for_tr),
                        Xte_t=tf_text(tvecs, fvec, tit_te, for_te),
                        Dtr_s=sc.transform(Dtr), Dte_s=sc.transform(Dte))
    print('orig-feat cache built dense=%d [%.0fs]' % (Dtr.shape[1], time.time() - t0), flush=True)

    def bx(bt, bd):
        return sparse.hstack([bt, sparse.csr_matrix(bd)]).tocsr()

    def compute(C, cw):
        oof = np.zeros((len(recs_tr), 5)); ta = np.zeros((len(recs_te), 5))
        for f in range(5):
            c = cache[f]; trm = c['trm']; vam = c['vam']
            m = LogisticRegression(C=C, class_weight=cw, solver='lbfgs', max_iter=500, tol=1e-3)
            m.fit(bx(c['Xtr_t'][trm], c['Dtr_s'][trm]), y[trm])
            oof[vam] = predict_full(m, bx(c['Xtr_t'][vam], c['Dtr_s'][vam]))
            ta += predict_full(m, bx(c['Xte_t'], c['Dte_s']))
        return oof, ta / 5

    def ev(P):
        sv, cv = official(P, ids, lens, fold_of, pars, truth, tbf, 'viterbi')
        sp, cp = official(P, ids, lens, fold_of, pars, truth, tbf, 'posterior')
        return sv, cv, sp, cp

    store = {}
    best = (53.139, 'viterbi', 'PREV(origC0.3bal)', None, None, None)
    for cw in ['balanced', None]:
        for C in [0.15, 0.2, 0.25, 0.3, 0.4]:
            oof, ta = compute(C, cw)
            store[(cw, C)] = (oof, ta)
            sv, cv, sp, cp = ev(oof)
            print('cw=%-8s C=%.2f  vit=%.3f post=%.3f  TypeF1=%.3f Anch=%.3f [%.0fs]' %
                  (str(cw), C, sv, sp, cv['TypeMacroF1'], cv['AnchorMacroF1'], time.time() - t0), flush=True)
            bm, bs = ('viterbi', sv) if sv >= sp else ('posterior', sp)
            if bs > best[0]:
                best = (bs, bm, ('single', cw, C), oof, ta, cv if bm == 'viterbi' else cp)
    for C in [0.15, 0.2, 0.25, 0.3, 0.4]:
        ob, tb = store[('balanced', C)]; on, tn = store[(None, C)]
        oof = np.sqrt(np.clip(ob, 1e-9, 1) * np.clip(on, 1e-9, 1)); oof /= oof.sum(1, keepdims=True)
        ta = np.sqrt(np.clip(tb, 1e-9, 1) * np.clip(tn, 1e-9, 1)); ta /= ta.sum(1, keepdims=True)
        sv, cv, sp, cp = ev(oof)
        print('BLEND C=%.2f  vit=%.3f post=%.3f  TypeF1=%.3f Anch=%.3f [%.0fs]' %
              (C, sv, sp, cv['TypeMacroF1'], cv['AnchorMacroF1'], time.time() - t0), flush=True)
        bm, bs = ('viterbi', sv) if sv >= sp else ('posterior', sp)
        if bs > best[0]:
            best = (bs, bm, ('blend', C), oof, ta, cv if bm == 'viterbi' else cp)

    print('\nBEST:', best[2], 'score=%.4f' % best[0], 'mode=', best[1], flush=True)
    if best[3] is not None:
        bs, bm, bcfg, oof, ta, comps = best
        np.save(os.path.expanduser('~/discourse/runs/textlin/oof_probs.npy'), oof.astype(np.float32))
        np.save(os.path.expanduser('~/discourse/runs/textlin/test_probs.npy'), ta.astype(np.float32))
        sv, cv, sp, cp = ev(oof)
        out = dict(score=bs, mode=bm, viterbi=sv, posterior=sp,
                   TypeMacroF1=comps['TypeMacroF1'], AnchorMacroF1=comps['AnchorMacroF1'],
                   TypeScore=comps['TypeScore'], OrderedScore=comps['OrderedScore'],
                   ParentScore=comps['ParentScore'], config=str(bcfg),
                   type_percls={k: round(v, 3) for k, v in comps['type_percls'].items()},
                   anchor_percls={k: round(v, 3) for k, v in comps['anchor_percls'].items()})
        json.dump(out, open(os.path.expanduser('~/discourse/runs/textlin/score.json'), 'w'), indent=2)
        print('SAVED new best config=%s' % str(bcfg), flush=True)
    else:
        print('no improvement; original artifacts (53.139) kept', flush=True)
    print('total %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
