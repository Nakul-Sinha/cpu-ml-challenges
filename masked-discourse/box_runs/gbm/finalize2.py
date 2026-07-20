"""Final GBM: fixed-iteration full-4-fold training (max data, no early-stop
variance), light lgb-seed averaging. Compare vs balanced/none/blend, save best."""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train as T

t0 = time.time()
D = T.load_all(hashing=True)
Xtr, Xte, y = D['Xtr'], D['Xte'], D['ytr']
node_fold = D['node_fold']
NTR, NTE = Xtr.shape[0], Xte.shape[0]
W = dict(learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.9,
         subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0)
SEEDS = [42, 7, 123]

def cv_fixed(n_est, cw):
    oof = np.zeros((NTR, 5)); test = np.zeros((NTE, 5))
    for f in range(5):
        om = node_fold == f; tm = ~om
        for s in SEEDS:
            p = dict(W); p['random_state'] = s; p['n_estimators'] = n_est
            clf = lgb.LGBMClassifier(objective='multiclass', num_class=5,
                                     class_weight=cw, n_jobs=5, verbose=-1, **p)
            clf.fit(Xtr[tm], y[tm])
            oof[om] += clf.predict_proba(Xtr[om]) / len(SEEDS)
            test += clf.predict_proba(Xte) / (len(SEEDS) * 5)
    return oof, test

def score(oof):
    sv, cv = T.quick_score(oof, D, 'viterbi')
    sp, cp = T.quick_score(oof, D, 'posterior')
    return (sv, cv, sp, cp)

def geo(a, b, w):
    m = np.exp(w * np.log(np.clip(a, 1e-9, 1)) + (1 - w) * np.log(np.clip(b, 1e-9, 1)))
    return m / m.sum(1, keepdims=True)

store = {}
for n_est in [110, 140, 170]:
    oof, test = cv_fixed(n_est, 'balanced')
    sv, cv, sp, cp = score(oof)
    store[('bal', n_est)] = (oof, test, sv, sp, cv, cp)
    print('  bal n=%d  V=%.4f P=%.4f  TMF=%.4f AMF=%.4f' % (n_est, sv, sp, cv['TypeMacroF1'], cv['AnchorMacroF1']))
# none at best n for blend diversity
oof_n, test_n = cv_fixed(140, None)
svn, cvn, spn, cpn = score(oof_n)
print('  none n=140  V=%.4f P=%.4f' % (svn, spn))

# best balanced n
best_n = max([110, 140, 170], key=lambda n: max(store[('bal', n)][2], store[('bal', n)][3]))
oof_b, test_b = store[('bal', best_n)][0], store[('bal', best_n)][1]

cands = {}
cands['bal'] = (oof_b, test_b)
cands['blend0.8'] = (geo(oof_b, oof_n, 0.8), geo(test_b, test_n, 0.8))
cands['blend0.7'] = (geo(oof_b, oof_n, 0.7), geo(test_b, test_n, 0.7))
best_tag, best_score, best_mode, best_comps, best_oof, best_test = None, -1, None, None, None, None
for tag, (oof, test) in cands.items():
    sv, cv, sp, cp = score(oof)
    m, sc, cm = ('viterbi', sv, cv) if sv >= sp else ('posterior', sp, cp)
    print('  %-9s V=%.4f P=%.4f best=%.4f(%s) TMF=%.4f AMF=%.4f' % (tag, sv, sp, sc, m[0], cm['TypeMacroF1'], cm['AnchorMacroF1']))
    if sc > best_score:
        best_tag, best_score, best_mode, best_comps, best_oof, best_test = tag, sc, m, cm, oof, test

print('\nWINNER:', best_tag, 'n=%d' % best_n, 'mode=', best_mode, 'score=%.4f' % best_score)
np.save(os.path.join(HERE, 'oof_probs.npy'), best_oof.astype(np.float64))
np.save(os.path.join(HERE, 'test_probs.npy'), best_test.astype(np.float64))
out = {k: float(v) for k, v in best_comps.items() if not isinstance(v, dict)}
out['score'] = float(best_score); out['mode'] = best_mode
out['recipe'] = 'fixed_iter_seed_avg'; out['internal'] = best_tag; out['n_estimators'] = int(best_n)
out['seeds'] = SEEDS; out['config'] = W; out['class_weight'] = 'balanced'
out['n_features'] = int(NTR and Xtr.shape[1])
out['type_percls'] = {k: float(v) for k, v in best_comps['type_percls'].items()}
out['anchor_percls'] = {k: float(v) for k, v in best_comps['anchor_percls'].items()}
with open(os.path.join(HERE, 'score.json'), 'w') as fh:
    json.dump(out, fh, indent=2)
print('SAVED oof', best_oof.shape, 'test', best_test.shape, 'elapsed %.1fs' % (time.time() - t0))
