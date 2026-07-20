"""Foundation baseline: count-based naive-Bayes emissions + position-conditioned
transition Viterbi, scored OOF with the official metric. Also scorer sanity
checks against the published reference baselines.
"""
import sys, os, collections
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TYPES, parse_target, extract_nodes, make_folds,
                    format_submission, solve_parents)
from scorer import score_submission

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'public')
train = pd.read_csv(os.path.join(ROOT, 'train.csv'))

truth = train[['sample_id', 'target_sequence']].copy()

# ---------- scorer sanity checks ----------
perfect, _ = score_submission(truth, truth)
print(f'[check] perfect submission on train: {perfect:.4f} (expect 100)')

const = truth.copy()
best = None
for t in TYPES:
    for L3 in [' '.join([f'type_{t} parent_root'] * 3)]:
        const['target_sequence'] = L3
        s, _ = score_submission(const, truth)
        if best is None or s > best[1]:
            best = (t, s)
print(f'[check] best constant route on train: type={best[0]} score={best[1]:.4f} (published test const: 13.46)')

# topology parents + constant type
tp_types, tp_pars = [], []
for _, r in train.iterrows():
    pars = solve_parents(r)
    tp_pars.append(pars)
    tp_types.append([best[0]] * len(pars))
sub = format_submission(train['sample_id'].tolist(), tp_types, tp_pars)
s, comps = score_submission(sub, truth)
print(f'[check] topology-parent + constant type on train: {s:.4f} (published: 22.82); ParentScore={comps["ParentScore"]:.4f}')

# ---------- NB emissions + transition Viterbi ----------
recs = extract_nodes(train)
by_row = collections.defaultdict(list)
for rec in recs:
    by_row[rec['sample_id']].append(rec)
targets = {}
for _, r in train.iterrows():
    targets[r['sample_id']] = parse_target(r['target_sequence'])[0]

folds = make_folds(train)
fold_of = dict(zip(train['sample_id'], folds))

ALPHA = 1.0


def fit_tables(ids):
    prior = collections.defaultdict(collections.Counter)      # (pos,len) -> type
    trans = collections.defaultdict(collections.Counter)      # (prev,pos,len) -> type
    trans_g = collections.defaultdict(collections.Counter)    # prev -> type
    child = collections.defaultdict(collections.Counter)      # type -> childtype counts
    child_tot = collections.Counter()                          # type -> total kids
    node_tot = collections.Counter()                           # type -> node count
    par0 = collections.defaultdict(collections.Counter)       # par_type -> type (pos0)
    q0 = collections.defaultdict(collections.Counter)         # title_q -> type (pos0)
    for sid in ids:
        ts = targets[sid]
        L = len(ts)
        for rec, t in zip(by_row[sid], ts):
            i = rec['pos']
            prior[(i, L)][t] += 1
            node_tot[t] += 1
            for ct, k in rec['kid_types'].items():
                child[t][ct] += k
                child_tot[t] += k
            if i > 0:
                trans[(ts[i - 1], i, L)][t] += 1
                trans_g[ts[i - 1]][t] += 1
            else:
                par0[rec['par_type']][t] += 1
                q0[rec['title_q']][t] += 1
    return dict(prior=prior, trans=trans, trans_g=trans_g, child=child,
                child_tot=child_tot, node_tot=node_tot, par0=par0, q0=q0)


def logp(counter, key, alpha=ALPHA):
    tot = sum(counter.values())
    return np.log((counter[key] + alpha) / (tot + alpha * len(TYPES)))


def emis_logodds(tb, rec):
    """log-likelihood-ratio evidence per type (children + pos0 cues)."""
    e = np.zeros(len(TYPES))
    total_nodes = sum(tb['node_tot'].values())
    for ti, t in enumerate(TYPES):
        # children NB: P(child=ct | type t) vs marginal P(child=ct)
        for ct, k in rec['kid_types'].items():
            p_ct_t = (tb['child'][t][ct] + 0.5) / (tb['child_tot'][t] + 0.5 * 6)
            marg = (sum(tb['child'][x][ct] for x in TYPES) + 0.5) / (sum(tb['child_tot'].values()) + 0.5 * 6)
            e[ti] += k * (np.log(p_ct_t) - np.log(marg))
        if rec['pos'] == 0:
            pr = np.array([ (tb['node_tot'][x] + 1) for x in TYPES], float)
            pr /= pr.sum()
            c = tb['par0'][rec['par_type']]
            if sum(c.values()) >= 5:
                e[ti] += logp(c, t) - np.log(pr[ti])
            c = tb['q0'][rec['title_q']]
            e[ti] += logp(c, t) - np.log(pr[ti])
    return e


def decode_row(tb, recs_row, L):
    E = np.array([emis_logodds(tb, rec) for rec in recs_row])
    # initial: prior at pos0
    init = np.array([logp(tb['prior'][(0, L)], t) for t in TYPES])
    dp = init + E[0]
    bp = np.zeros((L, len(TYPES)), int)
    for i in range(1, L):
        scores = np.zeros((len(TYPES), len(TYPES)))
        for pi, pt in enumerate(TYPES):
            c = tb['trans'][(pt, i, L)]
            cg = tb['trans_g'][pt]
            for ci, ct in enumerate(TYPES):
                if sum(c.values()) >= 8:
                    lt = logp(c, ct)
                else:
                    lt = 0.5 * logp(c, ct) + 0.5 * logp(cg, ct)
                scores[pi, ci] = dp[pi] + lt + E[i][ci]
        bp[i] = scores.argmax(0)
        dp = scores.max(0)
    seq = [int(dp.argmax())]
    for i in range(L - 1, 0, -1):
        seq.append(int(bp[i][seq[-1]]))
    seq.reverse()
    return [TYPES[k] for k in seq]


oof_types = {}
for f in range(5):
    tr_ids = [sid for sid in train['sample_id'] if fold_of[sid] != f]
    va_ids = [sid for sid in train['sample_id'] if fold_of[sid] == f]
    tb = fit_tables(tr_ids)
    for sid in va_ids:
        rr = by_row[sid]
        oof_types[sid] = decode_row(tb, rr, len(rr))

ids = train['sample_id'].tolist()
pars_all = [solve_parents(r) for _, r in train.iterrows()]
sub = format_submission(ids, [oof_types[s] for s in ids], pars_all)
s, comps = score_submission(sub, truth, verbose=True)
print(f'\n[baseline NB+Viterbi OOF] {s:.4f} (published local-window transition: 38.96, public-feature ensemble: 41.57)')
