"""Masked Discourse Sequence Recovery — end-to-end solution.

Pipeline (everything fit from ./dataset/public/train.csv at run time):
  1. Parse thread_view cards; recover masked parents from graph topology
     (the view is a DFS pre-order listing, so a node's parent is the nearest
     preceding card at depth d-1; depth-1 nodes attach to the post card when
     present, else ROOT).
  2. Train three diverse per-node type-emission families with shared 5-fold CV:
       A. LightGBM multiclass on 183 dense structural/context features,
       B. Logistic regression on title/forum TF-IDF + 90 dense features,
       C. Torch BiGRU + MLP joint-route taggers (seed-averaged),
     producing out-of-fold probabilities for every masked train node and
     fold-averaged probabilities for test nodes.
  3. Blend the family probabilities (log-linear weights selected on OOF with
     the exact competition metric), then train a stacked logistic-regression
     meta-model on [family log-probs | blend | neighbor-node blend probs |
     structural features | TF-IDF text] with the same fold protocol, plus a
     dedicated anchor-position specialist geo-mixed at the route's last node.
  4. Decode each route with position-conditioned label transitions estimated
     from the training targets (max-marginal / Viterbi, mode selected on OOF)
     and write the alternating type/parent token sequences.

A wall-clock budget manager keeps the run inside the grading limit and a
fallback path always writes a valid submission.

Runtime command: python3 solution.py <public_dir> <submission_out>
(Both arguments are optional; without them the script auto-discovers the data
directory and writes ./working/submission.csv.)
"""
import os
import re
import sys
import time
import json
import collections
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

T0 = time.time()
BUDGET_S = float(os.environ.get('SOL_BUDGET_S', 3000))


def time_left():
    return BUDGET_S - (time.time() - T0)


def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)


# ---------------------------------------------------------------------------
# data discovery
# ---------------------------------------------------------------------------

def find_data_root():
    cands = []
    if len(sys.argv) > 1 and sys.argv[1]:
        cands.append(sys.argv[1])
    cands += ['dataset/public', 'dataset', '.', './public', '../dataset/public',
              '/kaggle/input']
    for c in cands:
        if os.path.exists(os.path.join(c, 'train.csv')) and \
           os.path.exists(os.path.join(c, 'test.csv')):
            return c
    for base, _dirs, files in os.walk('.'):
        if 'train.csv' in files and 'test.csv' in files:
            return base
    for base, _dirs, files in os.walk('/kaggle/input'):
        if 'train.csv' in files and 'test.csv' in files:
            return base
    raise FileNotFoundError('train.csv/test.csv not found')


# ---------------------------------------------------------------------------
# parsing + topology
# ---------------------------------------------------------------------------

TYPES = ['answer', 'elaboration', 'question', 'appreciation', 'agreement']
TYPE_IDX = {t: i for i, t in enumerate(TYPES)}
VIS_TYPES = TYPES + ['other']
NT = 5
CARD = re.compile(r'^(N\d+)\[d=(-?\d+),p=([A-Za-z0-9_]+),t=([A-Za-z_]+)\]$')
WH_WORDS = ('what', 'why', 'how', 'who', 'when', 'where', 'which', 'is ', 'are ',
            'do ', 'does ', 'can ', 'should ', 'anyone', 'any ')


def parse_view(view):
    nodes = collections.OrderedDict()
    for i, card in enumerate(view.split('||')):
        m = CARD.match(card.strip())
        nodes[m.group(1)] = dict(d=int(m.group(2)), p=m.group(3), t=m.group(4), idx=i)
    return nodes


def parse_target(ts):
    toks = str(ts).split()
    return ([t[5:] for t in toks if t.startswith('type_')],
            [t[7:] for t in toks if t.startswith('parent_')])


def dfs_parent(nodes, nid):
    order = list(nodes.keys())
    i = nodes[nid]['idx']
    d = nodes[nid]['d']
    for j in range(i - 1, -1, -1):
        if nodes[order[j]]['d'] == d - 1:
            return order[j].lower()
    if d == 1:
        for j in range(i - 1, -1, -1):
            if nodes[order[j]]['d'] in (-1, 0):
                return order[j].lower()
        return 'root'
    return 'root'


def solve_parents(row):
    nodes = parse_view(row['thread_view'])
    return [dfs_parent(nodes, nid) for nid in row['masked_nodes'].split()]


def build_children(nodes):
    ch = collections.defaultdict(list)
    for k, v in nodes.items():
        if v['p'] in nodes:
            ch[v['p']].append(k)
    return ch


def descendants(nodes, ch, nid):
    out = []
    stack = list(ch.get(nid, []))
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(ch.get(c, []))
    return out


def extract_row_nodes(row):
    nodes = parse_view(row['thread_view'])
    ch = build_children(nodes)
    masked = row['masked_nodes'].split()
    pred_parents = [dfs_parent(nodes, nid) for nid in masked]
    title = str(row['thread_title'])
    forum = str(row['forum'])
    profile = str(row['thread_profile']).split()
    vis_counts = collections.Counter(v['t'] for v in nodes.values() if v['t'] != 'MASK')
    n_nodes = len(nodes)
    L = len(masked)
    recs = []
    for i, nid in enumerate(masked):
        nd = nodes[nid]
        d = nd['d']
        kids = ch.get(nid, [])
        kid_types = collections.Counter(nodes[k]['t'] for k in kids)
        desc = descendants(nodes, ch, nid)
        desc_types = collections.Counter(nodes[k]['t'] for k in desc)
        par_tok = pred_parents[i]
        if par_tok == 'root':
            par_kind, par_type, par_depth = 'root', 'ROOT', 0
        else:
            pn = par_tok.upper()
            pt = nodes[pn]['t']
            par_depth = nodes[pn]['d']
            if pt == 'MASK':
                par_kind, par_type = 'masked', 'MASK'
            elif nodes[pn]['d'] in (-1, 0):
                par_kind, par_type = 'post', pt
            else:
                par_kind, par_type = 'visible', pt
        sibs = [k for k, v in nodes.items()
                if v['p'] == par_tok.upper() and k != nid] if par_tok != 'root' else \
               [k for k, v in nodes.items() if v['p'] == 'ROOT' and k != nid]
        sib_types = collections.Counter(nodes[k]['t'] for k in sibs)
        gap_prev = d - nodes[masked[i - 1]]['d'] if i > 0 else 0
        gap_next = nodes[masked[i + 1]]['d'] - d if i < L - 1 else 0
        between = []
        if i > 0 and gap_prev > 1:
            cur = par_tok
            while cur not in ('root',) and cur.upper() in nodes:
                cn = cur.upper()
                if cn == masked[i - 1]:
                    break
                if nodes[cn]['t'] != 'MASK':
                    between.append(nodes[cn]['t'])
                cur = dfs_parent(nodes, cn) if nodes[cn]['p'] == 'MASK' else nodes[cn]['p'].lower()
        recs.append(dict(
            sample_id=row['sample_id'], nid=nid, pos=i, route_len=L, depth=d,
            pred_parent=par_tok, par_kind=par_kind, par_type=par_type, par_depth=par_depth,
            n_kids_vis=sum(kid_types.values()), kid_types=dict(kid_types),
            n_desc_vis=sum(desc_types.values()), desc_types=dict(desc_types),
            n_masked_kids=sum(1 for k in kids if nodes[k]['t'] == 'MASK'),
            sib_types=dict(sib_types), n_sibs=len(sibs),
            gap_prev=gap_prev, gap_next=gap_next, between=between,
            n_nodes=n_nodes, vis_counts=dict(vis_counts),
            view_idx=nd['idx'], view_frac=nd['idx'] / max(n_nodes - 1, 1),
            max_depth=max(v['d'] for v in nodes.values()),
            n_out=sum(1 for v in nodes.values() if v['p'] == 'OUT'),
            has_post=any(v['d'] in (-1, 0) for v in nodes.values()),
            title=title, forum=forum,
            title_len=len(title), title_words=len(title.split()),
            title_q='?' in title, title_excl='!' in title,
            title_wh=title.lower().startswith(WH_WORDS),
            prof_wide='wide' in profile, prof_long='long' in profile,
            prof_self='self_post' in profile,
        ))
    return recs


def extract_nodes(df):
    recs = []
    for _, row in df.iterrows():
        recs.extend(extract_row_nodes(row))
    return recs


def make_folds(train, n_folds=5, seed=42):
    anchors = []
    for _, r in train.iterrows():
        tt, _ = parse_target(r['target_sequence'])
        anchors.append(tt[-1] + '_' + str(len(tt)))
    anchors = np.array(anchors)
    rng = np.random.RandomState(seed)
    folds = np.zeros(len(train), dtype=int)
    for s in np.unique(anchors):
        idx = np.where(anchors == s)[0]
        rng.shuffle(idx)
        for k, j in enumerate(idx):
            folds[j] = k % n_folds
    return folds


# ---------------------------------------------------------------------------
# transitions + batch decoding
# ---------------------------------------------------------------------------

def fit_transitions(targets_by_id, ids, alpha=1.0):
    trans = collections.defaultdict(collections.Counter)
    trans_g = collections.defaultdict(collections.Counter)
    init = collections.defaultdict(collections.Counter)
    for sid in ids:
        ts = targets_by_id[sid]
        L = len(ts)
        init[L][ts[0]] += 1
        for i in range(1, L):
            trans[(ts[i - 1], i, L)][ts[i]] += 1
            trans_g[ts[i - 1]][ts[i]] += 1

    def logp(counter, key):
        tot = sum(counter.values())
        return np.log((counter[key] + alpha) / (tot + alpha * NT))

    T, I = {}, {}
    for L in (3, 4):
        I[L] = np.array([logp(init[L], t) for t in TYPES])
        T[L] = [None] * L
        for i in range(1, L):
            M = np.zeros((NT, NT))
            for pi, pt in enumerate(TYPES):
                c = trans[(pt, i, L)]
                for ci, ct in enumerate(TYPES):
                    if sum(c.values()) >= 8:
                        M[pi, ci] = logp(c, ct)
                    else:
                        M[pi, ci] = 0.5 * logp(c, ct) + 0.5 * logp(trans_g[pt], ct)
            T[L][i] = M
    return T, I


def batch_viterbi(E, T, I):
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
    A = np.zeros((R, L, NT))
    A[:, 0] = I[None, :] + E[:, 0]
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


def decode_matrix(P, lens, tables, mode):
    T, I = tables
    offs = np.concatenate([[0], np.cumsum(lens)])
    E_all = np.log(np.clip(P, 1e-9, 1.0))
    n = len(lens)
    out = [None] * n
    for L in (3, 4):
        idx = [r for r in range(n) if lens[r] == L]
        if not idx:
            continue
        E = np.stack([E_all[offs[r]:offs[r] + L] for r in idx])
        dec = batch_viterbi(E, T[L], I[L]) if mode == 'viterbi' else batch_posterior(E, T[L], I[L])
        for k, r in enumerate(idx):
            out[r] = dec[k]
    return out


# ---------------------------------------------------------------------------
# exact metric (for in-script OOF selection only)
# ---------------------------------------------------------------------------

class FastScorer:
    def __init__(self, true_types_by_row, true_pars_by_row, pred_pars_by_row):
        self.n = len(true_types_by_row)
        self.Ls = np.array([len(t) for t in true_types_by_row])
        self.true_types = true_types_by_row
        self.par_eq = [np.array([a == b for a, b in zip(p, t)])
                       for p, t in zip(pred_pars_by_row, true_pars_by_row)]
        self.par_score = float(np.mean([pe.mean() for pe in self.par_eq]))
        vocab = {}

        def pid(tok):
            if tok not in vocab:
                vocab[tok] = 5 + len(vocab)
            return vocab[tok]

        self.true_seq = []
        self.pred_par_ids = []
        for tt, tp, pp in zip(true_types_by_row, true_pars_by_row, pred_pars_by_row):
            s = []
            for ty, pa in zip(tt, tp):
                s.append(ty)
                s.append(pid(pa))
            self.true_seq.append(np.array(s))
            self.pred_par_ids.append(np.array([pid(x) for x in pp]))
        self.groups = {L: np.where(self.Ls == L)[0] for L in np.unique(self.Ls)}
        self.tt_arr = {L: np.stack([np.array(self.true_types[i]) for i in idx])
                       for L, idx in self.groups.items()}
        self.ts_arr = {L: np.stack([self.true_seq[i] for i in idx])
                       for L, idx in self.groups.items()}
        self.pp_arr = {L: np.stack([self.pred_par_ids[i] for i in idx])
                       for L, idx in self.groups.items()}

    def score(self, pred_by_row):
        tp = np.zeros(NT); fp = np.zeros(NT); fn = np.zeros(NT)
        atp = np.zeros(NT); afp = np.zeros(NT); afn = np.zeros(NT)
        type_pos_sum = 0.0
        ordered_sum = 0.0
        for L, idx in self.groups.items():
            P = np.stack([pred_by_row[i] for i in idx])
            T = self.tt_arr[L]
            type_pos_sum += (P == T).mean(1).sum()
            for c in range(NT):
                tp[c] += ((P == c) & (T == c)).sum()
                fp[c] += ((P == c) & (T != c)).sum()
                fn[c] += ((P != c) & (T == c)).sum()
                atp[c] += ((P[:, -1] == c) & (T[:, -1] == c)).sum()
                afp[c] += ((P[:, -1] == c) & (T[:, -1] != c)).sum()
                afn[c] += ((P[:, -1] != c) & (T[:, -1] == c)).sum()
            R = len(idx)
            A = np.empty((R, 2 * L), dtype=np.int64)
            A[:, 0::2] = P
            A[:, 1::2] = self.pp_arr[L]
            B = self.ts_arr[L]
            m = 2 * L
            dp = np.zeros((R, m + 1, m + 1), dtype=np.int16)
            for i in range(1, m + 1):
                Ai = A[:, i - 1][:, None]
                match = (Ai == B).astype(np.int16)
                dp[:, i, 1:] = np.maximum(dp[:, i - 1, 1:], dp[:, i - 1, :-1] + match)
                np.maximum.accumulate(dp[:, i, :], axis=1, out=dp[:, i, :])
            l = dp[:, m, m].astype(float)
            f1 = l / m
            ordered_sum += f1.sum()

        def macro(a, b, c):
            den = 2 * a + b + c
            f = np.where(den > 0, 2 * a / np.clip(den, 1e-12, None), 0.0)
            return f.mean()

        tm = macro(tp, fp, fn)
        am = macro(atp, afp, afn)
        comps = dict(TypeMacroF1=float(tm), AnchorMacroF1=float(am),
                     TypeScore=float(type_pos_sum / self.n),
                     OrderedScore=float(ordered_sum / self.n),
                     ParentScore=self.par_score)
        score = 100 * float(np.clip(0.45 * tm + 0.25 * am + 0.10 * comps['TypeScore']
                                    + 0.15 * comps['OrderedScore'] + 0.05 * self.par_score, 0, 1))
        return score, comps


# ---------------------------------------------------------------------------
# family A: LightGBM dense features
# ---------------------------------------------------------------------------

PAR_TYPES = ['ROOT', 'MASK'] + VIS_TYPES
PAR_KINDS = ['root', 'masked', 'post', 'visible']


def _fnv(s):
    h = 2166136261
    for c in s:
        h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
    return h


def _hash_char_ngrams(s, n, nbuck, out, base):
    s = '^' + str(s).lower() + '$'
    for i in range(len(s) - n + 1):
        out[base + (_fnv(s[i:i + n]) % nbuck)] += 1.0


def gbm_features(recs, title_buck=48, forum_buck=32):
    names = []
    scal = ['pos', 'route_len', 'pos_frac', 'is_first', 'is_last', 'is_interior',
            'depth', 'par_depth', 'gap_prev', 'gap_next', 'depth_frac',
            'n_kids_vis', 'n_masked_kids', 'has_kid', 'n_desc_vis', 'n_sibs',
            'n_nodes', 'max_depth', 'n_out', 'has_post', 'view_idx', 'view_frac',
            'title_len', 'title_words', 'title_q', 'title_excl', 'title_wh',
            'prof_wide', 'prof_long', 'prof_self']
    names += scal
    names += ['park_' + k for k in PAR_KINDS]
    names += ['part_' + t for t in PAR_TYPES]
    for grp in ('kidc', 'kidh', 'descc', 'sibc', 'btwc', 'viewc', 'viewf'):
        names += [grp + '_' + t for t in VIS_TYPES]
    inter = ['q_first', 'parq_first', 'para_first', 'kidans_first', 'kidans_interior',
             'kidans_last', 'kidans_posfrac', 'kidq_any', 'kidapp_any',
             'anyq_x_first', 'depth1_first']
    names += inter
    names += ['partf_' + t for t in PAR_TYPES]
    n_struct = len(names)
    names += ['tth_%d' % i for i in range(title_buck)]
    names += ['fmh_%d' % i for i in range(forum_buck)]
    F = len(names)
    col = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(recs), F), dtype=np.float32)
    for r, rec in enumerate(recs):
        L = rec['route_len']; pos = rec['pos']
        is_first = 1.0 if pos == 0 else 0.0
        is_last = 1.0 if pos == L - 1 else 0.0
        is_interior = 1.0 if (0 < pos < L - 1) else 0.0
        kt = rec['kid_types']; dt = rec['desc_types']; st = rec['sib_types']
        vc = rec['vis_counts']
        btw = collections.Counter(rec['between'])
        nkids = rec['n_kids_vis']
        depth = rec['depth']; md = rec['max_depth']
        ptype = rec['par_type']
        vals = {
            'pos': pos, 'route_len': L, 'pos_frac': pos / max(L - 1, 1),
            'is_first': is_first, 'is_last': is_last, 'is_interior': is_interior,
            'depth': depth, 'par_depth': rec['par_depth'],
            'gap_prev': rec['gap_prev'], 'gap_next': rec['gap_next'],
            'depth_frac': depth / max(md, 1),
            'n_kids_vis': nkids, 'n_masked_kids': rec['n_masked_kids'],
            'has_kid': 1.0 if nkids > 0 else 0.0,
            'n_desc_vis': rec['n_desc_vis'], 'n_sibs': rec['n_sibs'],
            'n_nodes': rec['n_nodes'], 'max_depth': md, 'n_out': rec['n_out'],
            'has_post': 1.0 if rec['has_post'] else 0.0,
            'view_idx': rec['view_idx'], 'view_frac': rec['view_frac'],
            'title_len': rec['title_len'], 'title_words': rec['title_words'],
            'title_q': 1.0 if rec['title_q'] else 0.0,
            'title_excl': 1.0 if rec['title_excl'] else 0.0,
            'title_wh': 1.0 if rec['title_wh'] else 0.0,
            'prof_wide': 1.0 if rec['prof_wide'] else 0.0,
            'prof_long': 1.0 if rec['prof_long'] else 0.0,
            'prof_self': 1.0 if rec['prof_self'] else 0.0,
        }
        for k, v in vals.items():
            X[r, col[k]] = v
        X[r, col['park_' + rec['par_kind']]] = 1.0
        if ptype in PAR_TYPES:
            X[r, col['part_' + ptype]] = 1.0
        for t in VIS_TYPES:
            X[r, col['kidc_' + t]] = kt.get(t, 0)
            X[r, col['kidh_' + t]] = 1.0 if kt.get(t, 0) > 0 else 0.0
            X[r, col['descc_' + t]] = dt.get(t, 0)
            X[r, col['sibc_' + t]] = st.get(t, 0)
            X[r, col['btwc_' + t]] = btw.get(t, 0)
            X[r, col['viewc_' + t]] = vc.get(t, 0)
            X[r, col['viewf_' + t]] = vc.get(t, 0) / max(rec['n_nodes'], 1)
        kidans = 1.0 if kt.get('answer', 0) > 0 else 0.0
        X[r, col['q_first']] = vals['title_q'] * is_first
        X[r, col['parq_first']] = (1.0 if ptype == 'question' else 0.0) * is_first
        X[r, col['para_first']] = (1.0 if ptype == 'answer' else 0.0) * is_first
        X[r, col['kidans_first']] = kidans * is_first
        X[r, col['kidans_interior']] = kidans * is_interior
        X[r, col['kidans_last']] = kidans * is_last
        X[r, col['kidans_posfrac']] = kidans * (pos / max(L - 1, 1))
        X[r, col['kidq_any']] = 1.0 if kt.get('question', 0) > 0 else 0.0
        X[r, col['kidapp_any']] = 1.0 if kt.get('appreciation', 0) > 0 else 0.0
        X[r, col['anyq_x_first']] = (1.0 if vc.get('question', 0) > 0 else 0.0) * is_first
        X[r, col['depth1_first']] = (1.0 if depth == 1 else 0.0) * is_first
        if ptype in PAR_TYPES:
            X[r, col['partf_' + ptype]] = is_first
        _hash_char_ngrams(rec['title'], 3, title_buck, X[r], n_struct)
        _hash_char_ngrams(rec['forum'], 3, forum_buck, X[r], n_struct + title_buck)
    return X


def run_gbm(Xtr, Xte, y, node_fold, seeds=(42, 7, 123), n_est=170):
    try:
        import lightgbm as lgb

        def fit_predict(tm, cw, seed):
            clf = lgb.LGBMClassifier(objective='multiclass', num_class=5,
                                     learning_rate=0.05, num_leaves=15,
                                     min_child_samples=20, subsample=0.9,
                                     subsample_freq=1, colsample_bytree=0.8,
                                     reg_lambda=1.0, n_estimators=n_est,
                                     class_weight=cw, random_state=seed,
                                     n_jobs=-1, verbose=-1)
            clf.fit(Xtr[tm], y[tm])
            return clf
    except Exception as e:
        log(f'lightgbm unavailable ({e}); falling back to HistGradientBoosting')
        from sklearn.ensemble import HistGradientBoostingClassifier

        def fit_predict(tm, cw, seed):
            clf = HistGradientBoostingClassifier(
                max_iter=n_est, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=1.0, random_state=seed,
                class_weight=cw)
            clf.fit(Xtr[tm], y[tm])
            return clf

    NTR, NTE = Xtr.shape[0], Xte.shape[0]
    out = {}
    for cw in ('balanced', None):
        oof = np.zeros((NTR, 5))
        test = np.zeros((NTE, 5))
        for f in range(5):
            om = node_fold == f
            tm = ~om
            for s in seeds:
                clf = fit_predict(tm, cw, s)
                oof[om] += clf.predict_proba(Xtr[om]) / len(seeds)
                test += clf.predict_proba(Xte) / (len(seeds) * 5)
        out[cw] = (oof, test)
    ob, tb = out['balanced']
    on, tn = out[None]

    def geo(a, b, w):
        m = np.exp(w * np.log(np.clip(a, 1e-9, 1)) + (1 - w) * np.log(np.clip(b, 1e-9, 1)))
        return m / m.sum(1, keepdims=True)

    return dict(bal=(ob, tb), blend07=(geo(ob, on, 0.7), geo(tb, tn, 0.7)),
                blend08=(geo(ob, on, 0.8), geo(tb, tn, 0.8)))


# ---------------------------------------------------------------------------
# family B: TF-IDF + logistic regression
# ---------------------------------------------------------------------------

def textlin_dense(recs):
    PARTYPES = VIS_TYPES + ['ROOT', 'MASK']
    rows = []
    for r in recs:
        L = r['route_len']; pos = r['pos']
        is0 = pos == 0; isanc = pos == L - 1; isint = (not is0) and (not isanc)
        f = [float(is0), float(isint), float(isanc),
             float(pos), float(L == 4), float(r['depth']), float(r['par_depth']),
             float(r['gap_prev']), float(r['gap_next']),
             float(r['view_idx']), float(r['view_frac']), float(r['n_nodes']),
             float(r['max_depth']), float(r['n_out']), float(r['has_post'])]
        for k in PAR_KINDS:
            f.append(float(r['par_kind'] == k))
        for t in PARTYPES:
            f.append(float(r['par_type'] == t))
        kt = r['kid_types']; nk = r['n_kids_vis']
        for t in VIS_TYPES:
            f.append(float(kt.get(t, 0)))
        for t in VIS_TYPES:
            f.append(kt.get(t, 0) / nk if nk > 0 else 0.0)
        f += [float(nk), float(r['n_masked_kids']), float(nk > 0)]
        dt = r['desc_types']; nd = r['n_desc_vis']
        for t in VIS_TYPES:
            f.append(float(dt.get(t, 0)))
        for t in VIS_TYPES:
            f.append(dt.get(t, 0) / nd if nd > 0 else 0.0)
        f += [float(nd), float(nd > 0)]
        st = r['sib_types']; ns = r['n_sibs']
        for t in VIS_TYPES:
            f.append(float(st.get(t, 0)))
        for t in VIS_TYPES:
            f.append(st.get(t, 0) / ns if ns > 0 else 0.0)
        f += [float(ns)]
        bt = collections.Counter(r['between'])
        for t in VIS_TYPES:
            f.append(float(bt.get(t, 0)))
        vc = r['vis_counts']; nn = r['n_nodes']
        for t in VIS_TYPES:
            f.append(vc.get(t, 0) / nn if nn > 0 else 0.0)
        f.append(float(nd == 0 and r['n_masked_kids'] == 0))
        f += [float(r['title_len']), float(r['title_words']), float(r['title_q']),
              float(r['title_excl']), float(r['title_wh'])]
        f += [float(r['prof_wide']), float(r['prof_long']), float(r['prof_self'])]
        rows.append(f)
    return np.array(rows, dtype=np.float64)


def make_text_vecs():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return ([TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2,
                             sublinear_tf=True, lowercase=True, strip_accents='unicode'),
             TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3,
                             sublinear_tf=True, lowercase=True)],
            TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2,
                            sublinear_tf=True, lowercase=True))


def predict_full(clf, X):
    p = clf.predict_proba(X)
    full = np.full((X.shape[0], 5), 1e-6)
    for j, c in enumerate(clf.classes_):
        full[:, int(c)] = p[:, j]
    return full / full.sum(1, keepdims=True)


def run_textlin(Dtr, Dte, tit_tr, for_tr, tit_te, for_te, y, node_fold, C=0.3):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from scipy import sparse
    oof = np.zeros((Dtr.shape[0], 5))
    test = np.zeros((Dte.shape[0], 5))
    for f in range(5):
        tm = node_fold != f
        vm = ~tm
        idx = np.where(tm)[0]
        tvecs, fvec = make_text_vecs()
        for v in tvecs:
            v.fit([tit_tr[i] for i in idx])
        fvec.fit([for_tr[i] for i in idx])

        def tf(tit, forum):
            return sparse.hstack([v.transform(tit) for v in tvecs] +
                                 [fvec.transform(forum)]).tocsr()

        sc = StandardScaler().fit(Dtr[tm])
        Xtr = sparse.hstack([tf([tit_tr[i] for i in np.where(tm)[0]],
                                [for_tr[i] for i in np.where(tm)[0]]),
                             sparse.csr_matrix(sc.transform(Dtr[tm]))]).tocsr()
        Xva = sparse.hstack([tf([tit_tr[i] for i in np.where(vm)[0]],
                                [for_tr[i] for i in np.where(vm)[0]]),
                             sparse.csr_matrix(sc.transform(Dtr[vm]))]).tocsr()
        Xte = sparse.hstack([tf(tit_te, for_te),
                             sparse.csr_matrix(sc.transform(Dte))]).tocsr()
        m = LogisticRegression(C=C, class_weight='balanced', solver='lbfgs',
                               max_iter=500, tol=1e-3)
        m.fit(Xtr, y[tm])
        oof[vm] = predict_full(m, Xva)
        test += predict_full(m, Xte) / 5
    return oof, test


# ---------------------------------------------------------------------------
# family C: torch joint-route taggers
# ---------------------------------------------------------------------------

def run_nnseq(train_df, test_df, folds_rows, seeds_per_arch):
    import hashlib
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(max(2, (os.cpu_count() or 8) - 2))
    PARTYPE = VIS_TYPES + ['MASK', 'ROOT']
    PARTYPE_IX = {t: i for i, t in enumerate(PARTYPE)}
    PARKIND_IX = {'root': 0, 'masked': 1, 'post': 2, 'visible': 3}

    def _hash_ngrams(s, n_lo, n_hi, nbuckets, prefix=''):
        s = (prefix + ' ' + str(s).lower()).strip()
        s = '^' + s + '$'
        out = []
        for n in range(n_lo, n_hi + 1):
            for i in range(len(s) - n + 1):
                out.append(int(hashlib.md5(s[i:i + n].encode()).hexdigest(), 16) % nbuckets)
        return out or [0]

    def node_dense(rec):
        kt = rec['kid_types']; dt = rec['desc_types']; st = rec['sib_types']
        vc = rec['vis_counts']
        bt = collections.Counter(rec['between'])
        nk = max(rec['n_kids_vis'], 1)
        nd = max(rec['n_desc_vis'], 1)
        L = rec['route_len']; pos = rec['pos']
        feats = [pos, L, rec['depth'], rec['par_depth'], rec['view_idx'], rec['view_frac'],
                 rec['max_depth'], rec['n_kids_vis'], rec['n_desc_vis'], rec['n_masked_kids'],
                 rec['n_sibs'], rec['n_nodes'], rec['n_out'], rec['gap_prev'], rec['gap_next'],
                 np.log1p(rec['title_len']), np.log1p(rec['title_words']),
                 float(rec['title_q']), float(rec['title_excl']), float(rec['title_wh']),
                 float(rec['prof_wide']), float(rec['prof_long']), float(rec['prof_self']),
                 float(rec['has_post']), float(pos == 0), float(pos == L - 1),
                 float(0 < pos < L - 1), rec['depth'] / max(rec['max_depth'], 1)]
        for dct in (kt,):
            feats += [float(dct.get(t, 0)) for t in VIS_TYPES]
        feats += [kt.get(t, 0) / nk for t in VIS_TYPES]
        feats += [float(dt.get(t, 0)) for t in VIS_TYPES]
        feats += [dt.get(t, 0) / nd for t in VIS_TYPES]
        feats += [float(st.get(t, 0)) for t in VIS_TYPES]
        feats += [float(vc.get(t, 0)) for t in VIS_TYPES]
        feats += [float(bt.get(t, 0)) for t in VIS_TYPES]
        feats += [float(kt.get('answer', 0) > 0), float(kt.get('question', 0) > 0),
                  float(kt.get('elaboration', 0) > 0), float(kt.get('appreciation', 0) > 0),
                  float(kt.get('agreement', 0) > 0), float(kt.get('other', 0) > 0),
                  float(dt.get('answer', 0) > 0), float(dt.get('question', 0) > 0),
                  float(rec['n_kids_vis'] == 0)]
        return feats

    def build_rows(df, targets=True):
        rows = []
        g = 0
        for _, r in df.iterrows():
            recs = extract_row_nodes(r)
            L = len(recs)
            dense = [node_dense(rc) for rc in recs]
            pt = [PARTYPE_IX.get(rc['par_type'], PARTYPE_IX['other']) for rc in recs]
            pk = [PARKIND_IX.get(rc['par_kind'], 3) for rc in recs]
            depthb = [min(max(rc['depth'], 0), 7) for rc in recs]
            poslen = [rc['pos'] + (0 if rc['route_len'] == 3 else 4) for rc in recs]
            role = [0 if rc['pos'] == 0 else (2 if rc['pos'] == rc['route_len'] - 1 else 1)
                    for rc in recs]
            title_ng = _hash_ngrams(recs[0]['title'], 3, 4, 4096)
            forum_ng = _hash_ngrams(recs[0]['forum'], 3, 5, 1024, prefix='F')
            yv = None
            if targets:
                ts = parse_target(r['target_sequence'])[0]
                yv = [TYPE_IDX[t] for t in ts]
            rows.append(dict(nidx=list(range(g, g + L)), L=L,
                             dense=np.array(dense, np.float32),
                             par_type=np.array(pt), par_kind=np.array(pk),
                             depthb=np.array(depthb), poslen=np.array(poslen),
                             role=np.array(role), title_ng=title_ng, forum_ng=forum_ng,
                             y=(np.array(yv) if yv is not None else None)))
            g += L
        return rows, g

    class Tagger(nn.Module):
        def __init__(self, D, seq):
            super().__init__()
            H = 96
            self.seq = seq
            self.dense_proj = nn.Linear(D, H)
            self.e_pt = nn.Embedding(len(PARTYPE), 8)
            self.e_pk = nn.Embedding(4, 4)
            self.e_db = nn.Embedding(8, 6)
            self.e_pl = nn.Embedding(8, 8)
            self.e_ro = nn.Embedding(3, 4)
            self.eb_t = nn.EmbeddingBag(4096, 20, mode='mean')
            self.eb_f = nn.EmbeddingBag(1024, 12, mode='mean')
            extra = 8 + 4 + 6 + 8 + 4 + 20 + 12
            self.pre = nn.Linear(H + extra, H)
            self.fdrop = nn.Dropout(0.1)
            self.drop = nn.Dropout(0.3)
            self.rnn = nn.GRU(H, H // 2, batch_first=True, bidirectional=True) if seq == 'gru' else None
            self.head = nn.Linear(H, NT)

        def forward(self, b):
            x = self.dense_proj(self.fdrop(b['dense']))
            cats = [self.e_pt(b['par_type']), self.e_pk(b['par_kind']),
                    self.e_db(b['depthb']), self.e_pl(b['poslen']), self.e_ro(b['role'])]
            tv = self.eb_t(b['t_flat'], b['t_off'])
            fv = self.eb_f(b['f_flat'], b['f_off'])
            L = x.shape[1]
            parts = [x] + cats + [tv.unsqueeze(1).expand(-1, L, -1),
                                  fv.unsqueeze(1).expand(-1, L, -1)]
            h = torch.cat(parts, -1)
            h = self.drop(F.relu(self.pre(h)))
            if self.rnn is not None:
                h, _ = self.rnn(h)
            return self.head(self.drop(h))

    def make_batch(rows, idx):
        dense = torch.tensor(np.stack([rows[i]['dense'] for i in idx]))

        def cat(k):
            return torch.tensor(np.stack([rows[i][k] for i in idx]), dtype=torch.long)

        b = dict(dense=dense, par_type=cat('par_type'), par_kind=cat('par_kind'),
                 depthb=cat('depthb'), poslen=cat('poslen'), role=cat('role'),
                 L=rows[idx[0]]['L'])
        for pre, key in (('t', 'title_ng'), ('f', 'forum_ng')):
            flat, off, o = [], [], 0
            for i in idx:
                off.append(o)
                ng = rows[i][key]
                flat.extend(ng)
                o += len(ng)
            b[pre + '_flat'] = torch.tensor(flat, dtype=torch.long)
            b[pre + '_off'] = torch.tensor(off, dtype=torch.long)
        y = None
        if rows[idx[0]]['y'] is not None:
            y = torch.tensor(np.stack([rows[i]['y'] for i in idx]), dtype=torch.long)
        return b, y

    def buckets(rows, idx):
        d = collections.defaultdict(list)
        for i in idx:
            d[rows[i]['L']].append(i)
        return d

    def predict(model, rows, idx):
        model.eval()
        out = {}
        with torch.no_grad():
            for L, ids2 in buckets(rows, idx).items():
                for s in range(0, len(ids2), 256):
                    chunk = ids2[s:s + 256]
                    b, _ = make_batch(rows, chunk)
                    p = F.softmax(model(b), -1).numpy()
                    for bi, i in enumerate(chunk):
                        out[i] = p[bi]
        return out

    def eval_f1(model, rows, idx):
        from sklearn.metrics import f1_score
        pr = predict(model, rows, idx)
        yt, yp = [], []
        for i in idx:
            yt.extend(rows[i]['y'])
            yp.extend(pr[i].argmax(1))
        return f1_score(yt, yp, average='macro')

    def train_one(rows, fit_idx, es_idx, seq, seed, cw):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = Tagger(rows[0]['dense'].shape[1], seq)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
        best_f1, best_state, bad = -1, None, 0
        tr_b = buckets(rows, fit_idx)
        for ep in range(60):
            model.train()
            order = []
            for L, ids2 in tr_b.items():
                ids2 = ids2.copy()
                np.random.shuffle(ids2)
                for s in range(0, len(ids2), 64):
                    order.append(ids2[s:s + 64])
            np.random.shuffle(order)
            for chunk in order:
                b, y = make_batch(rows, chunk)
                logits = model(b)
                loss = F.cross_entropy(logits.reshape(-1, NT), y.reshape(-1),
                                       weight=cw, label_smoothing=0.05, reduction='none')
                loss = loss.reshape(y.shape)
                wpos = torch.ones_like(loss)
                wpos[:, b['L'] - 1] = 1.8
                loss = (loss * wpos).sum() / wpos.sum()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                opt.step()
            sched.step()
            f1 = eval_f1(model, rows, es_idx)
            if f1 > best_f1:
                best_f1, bad = f1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 12:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    tr_rows, n_tr = build_rows(train_df, targets=True)
    te_rows, n_te = build_rows(test_df, targets=False)
    oof = np.zeros((n_tr, NT), np.float32)
    test_acc = np.zeros((n_te, NT), np.float32)
    n_models = 0
    for f in range(5):
        tr_idx = [i for i in range(len(tr_rows)) if folds_rows[i] != f]
        va_idx = [i for i in range(len(tr_rows)) if folds_rows[i] == f]
        alld = np.concatenate([tr_rows[i]['dense'] for i in tr_idx], 0)
        mu = alld.mean(0)
        sd = alld.std(0)
        sd[sd < 1e-6] = 1.0
        trs = [dict(r, dense=(r['dense'] - mu) / sd) for r in tr_rows]
        tes = [dict(r, dense=(r['dense'] - mu) / sd) for r in te_rows]
        c = np.zeros(NT)
        for i in tr_idx:
            for t in trs[i]['y']:
                c[t] += 1
        cw = torch.tensor(((c.sum() / np.maximum(c, 1)) ** 0.5), dtype=torch.float32)
        cw = cw / cw.mean()
        rng = np.random.RandomState(100 + f)
        tr_sh = tr_idx.copy()
        rng.shuffle(tr_sh)
        n_es = max(int(0.15 * len(tr_sh)), 40)
        es_idx = tr_sh[:n_es]
        fit_idx = tr_sh[n_es:]
        va_sum = {i: np.zeros((trs[i]['L'], NT)) for i in va_idx}
        te_sum = {i: np.zeros((tes[i]['L'], NT)) for i in range(len(tes))}
        n_here = 0
        for seq in ('gru', 'mlp'):
            for sd_i in range(seeds_per_arch):
                model = train_one(trs, fit_idx, es_idx, seq, seed=sd_i, cw=cw)
                vp = predict(model, trs, va_idx)
                for i in va_idx:
                    va_sum[i] += vp[i]
                tp = predict(model, tes, list(range(len(tes))))
                for i in range(len(tes)):
                    te_sum[i] += tp[i]
                n_models += 1
                n_here += 1
        for i in va_idx:
            p = va_sum[i] / n_here
            for k, g in enumerate(trs[i]['nidx']):
                oof[g] = p[k]
        for i in range(len(tes)):
            p = te_sum[i] / n_here
            for k, g in enumerate(tes[i]['nidx']):
                test_acc[g] += p[k]
        log(f'nnseq fold {f} done ({n_here} models)')
    return oof.astype(np.float64), (test_acc / 5.0).astype(np.float64), n_models


# ---------------------------------------------------------------------------
# meta stacker
# ---------------------------------------------------------------------------

def run_meta(P3_oof, P3_test, Bo, Bt, prev_tr, next_tr, prev_te, next_te,
             Xs_tr, Xs_te, tit_tr, for_tr, tit_te, for_te, y, node_fold, C,
             mask_tr=None, mask_te=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from scipy import sparse

    def neigh(B, pv, nx):
        n = B.shape[0]
        p = np.zeros((n, NT))
        q = np.zeros((n, NT))
        m = pv >= 0
        p[m] = B[pv[m]]
        m2 = nx >= 0
        q[m2] = B[nx[m2]]
        return p, q

    pvo, nxo = neigh(Bo, prev_tr, next_tr)
    pvt, nxt_ = neigh(Bt, prev_te, next_te)

    def dense(P3, B, pv, nx, Xs):
        return np.concatenate([np.log(np.clip(np.concatenate(P3, 1), 1e-9, 1)),
                               np.log(np.clip(B, 1e-9, 1)),
                               np.log(np.clip(pv, 1e-9, 1)),
                               np.log(np.clip(nx, 1e-9, 1)), Xs], 1)

    Dtr = dense(P3_oof, Bo, pvo, nxo, Xs_tr)
    Dte = dense(P3_test, Bt, pvt, nxt_, Xs_te)
    if mask_tr is not None:
        keep_tr = np.where(mask_tr)[0]
        keep_te = np.where(mask_te)[0]
        Dtr = Dtr[keep_tr]
        Dte = Dte[keep_te]
        tit_tr = [tit_tr[i] for i in keep_tr]
        for_tr = [for_tr[i] for i in keep_tr]
        tit_te = [tit_te[i] for i in keep_te]
        for_te = [for_te[i] for i in keep_te]
        y = y[keep_tr]
        node_fold = node_fold[keep_tr]

    def vecs():
        from sklearn.feature_extraction.text import TfidfVectorizer
        return [TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True),
                TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)]

    tit_tr = np.array(tit_tr, dtype=object)
    for_tr = np.array(for_tr, dtype=object)
    tit_te = np.array(tit_te, dtype=object)
    for_te = np.array(for_te, dtype=object)

    def tf(vs, ti, fo):
        return sparse.hstack([vs[0].transform(ti), vs[1].transform(ti),
                              vs[2].transform(fo)]).tocsr()

    oof = np.zeros((len(Dtr), NT))
    for f in range(5):
        trm = np.where(node_fold != f)[0]
        prm = np.where(node_fold == f)[0]
        sc = StandardScaler().fit(Dtr[trm])
        vs = vecs()
        vs[0].fit(tit_tr[trm]); vs[1].fit(tit_tr[trm]); vs[2].fit(for_tr[trm])
        Xtr = sparse.hstack([sparse.csr_matrix(sc.transform(Dtr[trm])),
                             tf(vs, tit_tr[trm], for_tr[trm])]).tocsr()
        Xpr = sparse.hstack([sparse.csr_matrix(sc.transform(Dtr[prm])),
                             tf(vs, tit_tr[prm], for_tr[prm])]).tocsr()
        m = LogisticRegression(C=C, class_weight='balanced', max_iter=4000, tol=1e-3)
        m.fit(Xtr, y[trm])
        oof[prm] = predict_full(m, Xpr)
    sc = StandardScaler().fit(Dtr)
    vs = vecs()
    vs[0].fit(tit_tr); vs[1].fit(tit_tr); vs[2].fit(for_tr)
    Xtr = sparse.hstack([sparse.csr_matrix(sc.transform(Dtr)), tf(vs, tit_tr, for_tr)]).tocsr()
    Xte = sparse.hstack([sparse.csr_matrix(sc.transform(Dte)), tf(vs, tit_te, for_te)]).tocsr()
    m = LogisticRegression(C=C, class_weight='balanced', max_iter=4000, tol=1e-3)
    m.fit(Xtr, y)
    test = predict_full(m, Xte)
    return oof, test


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    root = find_data_root()
    log(f'data root: {root}')
    train = pd.read_csv(os.path.join(root, 'train.csv'))
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    sample = pd.read_csv(os.path.join(root, 'sample_submission.csv'))
    out_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else \
        os.path.join('working', 'submission.csv')
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    log(f'submission target: {out_file}')

    ids = train['sample_id'].tolist()
    test_ids = test['sample_id'].tolist()
    lens = np.array([len(r.split()) for r in train['masked_nodes']])
    test_lens = np.array([len(r.split()) for r in test['masked_nodes']])
    targets = {r['sample_id']: parse_target(r['target_sequence'])[0] for _, r in train.iterrows()}
    pars_tr = {r['sample_id']: solve_parents(r) for _, r in train.iterrows()}
    pars_te = {r['sample_id']: solve_parents(r) for _, r in test.iterrows()}
    folds_rows = make_folds(train)

    # fallback submission writer (used on catastrophic failure and at the end)
    def write_submission(seqs_idx_by_row, path=None):
        path = path or out_file
        out_rows = {}
        for r, sid in enumerate(test_ids):
            toks = []
            for k, ci in enumerate(seqs_idx_by_row[r]):
                p = pars_te[sid][k]
                toks.append('type_' + TYPES[int(ci)])
                toks.append('parent_' + p)
            out_rows[sid] = ' '.join(toks)
        sub = pd.DataFrame({'sample_id': sample['sample_id'],
                            'target_sequence': [out_rows[s] for s in sample['sample_id']]})
        sub.to_csv(path, index=False)
        return sub

    # transitions fallback (position-prior decode; still fit from train)
    full_tables = fit_transitions(targets, ids)

    def fallback():
        E = np.zeros((int(test_lens.sum()), NT))
        seqs = decode_matrix(np.full((int(test_lens.sum()), NT), 0.2), test_lens,
                             full_tables, 'viterbi')
        write_submission(seqs)
        log('FALLBACK submission written (transition-prior decode)')

    try:
        run_pipeline(train, test, ids, test_ids, lens, test_lens, targets,
                     pars_tr, pars_te, folds_rows, full_tables, write_submission)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f'pipeline failed ({e}); writing fallback')
        fallback()


def run_pipeline(train, test, ids, test_ids, lens, test_lens, targets,
                 pars_tr, pars_te, folds_rows, full_tables, write_submission):
    recs_tr = extract_nodes(train)
    recs_te = extract_nodes(test)
    n_tr, n_te = len(recs_tr), len(recs_te)
    sid_to_fold = dict(zip(ids, folds_rows))
    node_fold = np.array([sid_to_fold[r['sample_id']] for r in recs_tr])
    y = np.array([TYPE_IDX[targets[r['sample_id']][r['pos']]] for r in recs_tr])
    log(f'nodes: train {n_tr}, test {n_te}')

    offs = np.concatenate([[0], np.cumsum(lens)])
    t_offs = np.concatenate([[0], np.cumsum(test_lens)])
    prev_tr = np.array([offs[i] + k - 1 if k > 0 else -1
                        for i in range(len(ids)) for k in range(lens[i])])
    next_tr = np.array([offs[i] + k + 1 if k < lens[i] - 1 else -1
                        for i in range(len(ids)) for k in range(lens[i])])
    prev_te = np.array([t_offs[i] + k - 1 if k > 0 else -1
                        for i in range(len(test_ids)) for k in range(test_lens[i])])
    next_te = np.array([t_offs[i] + k + 1 if k < test_lens[i] - 1 else -1
                        for i in range(len(test_ids)) for k in range(test_lens[i])])

    # per-fold transition tables + scorer for OOF selection
    tables_by_fold = {f: fit_transitions(targets, [s for s in ids if sid_to_fold[s] != f])
                      for f in range(5)}
    true_types = [[TYPE_IDX[t] for t in targets[s]] for s in ids]
    true_pars = [parse_target(train.iloc[i]['target_sequence'])[1] for i in range(len(ids))]
    fs = FastScorer(true_types, true_pars, [pars_tr[s] for s in ids])

    def decode_oof(P, mode):
        out = [None] * len(ids)
        E_all = np.log(np.clip(P, 1e-9, 1.0))
        for f in range(5):
            T, I = tables_by_fold[f]
            for L in (3, 4):
                idx = [r for r in range(len(ids))
                       if lens[r] == L and sid_to_fold[ids[r]] == f]
                if not idx:
                    continue
                E = np.stack([E_all[offs[r]:offs[r] + L] for r in idx])
                dec = batch_viterbi(E, T[L], I[L]) if mode == 'viterbi' else \
                    batch_posterior(E, T[L], I[L])
                for k, r in enumerate(idx):
                    out[r] = dec[k]
        return out

    def oof_score(P, mode):
        return fs.score(decode_oof(P, mode))[0]

    # ---------------- family A: GBM ----------------
    Xg_tr = gbm_features(recs_tr)
    Xg_te = gbm_features(recs_te)
    gbm_out = run_gbm(Xg_tr, Xg_te, y, node_fold)
    g_best, g_score = None, -1
    for tag, (o, t) in gbm_out.items():
        s = max(oof_score(o, 'viterbi'), oof_score(o, 'posterior'))
        log(f'gbm[{tag}] OOF {s:.4f}')
        if s > g_score:
            g_best, g_score = tag, s
    g_oof, g_test = gbm_out[g_best]
    log(f'gbm winner {g_best} {g_score:.4f}')

    # ---------------- family B: textlin ----------------
    Dl_tr = textlin_dense(recs_tr)
    Dl_te = textlin_dense(recs_te)
    tit_tr = [str(r['title']) for r in recs_tr]
    for_tr = [str(r['forum']) for r in recs_tr]
    tit_te = [str(r['title']) for r in recs_te]
    for_te = [str(r['forum']) for r in recs_te]
    t_oof, t_test = run_textlin(Dl_tr, Dl_te, tit_tr, for_tr, tit_te, for_te, y, node_fold)
    log(f'textlin OOF {max(oof_score(t_oof, "viterbi"), oof_score(t_oof, "posterior")):.4f}')

    # ---------------- family C: nnseq (adaptive seeds) ----------------
    nn_budget = min(0.55 * max(time_left() - 600, 0), 1500)
    seeds_per_arch = 3 if nn_budget > 700 else 2
    log(f'nnseq budget {nn_budget:.0f}s -> {seeds_per_arch} seeds/arch')
    try:
        n_oof, n_test, n_models = run_nnseq(train, test, folds_rows, seeds_per_arch)
        log(f'nnseq ({n_models} models) OOF '
            f'{max(oof_score(n_oof, "viterbi"), oof_score(n_oof, "posterior")):.4f}')
    except Exception as e:
        log(f'nnseq family failed ({e}); substituting uninformative probs '
            f'(blend search will zero it out)')
        n_oof = np.full((n_tr, NT), 1.0 / NT)
        n_test = np.full((n_te, NT), 1.0 / NT)

    # ---------------- blend weight search ----------------
    logs = [np.log(np.clip(P, 1e-9, 1)) for P in (g_oof, t_oof, n_oof)]
    logs_t = [np.log(np.clip(P, 1e-9, 1)) for P in (g_test, t_test, n_test)]
    best = None
    for wg in range(9):
        for wt in range(9 - wg):
            wn = 8 - wg - wt
            ws = (wg / 8, wt / 8, wn / 8)
            Pb = np.exp(sum(w * l for w, l in zip(ws, logs)))
            Pb /= Pb.sum(1, keepdims=True)
            for mode in ('viterbi', 'posterior'):
                s = oof_score(Pb, mode)
                if best is None or s > best[0]:
                    best = (s, ws, mode)
    s_blend, ws, blend_mode = best
    log(f'blend {ws} {blend_mode} OOF {s_blend:.4f}')
    Bo = np.exp(sum(w * l for w, l in zip(ws, logs)))
    Bo /= Bo.sum(1, keepdims=True)
    Bt = np.exp(sum(w * l for w, l in zip(ws, logs_t)))
    Bt /= Bt.sum(1, keepdims=True)

    # ---------------- meta stacker (struct features = textlin dense basis) ----------------
    Cs = [2.5, 1.5] if time_left() > 900 else [2.5]
    m_best = None
    for C in Cs:
        m_oof, m_test = run_meta((g_oof, t_oof, n_oof), (g_test, t_test, n_test),
                                 Bo, Bt, prev_tr, next_tr, prev_te, next_te,
                                 Dl_tr, Dl_te, tit_tr, for_tr, tit_te, for_te,
                                 y, node_fold, C)
        for mode in ('viterbi', 'posterior'):
            s = oof_score(m_oof, mode)
            log(f'meta C={C} {mode} OOF {s:.4f}')
            if m_best is None or s > m_best[0]:
                m_best = (s, C, mode, m_oof, m_test)
    s_meta, C_meta, meta_mode, m_oof, m_test = m_best
    log(f'meta winner C={C_meta} {meta_mode} OOF {s_meta:.4f}')

    # ---------------- choose base final emissions ----------------
    def norm_geo(A, B, wb):
        M = np.exp((1 - wb) * np.log(np.clip(A, 1e-9, 1)) +
                   wb * np.log(np.clip(B, 1e-9, 1)))
        return M / M.sum(1, keepdims=True)

    cands = [('meta', m_oof, m_test, meta_mode, s_meta),
             ('blend3', Bo, Bt, blend_mode, s_blend)]
    for wn in (0.125, 0.25):
        Po = norm_geo(m_oof, n_oof, wn)
        Pt = norm_geo(m_test, n_test, wn)
        sm, bm = -1, 'posterior'
        for mode in ('viterbi', 'posterior'):
            s = oof_score(Po, mode)
            if s > sm:
                sm, bm = s, mode
        log(f'cand meta+nn{wn} {bm} OOF {sm:.4f}')
        cands.append((f'meta+nn{wn}', Po, Pt, bm, sm))
    tag, P_fin_o, P_fin_t, fin_mode, s_fin = max(cands, key=lambda c: c[4])
    log(f'final emissions: {tag} {fin_mode} OOF {s_fin:.4f}')

    # ---------------- anchor specialist (last-position model, geo-mixed) ----
    anchor_idx_tr = offs[1:] - 1
    anchor_idx_te = t_offs[1:] - 1
    mask_tr = np.zeros(n_tr, bool)
    mask_tr[anchor_idx_tr] = True
    mask_te = np.zeros(n_te, bool)
    mask_te[anchor_idx_te] = True
    a_oof, a_test = run_meta((g_oof, t_oof, n_oof), (g_test, t_test, n_test),
                             Bo, Bt, prev_tr, next_tr, prev_te, next_te,
                             Dl_tr, Dl_te, tit_tr, for_tr, tit_te, for_te,
                             y, node_fold, C_meta,
                             mask_tr=mask_tr, mask_te=mask_te)

    def anchor_mixed(P, A_probs, aidx, w):
        P2 = np.clip(P, 1e-9, 1).copy()
        P2[aidx] = np.exp((1 - w) * np.log(P2[aidx]) +
                          w * np.log(np.clip(A_probs, 1e-9, 1)))
        return P2 / P2.sum(1, keepdims=True)

    w_anchor, s_anchor = 0.0, s_fin
    for w in (0.25, 0.5):
        s = oof_score(anchor_mixed(P_fin_o, a_oof, anchor_idx_tr, w), fin_mode)
        log(f'anchor mix w={w} OOF {s:.4f}')
        if s > s_anchor:
            w_anchor, s_anchor = w, s
    if w_anchor > 0:
        P_fin_o = anchor_mixed(P_fin_o, a_oof, anchor_idx_tr, w_anchor)
        P_fin_t = anchor_mixed(P_fin_t, a_test, anchor_idx_te, w_anchor)
        s_fin = s_anchor
        log(f'anchor specialist kept: w={w_anchor} OOF {s_fin:.4f}')
    else:
        log('anchor specialist rejected on OOF')

    s_final, comps = fs.score(decode_oof(P_fin_o, fin_mode))
    log(f'FINAL OOF {s_final:.4f} components ' +
        ' '.join(f'{k}={v:.4f}' for k, v in comps.items()))

    seqs_te = decode_matrix(P_fin_t, test_lens, full_tables, fin_mode)
    sub = write_submission(seqs_te)
    log(f'submission written: {len(sub)} rows, OOF estimate {s_fin:.4f}')

    # strict format self-check
    tok_re = re.compile(r'[a-z0-9_]+')
    par_re = re.compile(r'^parent_(root|n\d\d)$')
    for s in sub['target_sequence']:
        toks = tok_re.findall(s.lower())
        assert len(toks) % 2 == 0 and len(toks) > 0
        for i, t in enumerate(toks):
            if i % 2 == 0:
                assert t in {'type_' + c for c in TYPES}, t
            else:
                assert par_re.match(t), t
    assert list(sub.columns) == ['sample_id', 'target_sequence']
    assert len(sub) == len(test)
    log('format self-check passed')


if __name__ == '__main__':
    main()
