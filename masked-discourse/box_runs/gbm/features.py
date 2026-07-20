"""Dense feature basis for the GBM family. Deterministic (FNV-1a hashing) so
train/test features are identical across processes.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'foundation'))
from common import TYPES, VIS_TYPES  # VIS_TYPES = TYPES + ['other']

PAR_TYPES = ['ROOT', 'MASK'] + VIS_TYPES           # 8
PAR_KINDS = ['root', 'masked', 'post', 'visible']  # 4


def _fnv(s):
    h = 2166136261
    for c in s:
        h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
    return h


def _hash_char_ngrams(s, n, nbuck, out, base):
    s = '^' + str(s).lower() + '$'
    for i in range(len(s) - n + 1):
        out[base + (_fnv(s[i:i + n]) % nbuck)] += 1.0


def build_features(recs, title_buck=48, forum_buck=32, ngram=3, hashing=True):
    """recs: list of node dicts from common.extract_nodes. Returns (X float32, names)."""
    names = []
    # --- fixed schema of scalar/onehot/count columns ---
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
    # interactions
    inter = ['q_first', 'parq_first', 'para_first', 'kidans_first', 'kidans_interior',
             'kidans_last', 'kidans_posfrac', 'kidq_any', 'kidapp_any',
             'anyq_x_first', 'depth1_first']
    names += inter
    names += ['partf_' + t for t in PAR_TYPES]   # par_type one-hot gated by is_first
    n_struct = len(names)
    if hashing:
        names += ['tth_%d' % i for i in range(title_buck)]
        names += ['fmh_%d' % i for i in range(forum_buck)]
    F = len(names)

    X = np.zeros((len(recs), F), dtype=np.float32)
    for r, rec in enumerate(recs):
        L = rec['route_len']; pos = rec['pos']
        is_first = 1.0 if pos == 0 else 0.0
        is_last = 1.0 if pos == L - 1 else 0.0
        is_interior = 1.0 if (0 < pos < L - 1) else 0.0
        kt = rec['kid_types']; dt = rec['desc_types']; st = rec['sib_types']
        vc = rec['vis_counts']
        btw = {}
        for t in rec['between']:
            btw[t] = btw.get(t, 0) + 1
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
        col = {n: i for i, n in enumerate(names[:n_struct])}
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
        # interactions
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
        if hashing:
            _hash_char_ngrams(rec['title'], ngram, title_buck, X[r], n_struct)
            _hash_char_ngrams(rec['forum'], ngram, forum_buck, X[r], n_struct + title_buck)
    return X, names
