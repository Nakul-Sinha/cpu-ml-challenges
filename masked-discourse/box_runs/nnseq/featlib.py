"""Feature extraction for the nnseq family. Builds per-row structured records
with dense node features + hashed char-ngram bags for title/forum. Canonical
node order preserved (df row order, masked_nodes order)."""
import os, sys, collections, hashlib
import numpy as np
sys.path.insert(0, os.path.expanduser('~/discourse/foundation'))
from common import TYPES, TYPE_IDX, extract_row_nodes, parse_target

VIS6 = ['answer', 'elaboration', 'question', 'appreciation', 'agreement', 'other']
PARTYPE = ['answer', 'elaboration', 'question', 'appreciation', 'agreement', 'other', 'MASK', 'ROOT']
PARTYPE_IX = {t: i for i, t in enumerate(PARTYPE)}
PARKIND = {'root': 0, 'masked': 1, 'post': 2, 'visible': 3}

def _cv(d):
    return [float(d.get(t, 0)) for t in VIS6]

def _hash_ngrams(s, n_lo, n_hi, nbuckets, prefix=''):
    s = (prefix + ' ' + str(s).lower()).strip()
    s = '^' + s + '$'
    out = []
    for n in range(n_lo, n_hi + 1):
        for i in range(len(s) - n + 1):
            g = s[i:i+n]
            h = int(hashlib.md5(g.encode()).hexdigest(), 16) % nbuckets
            out.append(h)
    if not out:
        out = [0]
    return out

DENSE_NAMES = None

def node_dense(rec):
    kt = rec['kid_types']; dt = rec['desc_types']; st = rec['sib_types']
    vc = rec['vis_counts']
    bt = collections.Counter(rec['between'])
    nk = max(rec['n_kids_vis'], 1); nd = max(rec['n_desc_vis'], 1)
    L = rec['route_len']; pos = rec['pos']
    feats = [
        pos, L, rec['depth'], rec['par_depth'], rec['view_idx'], rec['view_frac'],
        rec['max_depth'], rec['n_kids_vis'], rec['n_desc_vis'], rec['n_masked_kids'],
        rec['n_sibs'], rec['n_nodes'], rec['n_out'], rec['gap_prev'], rec['gap_next'],
        np.log1p(rec['title_len']), np.log1p(rec['title_words']),
        float(rec['title_q']), float(rec['title_excl']), float(rec['title_wh']),
        float(rec['prof_wide']), float(rec['prof_long']), float(rec['prof_self']),
        float(rec['has_post']), float(pos == 0), float(pos == L - 1),
        float(0 < pos < L - 1),
        rec['depth'] / max(rec['max_depth'], 1),
    ]
    feats += _cv(kt)                                   # kid counts (6)
    feats += [kt.get(t, 0) / nk for t in VIS6]         # kid frac (6)
    feats += _cv(dt)                                   # desc counts (6)
    feats += [dt.get(t, 0) / nd for t in VIS6]         # desc frac (6)
    feats += _cv(st)                                   # sib counts (6)
    feats += _cv(vc)                                   # neighborhood vis counts (6)
    feats += [float(bt.get(t, 0)) for t in VIS6]       # between-chain types (6)
    # explicit strong-signal booleans
    feats += [
        float(kt.get('answer', 0) > 0), float(kt.get('question', 0) > 0),
        float(kt.get('elaboration', 0) > 0), float(kt.get('appreciation', 0) > 0),
        float(kt.get('agreement', 0) > 0), float(kt.get('other', 0) > 0),
        float(dt.get('answer', 0) > 0), float(dt.get('question', 0) > 0),
        float(rec['n_kids_vis'] == 0),
    ]
    return feats

def build_rows(df, tcfg, targets=True):
    """Return list of row dicts + total node count. tcfg: dict with hashing cfg."""
    rows = []
    g = 0
    for _, r in df.iterrows():
        recs = extract_row_nodes(r)
        L = len(recs)
        dense = [node_dense(rc) for rc in recs]
        pt = [PARTYPE_IX.get(rc['par_type'], PARTYPE_IX['other']) for rc in recs]
        pk = [PARKIND.get(rc['par_kind'], 3) for rc in recs]
        depthb = [min(max(rc['depth'], 0), 7) for rc in recs]
        poslen = [rc['pos'] + (0 if rc['route_len'] == 3 else 4) for rc in recs]
        role = [0 if rc['pos'] == 0 else (2 if rc['pos'] == rc['route_len'] - 1 else 1) for rc in recs]
        title_ng = _hash_ngrams(recs[0]['title'], tcfg['t_lo'], tcfg['t_hi'], tcfg['t_buck'])
        forum_ng = _hash_ngrams(recs[0]['forum'], tcfg['f_lo'], tcfg['f_hi'], tcfg['f_buck'], prefix='F')
        y = None
        if targets:
            ts = parse_target(r['target_sequence'])[0]
            y = [TYPE_IDX[t] for t in ts]
        rows.append(dict(nidx=list(range(g, g + L)), L=L, dense=np.array(dense, np.float32),
                         par_type=np.array(pt), par_kind=np.array(pk), depthb=np.array(depthb),
                         poslen=np.array(poslen), role=np.array(role),
                         title_ng=title_ng, forum_ng=forum_ng,
                         y=(np.array(y) if y is not None else None)))
        g += L
    return rows, g
