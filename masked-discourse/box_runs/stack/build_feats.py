import sys, os, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/discourse/foundation'))
from common import (TYPES, TYPE_IDX, VIS_TYPES, parse_target, make_folds,
                    extract_nodes, extract_row_nodes)

ROOT = os.path.expanduser('~/discourse/dataset/public')
BASE = os.path.expanduser('~/discourse')
train = pd.read_csv(os.path.join(ROOT, 'train.csv'))
test = pd.read_csv(os.path.join(ROOT, 'test.csv'))

def build(df, has_label):
    recs = extract_nodes(df)
    rows = []
    labels = []
    row_id = []   # index of the source csv row per node
    pos_arr = []; L_arr = []
    # map sample_id -> row index
    sid_to_ri = {sid: i for i, sid in enumerate(df['sample_id'].tolist())}
    # labels aligned
    tgt = None
    if has_label:
        tgt = {r['sample_id']: parse_target(r['target_sequence'])[0] for _, r in df.iterrows()}
    for rec in recs:
        sid = rec['sample_id']; i = rec['pos']; L = rec['route_len']
        row_id.append(sid_to_ri[sid]); pos_arr.append(i); L_arr.append(L)
        if has_label:
            labels.append(TYPE_IDX[tgt[sid][i]])
        f = []
        # pos block
        f += [i, L, float(i==0), float(i==L-1), float(0<i<L-1)]
        # depth block
        f += [rec['depth'], rec['par_depth'], rec['gap_prev'], rec['gap_next']]
        # count scalars
        f += [rec['n_kids_vis'], rec['n_desc_vis'], rec['n_masked_kids'], rec['n_sibs']]
        # view block
        f += [rec['view_idx'], rec['view_frac'], rec['max_depth'], rec['n_out'],
              rec['n_nodes'], float(rec['has_post'])]
        # par_kind onehot
        for k in ('root','post','visible','masked'):
            f.append(float(rec['par_kind']==k))
        # par_type onehot
        for k in ('answer','elaboration','question','appreciation','agreement','other','MASK','ROOT'):
            f.append(float(rec['par_type']==k))
        # kid/desc/sib/vis counts over VIS_TYPES
        for dct in (rec['kid_types'], rec['desc_types'], rec['sib_types'], rec['vis_counts']):
            for k in VIS_TYPES:
                f.append(dct.get(k, 0))
        # has_answer_kid
        f.append(float(rec['kid_types'].get('answer',0) > 0))
        # between summary
        bt = collections.Counter(rec['between'])
        f.append(len(rec['between']))
        for k in VIS_TYPES:
            f.append(bt.get(k,0))
        # title / prof
        f += [rec['title_len'], rec['title_words'], float(rec['title_q']),
              float(rec['title_excl']), float(rec['title_wh'])]
        f += [float(rec['prof_wide']), float(rec['prof_long']), float(rec['prof_self'])]
        rows.append(f)
    X = np.array(rows, dtype=np.float64)
    return X, np.array(labels, dtype=np.int64) if has_label else None, \
           np.array(row_id), np.array(pos_arr), np.array(L_arr)

Xtr, ytr, row_tr, pos_tr, L_tr = build(train, True)
Xte, _, row_te, pos_te, L_te = build(test, False)

# folds per node
folds_row = make_folds(train)
folds_tr = folds_row[row_tr]

# neighbor indices within same row (prev/next node global index, -1 at edge)
def neighbor_idx(row_id, pos, L):
    n = len(row_id)
    prev = np.full(n, -1, int); nxt = np.full(n, -1, int)
    for gi in range(n):
        if pos[gi] > 0: prev[gi] = gi-1
        if pos[gi] < L[gi]-1: nxt[gi] = gi+1
    return prev, nxt
prev_tr, next_tr = neighbor_idx(row_tr, pos_tr, L_tr)
prev_te, next_te = neighbor_idx(row_te, pos_te, L_te)

print('Xtr', Xtr.shape, 'Xte', Xte.shape, 'ytr', ytr.shape)
print('nodes train', len(row_tr), 'test', len(row_te))
print('label dist', np.bincount(ytr, minlength=5).tolist())
print('fold sizes', np.bincount(folds_tr).tolist())
anchor_tr = (pos_tr == L_tr-1)
print('anchors train', anchor_tr.sum(), 'test', (pos_te==L_te-1).sum())
print('anchor label dist', np.bincount(ytr[anchor_tr], minlength=5).tolist())
print('L dist train', np.bincount(L_tr).tolist(), 'rows', len(train))

np.savez(os.path.join(BASE,'runs/stack/feats.npz'),
         Xtr=Xtr, Xte=Xte, ytr=ytr,
         row_tr=row_tr, pos_tr=pos_tr, L_tr=L_tr,
         row_te=row_te, pos_te=pos_te, L_te=L_te,
         folds_tr=folds_tr, folds_row=folds_row,
         prev_tr=prev_tr, next_tr=next_tr, prev_te=prev_te, next_te=next_te)
print('saved feats.npz; n_struct_feats =', Xtr.shape[1])
