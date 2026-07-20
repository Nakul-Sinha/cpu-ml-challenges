"""Canonical parsing, parent recovery, split, and feature basis for the
masked-discourse challenge. ALL experiments must import from here so runs are
comparable.
"""
import re
import collections
import numpy as np
import pandas as pd

TYPES = ['answer', 'elaboration', 'question', 'appreciation', 'agreement']
TYPE_IDX = {t: i for i, t in enumerate(TYPES)}
VIS_TYPES = TYPES + ['other']  # visible cards can also be 'other'

CARD = re.compile(r'^(N\d+)\[d=(-?\d+),p=([A-Za-z0-9_]+),t=([A-Za-z_]+)\]$')


def parse_view(view):
    """Return ordered dict nid -> {d, p, t, idx} preserving view order."""
    nodes = collections.OrderedDict()
    for i, card in enumerate(view.split('||')):
        m = CARD.match(card.strip())
        if not m:
            raise ValueError(f'bad card: {card!r}')
        nodes[m.group(1)] = dict(d=int(m.group(2)), p=m.group(3), t=m.group(4), idx=i)
    return nodes


def parse_target(ts):
    toks = str(ts).split()
    types = [t[len('type_'):] for t in toks if t.startswith('type_')]
    pars = [t[len('parent_'):] for t in toks if t.startswith('parent_')]
    return types, pars


def dfs_parent(nodes, nid):
    """Recover the masked parent token for nid.

    The view is a DFS pre-order listing of the neighborhood, so a node's parent
    is the nearest preceding node whose depth is exactly d-1. Depth-1 nodes
    attach to the submission node (d in {-1,0}) when it is in view, else ROOT.
    Verified 99.90% on train.
    Returns lowercase token WITHOUT the 'parent_' prefix: 'n03' or 'root'.
    """
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
    """children[nid] = list of nids whose visible p == nid (view order)."""
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


WH_WORDS = ('what', 'why', 'how', 'who', 'when', 'where', 'which', 'is ', 'are ',
            'do ', 'does ', 'can ', 'should ', 'anyone', 'any ')


def extract_row_nodes(row):
    """Per-masked-node structured record list for one dataframe row.

    Provides the shared feature basis: topology, neighbor types, view stats,
    title/forum/profile context, and route-chain links (gap to prev/next masked
    node plus visible types strictly between them on the recovered chain).
    """
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
        # siblings under same predicted parent (visible ones)
        sibs = [k for k, v in nodes.items()
                if v['p'] == par_tok.upper() and k != nid] if par_tok != 'root' else \
               [k for k, v in nodes.items() if v['p'] == 'ROOT' and k != nid]
        sib_types = collections.Counter(nodes[k]['t'] for k in sibs)

        gap_prev = d - nodes[masked[i - 1]]['d'] if i > 0 else 0
        gap_next = nodes[masked[i + 1]]['d'] - d if i < L - 1 else 0
        # visible chain types strictly between prev masked node and this one
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
    all_recs = []
    for _, row in df.iterrows():
        all_recs.extend(extract_row_nodes(row))
    return all_recs


def make_folds(train, n_folds=5, seed=42):
    """Deterministic stratified folds by (anchor_type, route_len).
    Returns np.array of fold ids aligned with train rows."""
    anchors = []
    for _, r in train.iterrows():
        types, _ = parse_target(r['target_sequence'])
        anchors.append(types[-1] + '_' + str(len(types)))
    anchors = np.array(anchors)
    rng = np.random.RandomState(seed)
    folds = np.zeros(len(train), dtype=int)
    for s in np.unique(anchors):
        idx = np.where(anchors == s)[0]
        rng.shuffle(idx)
        for k, j in enumerate(idx):
            folds[j] = k % n_folds
    return folds


def format_submission(ids, type_seqs, par_seqs):
    """type_seqs/par_seqs: list (per row) of lists of bare tokens."""
    rows = []
    for sid, ts, ps in zip(ids, type_seqs, par_seqs):
        assert len(ts) == len(ps)
        toks = []
        for t, p in zip(ts, ps):
            assert t in TYPES, t
            assert p == 'root' or re.fullmatch(r'n\d\d', p), p
            toks.append('type_' + t)
            toks.append('parent_' + p)
        rows.append(' '.join(toks))
    return pd.DataFrame({'sample_id': ids, 'target_sequence': rows})
