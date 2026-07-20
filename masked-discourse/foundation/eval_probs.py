"""Uniform evaluator: takes an OOF prob matrix (N_nodes x 5, canonical order),
runs per-fold transition Viterbi decode, scores with the OFFICIAL metric.

Canonical node order = train.csv row order, masked_nodes order within row.
Usage: python eval_probs.py <oof_probs.npy> [--mode viterbi|posterior] [--json out.json]
"""
import sys, os, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_target, make_folds, format_submission, solve_parents
from scorer import score_submission
from decode import fit_transitions, decode_row


def load_ctx():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'public')
    train = pd.read_csv(os.path.join(root, 'train.csv'))
    ids = train['sample_id'].tolist()
    lens = {r['sample_id']: len(r['masked_nodes'].split()) for _, r in train.iterrows()}
    targets = {r['sample_id']: parse_target(r['target_sequence'])[0] for _, r in train.iterrows()}
    folds = make_folds(train)
    fold_of = dict(zip(ids, folds))
    pars = {r['sample_id']: solve_parents(r) for _, r in train.iterrows()}
    truth = train[['sample_id', 'target_sequence']]
    return train, ids, lens, targets, fold_of, pars, truth


def probs_by_row_from_matrix(P, ids, lens):
    out = {}
    o = 0
    for sid in ids:
        L = lens[sid]
        out[sid] = P[o:o + L]
        o += L
    assert o == P.shape[0], f'node count mismatch: {o} vs {P.shape[0]}'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('probs')
    ap.add_argument('--mode', default='viterbi')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    train, ids, lens, targets, fold_of, pars, truth = load_ctx()
    P = np.load(a.probs)
    pbr = probs_by_row_from_matrix(P, ids, lens)

    tables_by_fold = {}
    for f in range(5):
        tr_ids = [s for s in ids if fold_of[s] != f]
        tables_by_fold[f] = fit_transitions(targets, tr_ids)

    seqs = {}
    for sid in ids:
        seqs[sid] = decode_row(pbr[sid], tables_by_fold[fold_of[sid]], mode=a.mode)
    sub = format_submission(ids, [seqs[s] for s in ids], [pars[s] for s in ids])
    s, comps = score_submission(sub, truth, verbose=True)
    if a.json:
        comps2 = {k: v for k, v in comps.items() if not isinstance(v, dict)}
        comps2['score'] = s
        comps2['mode'] = a.mode
        with open(a.json, 'w') as f:
            json.dump(comps2, f, indent=2)
    return s


if __name__ == '__main__':
    main()
