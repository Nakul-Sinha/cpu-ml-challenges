"""Vectorized in-memory scorer equivalent to scorer.py for the always-valid
submissions we generate (exact-length alternating sequences). Used inside tight
tuning loops. Validated against scorer.score_submission.
"""
import numpy as np

from common import TYPES

NT = 5


class FastScorer:
    def __init__(self, true_types_by_row, true_pars_by_row, pred_pars_by_row):
        """true_types_by_row: list of lists of type indices (per row).
        true_pars_by_row / pred_pars_by_row: list of lists of parent token strings.
        Parents are FIXED across evaluations (only types change)."""
        self.n = len(true_types_by_row)
        self.Ls = np.array([len(t) for t in true_types_by_row])
        self.true_types = true_types_by_row
        # parent correctness per position (fixed)
        self.par_eq = [np.array([a == b for a, b in zip(p, t)])
                       for p, t in zip(pred_pars_by_row, true_pars_by_row)]
        self.par_score = float(np.mean([pe.mean() for pe in self.par_eq]))
        # parent token ids for LCS: encode parents as 5 + hash bucket (distinct space from types)
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
                s.append(ty); s.append(pid(pa))
            self.true_seq.append(np.array(s))
            self.pred_par_ids.append(np.array([pid(x) for x in pp]))
        # group rows by L for batch ops
        self.groups = {L: np.where(self.Ls == L)[0] for L in np.unique(self.Ls)}
        self.tt_arr = {L: np.stack([np.array(self.true_types[i]) for i in idx])
                       for L, idx in self.groups.items()}
        self.ts_arr = {L: np.stack([self.true_seq[i] for i in idx])
                       for L, idx in self.groups.items()}
        self.pp_arr = {L: np.stack([self.pred_par_ids[i] for i in idx])
                       for L, idx in self.groups.items()}

    def score(self, pred_by_row):
        """pred_by_row: (n,) object list of int arrays (type indices). Returns
        (score, comps). Assumes exact lengths (guaranteed by our decoder)."""
        tp = np.zeros(NT); fp = np.zeros(NT); fn = np.zeros(NT)
        atp = np.zeros(NT); afp = np.zeros(NT); afn = np.zeros(NT)
        type_pos_sum = 0.0
        ordered_sum = 0.0
        for L, idx in self.groups.items():
            P = np.stack([pred_by_row[i] for i in idx])          # (R, L)
            T = self.tt_arr[L]
            eq = P == T
            type_pos_sum += (eq.mean(1)).sum()
            # pooled confusion
            for c in range(NT):
                tp[c] += ((P == c) & (T == c)).sum()
                fp[c] += ((P == c) & (T != c)).sum()
                fn[c] += ((P != c) & (T == c)).sum()
                atp[c] += ((P[:, -1] == c) & (T[:, -1] == c)).sum()
                afp[c] += ((P[:, -1] == c) & (T[:, -1] != c)).sum()
                afn[c] += ((P[:, -1] != c) & (T[:, -1] == c)).sum()
            # batch LCS between pred interleaved and true interleaved
            R = len(idx)
            A = np.empty((R, 2 * L), dtype=np.int64)
            A[:, 0::2] = P
            A[:, 1::2] = self.pp_arr[L]
            B = self.ts_arr[L]
            m = 2 * L
            dp = np.zeros((R, m + 1, m + 1), dtype=np.int16)
            for i in range(1, m + 1):
                Ai = A[:, i - 1][:, None]
                match = (Ai == B).astype(np.int16)               # (R, m)
                dp[:, i, 1:] = np.maximum(dp[:, i - 1, 1:], dp[:, i - 1, :-1] + match)
                np.maximum.accumulate(dp[:, i, :], axis=1, out=dp[:, i, :])
            l = dp[:, m, m].astype(float)
            f1 = 2 * (l / m) * (l / m) / np.clip(l / m + l / m, 1e-12, None)
            f1[l == 0] = 0.0
            ordered_sum += f1.sum()

        def macro(tp_, fp_, fn_):
            den = 2 * tp_ + fp_ + fn_
            f = np.where(den > 0, 2 * tp_ / np.clip(den, 1e-12, None), 0.0)
            return f.mean(), f

        tm, tm_cls = macro(tp, fp, fn)
        am, am_cls = macro(atp, afp, afn)
        comps = dict(TypeMacroF1=float(tm), AnchorMacroF1=float(am),
                     TypeScore=float(type_pos_sum / self.n),
                     OrderedScore=float(ordered_sum / self.n),
                     ParentScore=self.par_score,
                     type_percls={TYPES[i]: float(tm_cls[i]) for i in range(NT)},
                     anchor_percls={TYPES[i]: float(am_cls[i]) for i in range(NT)})
        score = 100 * float(np.clip(0.45 * tm + 0.25 * am + 0.10 * comps['TypeScore']
                                    + 0.15 * comps['OrderedScore'] + 0.05 * self.par_score, 0, 1))
        return score, comps
