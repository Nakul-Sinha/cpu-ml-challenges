"""Canonical ELRU scorer for Institutional Edit Ledger Recovery.

Implements the spec exactly:
  pair_quality   = 0.45*source_span_f1 + 0.45*replacement_chrf + 0.10*operation_type_match
  matched_utility= sum(matched pair quality) / max(n_true, n_pred)
  budget_fidelity= min(n_true, n_pred) / max(n_true, n_pred)
  row_score      = 0.85*matched_utility + 0.15*budget_fidelity
  language_score = 0.75*mean(edited rows) + 0.25*mean(unchanged rows)
  ELRU           = mean over languages [de, en, it]
Both ledgers empty -> row 1. Exactly one empty -> row 0.
replacement_chrf = mean over n=1..3 of char n-gram F1 (both empty -> 1, one empty -> 0).
Greedy matching by descending pair quality.
All agents must import THIS module so scores are comparable.
"""
import json
from collections import Counter


def _ngram_f1(a, b, n):
    if len(a) < n and len(b) < n:
        return None  # no n-grams on either side; skip this n
    ca = Counter(a[i:i + n] for i in range(len(a) - n + 1))
    cb = Counter(b[i:i + n] for i in range(len(b) - n + 1))
    ta, tb = sum(ca.values()), sum(cb.values())
    if ta == 0 or tb == 0:
        return 0.0
    ov = sum(min(v, cb[k]) for k, v in ca.items())
    p, r = ov / ta, ov / tb
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def replacement_chrf(a, b, nmax=3):
    if a == "" and b == "":
        return 1.0
    if a == "" or b == "":
        return 0.0
    vals = [v for v in (_ngram_f1(a, b, n) for n in range(1, nmax + 1)) if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def span_f1(s1, e1, s2, e2):
    ov = max(0, min(e1, e2) - max(s1, s2))
    l1, l2 = e1 - s1, e2 - s2
    if ov == 0 or l1 == 0 or l2 == 0:
        return 0.0
    p, r = ov / l1, ov / l2
    return 2 * p * r / (p + r)


def pair_quality(pe, te):
    sf = span_f1(pe["start"], pe["end"], te["start"], te["end"])
    rc = replacement_chrf(pe["replacement"], te["replacement"])
    tm = 1.0 if (pe["replacement"] == "") == (te["replacement"] == "") else 0.0
    return 0.45 * sf + 0.45 * rc + 0.10 * tm


def row_score(pred, true):
    np_, nt = len(pred), len(true)
    if np_ == 0 and nt == 0:
        return 1.0
    if np_ == 0 or nt == 0:
        return 0.0
    pairs = []
    for i, pe in enumerate(pred):
        for j, te in enumerate(true):
            pairs.append((pair_quality(pe, te), i, j))
    pairs.sort(key=lambda x: -x[0])
    used_p, used_t = set(), set()
    tot = 0.0
    for q, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        tot += q
    mx = max(nt, np_)
    matched_utility = tot / mx
    budget_fidelity = min(nt, np_) / mx
    return 0.85 * matched_utility + 0.15 * budget_fidelity


def elru(pred_map, true_map, langs_map, detail=False):
    """pred_map/true_map: id -> list[edit dict]; langs_map: id -> language."""
    per_lang = {}
    for _id, true in true_map.items():
        lang = langs_map[_id]
        pred = pred_map.get(_id, [])
        rs = row_score(pred, true)
        d = per_lang.setdefault(lang, {"edited": [], "unchanged": []})
        d["edited" if len(true) > 0 else "unchanged"].append(rs)
    lang_scores = {}
    for lang, d in per_lang.items():
        parts = []
        w = []
        if d["edited"]:
            parts.append(sum(d["edited"]) / len(d["edited"]))
            w.append(0.75)
        if d["unchanged"]:
            parts.append(sum(d["unchanged"]) / len(d["unchanged"]))
            w.append(0.25)
        lang_scores[lang] = sum(p * x for p, x in zip(w, parts)) / sum(w)
    score = sum(lang_scores.values()) / len(lang_scores)
    if detail:
        det = {lang: {"edited_mean": (sum(d["edited"]) / len(d["edited"]) if d["edited"] else None),
                      "unchanged_mean": (sum(d["unchanged"]) / len(d["unchanged"]) if d["unchanged"] else None),
                      "n_edited": len(d["edited"]), "n_unchanged": len(d["unchanged"]),
                      "lang_score": lang_scores[lang]}
               for lang, d in per_lang.items()}
        return score, det
    return score


def score_frames(pred_df, true_df):
    """pred_df: id, edits_json; true_df: id, language, edits_json."""
    tm = {r.id: json.loads(r.edits_json) for r in true_df.itertuples()}
    pm = {r.id: json.loads(r.edits_json) for r in pred_df.itertuples()}
    lm = {r.id: r.language for r in true_df.itertuples()}
    return elru(pm, tm, lm, detail=True)


def validate_edits(edits, text_len):
    """Submission validity check for one row's edit list."""
    if not isinstance(edits, list) or len(edits) > 8:
        return False
    prev_end = -1
    for e in edits:
        if set(e.keys()) != {"start", "end", "replacement"}:
            return False
        s, en, rep = e["start"], e["end"], e["replacement"]
        if not (isinstance(s, int) and isinstance(en, int) and isinstance(rep, str)):
            return False
        if not (0 <= s < en <= text_len):
            return False
        if len(rep) > 160:
            return False
        if s < prev_end:  # sorted by start and non-overlapping
            return False
        prev_end = en
    return True
