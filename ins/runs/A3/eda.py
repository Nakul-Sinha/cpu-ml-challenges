"""A3 EDA: mark coverage, token-level edit signal, template inducibility."""
import sys, json, re, collections
import pandas as pd

sys.path.insert(0, "solution")
WORD_RE = re.compile(r"\S+")
tr = pd.read_csv("dataset/train.csv")
tr["ed"] = tr.edits_json.apply(json.loads)

def toks(text):
    return [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(text)]

MARKS = set(":*/")
def mark_of(tokstr):
    # mark char that appears NOT at the very ends of the token
    inner = tokstr[1:-1] if len(tokstr) > 2 else ""
    for c in tokstr:
        if c in MARKS:
            return c
    return ""

# 1. span-level: n_tokens dist, first-token-has-mark, any-token-has-mark
span_ntok = collections.Counter()
span_firstmark = collections.Counter()   # (lang, has_mark_anywhere)
del_frac = collections.Counter()
by_lang_spans = collections.Counter()
for r in tr.itertuples():
    for e in r.ed:
        s, en, rep = e["start"], e["end"], e["replacement"]
        span = r.text[s:en]
        stoks = toks(span)
        span_ntok[len(stoks)] += 1
        anymark = any(any(c in MARKS for c in t[2]) for t in stoks)
        span_firstmark[(r.language, anymark)] += 1
        by_lang_spans[r.language] += 1
        if rep == "":
            del_frac[r.language] += 1
print("=== span n_tokens ===", dict(sorted(span_ntok.items())))
print("=== span any-mark by lang (lang,hasmark)->count ===")
for k in sorted(span_firstmark): print("   ", k, span_firstmark[k])
print("=== deletions by lang ===", dict(del_frac), "of", dict(by_lang_spans))

# 2. token-level: edit signal. For each token, is it the START of an edited span?
#    Measure: among tokens WITH a mark, what frac start an edit; among tokens without.
tok_mark_edit = collections.Counter()  # (lang, hasmark, is_edit_start)
# build set of edit-start offsets per row
for r in tr.itertuples():
    starts = {e["start"] for e in r.ed}
    inside = set()
    for e in r.ed:
        inside.add((e["start"], e["end"]))
    span_char = []
    for e in r.ed:
        span_char.append((e["start"], e["end"]))
    for s, en, w in toks(r.text):
        hasmark = any(c in MARKS for c in w)
        # is this token inside any edited span?
        is_edit = any(a <= s and en <= b for a, b in span_char)
        tok_mark_edit[(r.language, hasmark, is_edit)] += 1
print("=== token-level (lang, hasmark, is_edited) -> count ===")
for k in sorted(tok_mark_edit): print("   ", k, tok_mark_edit[k])

# 3. German connector induction: single-token marked src -> rep. Find middle word.
print("=== German single-token marked templates ===")
de_tmpl = collections.Counter()
for r in tr[tr.language == "de"].itertuples():
    for e in r.ed:
        span = r.text[e["start"]:e["end"]]
        stoks = toks(span)
        if len(stoks) == 1 and any(c in MARKS for c in span):
            rep = e["replacement"]
            # source: STEM<mark><suf>. abstract stem.
            m = re.search(r"[:*/]", span)
            mark = span[m.start()]
            stem = span[:m.start()]
            suf = span[m.start()+1:]
            # try to express rep as stem + MID + stem + suf  OR general
            if rep.startswith(stem) and stem:
                tail = rep[len(stem):]
                de_tmpl[(mark, suf, "TEMPLATE:{stem}"+tail.replace(stem, "{stem}"))] += 1
            else:
                de_tmpl[(mark, suf, "NOSTEMPREFIX rep="+rep[:40])] += 1
for k, v in de_tmpl.most_common(25):
    print("   ", v, k)
