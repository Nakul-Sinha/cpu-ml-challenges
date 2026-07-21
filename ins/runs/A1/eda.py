"""EDA for detection: token alignment, span adjacency, token-count dist, special chars."""
import sys, json, re, collections
import pandas as pd

WORD_RE = re.compile(r"\S+")
train = pd.read_csv("dataset/train.csv")
train["edits"] = train.edits_json.apply(json.loads)

def toks(text):
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]

# 1) alignment: does every edit start AND end on a token boundary?
starts_ok = ends_ok = both_ok = tot_edits = 0
tokcount = collections.Counter()
adj_distinct = 0          # pairs of true edits that are token-adjacent (no gap token between)
gap0_charadjacent = 0     # edits whose char end == next edit char start (touching)
deletion = 0
for r in train.itertuples():
    t = r.text
    tk = toks(t)
    starts = {s for s, e in tk}
    ends = {e for s, e in tk}
    spans = sorted([(e["start"], e["end"], e["replacement"]) for e in r.edits])
    # token index ranges per edit
    tok_ranges = []
    for a, b, rep in spans:
        tot_edits += 1
        so = a in starts
        eo = b in ends
        starts_ok += so; ends_ok += eo; both_ok += (so and eo)
        if rep == "": deletion += 1
        # count tokens fully inside [a,b]
        idxs = [i for i, (s, e) in enumerate(tk) if s >= a and e <= b]
        tok_ranges.append((idxs[0] if idxs else None, idxs[-1] if idxs else None))
        tokcount[len(idxs)] += 1
    # adjacency: consecutive edits with token index gap == 1 (no unedited token between)
    for i in range(len(tok_ranges) - 1):
        r0, r1 = tok_ranges[i], tok_ranges[i + 1]
        if r0[1] is not None and r1[0] is not None and r1[0] == r0[1] + 1:
            adj_distinct += 1
    for i in range(len(spans) - 1):
        if spans[i][1] == spans[i + 1][0]:
            gap0_charadjacent += 1

print(f"edits total={tot_edits} starts_on_tok={starts_ok} ends_on_tok={ends_ok} both={both_ok}")
print(f"token-adjacent distinct edit pairs (need B/I split): {adj_distinct}")
print(f"char-touching edit pairs: {gap0_charadjacent}")
print(f"deletions: {deletion}")
print("tokens-per-span dist:", dict(sorted(tokcount.items())))

# 2) special mid-token chars in edited single-token spans vs all tokens
edited_single = collections.Counter()
all_tok_chars = collections.Counter()
edited_tokens = set()
for r in train.itertuples():
    t = r.text
    spanset = {(e["start"], e["end"]) for e in r.edits}
    for s, e in toks(t):
        w = t[s:e]
        inner = w[1:-1] if len(w) > 2 else ""
        for ch in set(inner):
            if not ch.isalnum():
                all_tok_chars[ch] += 1
        if (s, e) in spanset:
            for ch in set(inner):
                if not ch.isalnum():
                    edited_single[ch] += 1
print("\nmid-token non-alnum chars in EXACT single-token edits (top):", edited_single.most_common(12))
print("mid-token non-alnum chars over ALL tokens (top):", all_tok_chars.most_common(12))

# 3) fraction of edited rows / tokens per language
for lang, d in train.groupby("language"):
    ne = d.edits.apply(len)
    ntok = d.text.apply(lambda x: len(toks(x)))
    edtok = 0
    for r in d.itertuples():
        sp = sorted([(e["start"], e["end"]) for e in r.edits])
        tk = toks(r.text)
        for a, b in sp:
            edtok += sum(1 for s, e in tk if s >= a and e <= b)
    print(f"{lang}: rows={len(d)} edited_rows={(ne>0).sum()} tot_tokens={ntok.sum()} edited_tokens={edtok} tok_edit_rate={edtok/ntok.sum():.4f}")
