"""Art-Auction Sale Reconstruction -- shared parsing, normalisation, and pair features.

Each pool is a shuffle of the lots of six sales; we must recover the six-way grouping. A
shared consignor (seller) is a near-deterministic same-sale cue (P(same-sale|same-seller)=0.97
in train) but is recorded for only ~69% of lots. Object-type and materials strings are
multilingual (Painting/Peinture/Gemalde), so we normalise them to canonical categories so a
single pairwise model transfers across pools of different catalogue languages.
"""
import numpy as np
import pandas as pd

FIELDS = ["artist", "seller", "nationality", "object", "materials", "currency", "logprice"]
LOTCOLS = [f"lot_{i:02d}" for i in range(1, 49)]

# object type -> canonical category (cross-lingual)
OBJ_MAP = {
    "painting": "painting", "peinture": "painting", "gemalde": "painting", "gemälde": "painting",
    "schilderij": "painting", "pittura": "painting", "quadro": "painting", "cuadro": "painting",
    "dipinto": "painting", "tableau": "painting",
    "drawing": "drawing", "dessin": "drawing", "zeichnung": "drawing", "tekening": "drawing",
    "disegno": "drawing", "dibujo": "drawing",
    "sculpture": "sculpture", "skulptur": "sculpture", "beeld": "sculpture", "scultura": "sculpture",
    "escultura": "sculpture", "statue": "sculpture",
    "miniature": "miniature", "miniatur": "miniature",
    "watercolor": "watercolor", "aquarelle": "watercolor", "watercolour": "watercolor",
    "aquarell": "watercolor",
    "pastel": "pastel",
    "enamel": "enamel", "émail": "enamel", "email": "enamel",
    "tapestry": "tapestry", "tapisserie": "tapestry",
    "mosaic": "mosaic", "mosaïque": "mosaic", "mosaique": "mosaic",
    "furniture": "furniture", "engraving": "engraving", "gravure": "engraving",
    "print": "print", "estampe": "print",
}

# materials -> canonical support/medium
def norm_materials(s):
    s = s.lower().strip().strip("[]").strip()
    if not s:
        return ""
    if any(k in s for k in ["toile", "canvas", "doek", "leinwand", "tela", "lienzo"]):
        return "canvas"
    if any(k in s for k in ["bois", "panel", "paneel", "holz", "tavola", "board", "wood", "tabla"]):
        return "panel"
    if any(k in s for k in ["cuivre", "copper", "koper", "kupfer", "rame"]):
        return "copper"
    if any(k in s for k in ["marble", "marbre", "marmor", "marmo"]):
        return "marble"
    if any(k in s for k in ["bronze", "brons"]):
        return "bronze"
    if any(k in s for k in ["paper", "papier", "papel", "carta"]):
        return "paper"
    if any(k in s for k in ["glass", "verre", "glas"]):
        return "glass"
    if "oil" in s or "huile" in s or "olie" in s or "öl" in s or "olio" in s:
        return "oil"
    if any(k in s for k in ["chalk", "crayon", "craie", "kreide"]):
        return "chalk"
    if "gouache" in s:
        return "gouache"
    if "copper" in s:
        return "copper"
    return "other"

BAD_NAT = {"unknown", "new", "non-unique", "", "n/a", "none"}


def norm_obj(s):
    s = s.lower().strip()
    if ";" in s:
        s = s.split(";")[0].strip()
    return OBJ_MAP.get(s, s if s else "")


def parse_lot(s):
    if not isinstance(s, str) or not s.strip():
        return None
    parts = [x.strip() for x in s.split("::")]
    parts = parts + [""] * (7 - len(parts))
    d = dict(zip(FIELDS, parts[:7]))
    d["object_n"] = norm_obj(d["object"])
    d["materials_n"] = norm_materials(d["materials"])
    d["nat_ok"] = d["nationality"].lower() not in BAD_NAT
    try:
        d["price"] = float(d["logprice"]) if d["logprice"] else np.nan
    except ValueError:
        d["price"] = np.nan
    return d


def pool_lots(row):
    """List of parsed lots for a pool row (in lot order), length n_lots."""
    n = int(row["n_lots"])
    out = []
    for i in range(1, n + 1):
        out.append(parse_lot(row[f"lot_{i:02d}"]))
    return out


def mask_lot(lot, rng, seller_keep=1.0, price_keep=1.0, nat_keep=1.0):
    """Return a copy of the lot with some fields dropped, to simulate the test set's lower field
    coverage (test has less seller/price/nationality than train)."""
    if lot is None:
        return None
    l = dict(lot)
    if l["seller"] and rng.random() > seller_keep:
        l["seller"] = ""
    if not np.isnan(l["price"]) and rng.random() > price_keep:
        l["price"] = np.nan
    if l["nat_ok"] and rng.random() > nat_keep:
        l["nat_ok"] = False
    return l


def mask_pool(lots, rng, **kw):
    return [mask_lot(l, rng, **kw) for l in lots]


def pool_context(lots):
    """Per-pool statistics so pair features can be rarity-aware: a shared attribute is strong
    evidence of a shared sale only when it is rare within the pool. Also imputes a seller for
    no-seller lots whose artist co-occurs with exactly one consignor in the pool."""
    from collections import Counter, defaultdict
    ac, nc, oc, mc = Counter(), Counter(), Counter(), Counter()
    prices = []
    nseller = 0; nlot = 0
    artist_sellers = defaultdict(set)
    for l in lots:
        if l is None:
            continue
        nlot += 1
        ac[l["artist"]] += 1
        if l["nat_ok"]:
            nc[l["nationality"]] += 1
        if l["object_n"]:
            oc[l["object_n"]] += 1
        if l["materials_n"]:
            mc[l["materials_n"]] += 1
        if not np.isnan(l["price"]):
            prices.append(l["price"])
        if l["seller"]:
            nseller += 1
            if l["artist"]:
                artist_sellers[l["artist"]].add(l["seller"])
    # artist -> the unique consignor it appears with (if unambiguous), for seller imputation
    artist_one_seller = {a: next(iter(s)) for a, s in artist_sellers.items() if len(s) == 1}
    prices_sorted = np.sort(prices) if prices else np.array([])
    return {"ac": ac, "nc": nc, "oc": oc, "mc": mc, "n": max(nlot, 1),
            "seller_cov": nseller / max(nlot, 1), "prices": prices_sorted,
            "artist_one_seller": artist_one_seller}


def eff_seller(lot, ctx):
    """Actual consignor, or the one imputed from an unambiguous artist->seller link in the pool."""
    if lot["seller"]:
        return lot["seller"]
    if ctx is not None:
        return ctx["artist_one_seller"].get(lot["artist"], "")
    return ""


def _rank(ctx, p):
    ps = ctx["prices"]
    if len(ps) == 0 or np.isnan(p):
        return None
    return np.searchsorted(ps, p) / len(ps)


def pair_features(a, b, ctx=None):
    """Feature vector for a pair of parsed lots in the same pool."""
    f = []
    # seller: strongest cue. 3-state: match / mismatch / unknown
    sa, sb = a["seller"], b["seller"]
    f.append(1.0 if (sa and sb and sa == sb) else 0.0)       # seller_match
    f.append(1.0 if (sa and sb and sa != sb) else 0.0)       # seller_mismatch
    f.append(1.0 if (not sa or not sb) else 0.0)             # seller_unknown
    # artist
    aa, ab = a["artist"], b["artist"]
    f.append(1.0 if (aa and ab and aa == ab) else 0.0)       # artist_match
    # nationality (only meaningful when both known/specific)
    f.append(1.0 if (a["nat_ok"] and b["nat_ok"] and a["nationality"] == b["nationality"]) else 0.0)
    f.append(1.0 if (a["nat_ok"] and b["nat_ok"] and a["nationality"] != b["nationality"]) else 0.0)
    f.append(1.0 if (not a["nat_ok"] or not b["nat_ok"]) else 0.0)
    # object type (normalised)
    oa, ob = a["object_n"], b["object_n"]
    f.append(1.0 if (oa and ob and oa == ob) else 0.0)
    f.append(1.0 if (oa and ob and oa != ob) else 0.0)
    # materials (normalised, sparse)
    ma, mb = a["materials_n"], b["materials_n"]
    f.append(1.0 if (ma and mb and ma == mb) else 0.0)
    f.append(1.0 if (ma and mb and ma != mb) else 0.0)
    f.append(1.0 if (not ma or not mb) else 0.0)
    # currency
    ca, cb = a["currency"], b["currency"]
    f.append(1.0 if (ca and cb and ca == cb) else 0.0)
    f.append(1.0 if (ca and cb and ca != cb) else 0.0)
    # price
    pa, pb = a["price"], b["price"]
    if np.isnan(pa) or np.isnan(pb):
        f.append(0.0)      # price_absdiff
        f.append(1.0)      # price_missing
    else:
        f.append(abs(pa - pb))
        f.append(0.0)
    # ---- pool-context / rarity features: a shared attribute rare in the pool is stronger ----
    if ctx is None:
        f += [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0]
    else:
        n = ctx["n"]
        # rarity-weighted shared attribute = 1/count-in-pool when the pair matches, else 0
        f.append(1.0 / ctx["ac"].get(a["artist"], n) if (a["artist"] and a["artist"] == b["artist"]) else 0.0)
        f.append(1.0 / ctx["nc"].get(a["nationality"], n)
                 if (a["nat_ok"] and b["nat_ok"] and a["nationality"] == b["nationality"]) else 0.0)
        f.append(1.0 / ctx["oc"].get(a["object_n"], n)
                 if (a["object_n"] and a["object_n"] == b["object_n"]) else 0.0)
        f.append(1.0 / ctx["mc"].get(a["materials_n"], n)
                 if (a["materials_n"] and a["materials_n"] == b["materials_n"]) else 0.0)
        ra, rb = _rank(ctx, pa), _rank(ctx, pb)
        f.append(abs(ra - rb) if (ra is not None and rb is not None) else 0.5)  # price rank gap
        f.append(ctx["seller_cov"])  # how much to trust the consignor signal in this pool
        # effective seller (actual or artist-imputed): match / mismatch / unknown
        ea, eb = eff_seller(a, ctx), eff_seller(b, ctx)
        f.append(1.0 if (ea and eb and ea == eb) else 0.0)
        f.append(1.0 if (ea and eb and ea != eb) else 0.0)
        f.append(1.0 if (not ea or not eb) else 0.0)
    return f


N_FEAT = 25
FEAT_NAMES = ["seller_match", "seller_mismatch", "seller_unknown", "artist_match",
              "nat_match", "nat_mismatch", "nat_unknown", "obj_match", "obj_mismatch",
              "mat_match", "mat_mismatch", "mat_unknown", "cur_match", "cur_mismatch",
              "price_absdiff", "price_missing",
              "artist_rarity", "nat_rarity", "obj_rarity", "mat_rarity",
              "price_rankgap", "pool_seller_cov",
              "effseller_match", "effseller_mismatch", "effseller_unknown"]


def prob_matrix(lots, model):
    """n x n same-sale probability matrix over the pool's lots (0 for empty slots)."""
    n = len(lots)
    idx = [i for i in range(n) if lots[i] is not None]
    ctx = pool_context(lots)
    P = np.zeros((n, n))
    feats, ij = [], []
    for x, a in enumerate(idx):
        for b in idx[x + 1:]:
            feats.append(pair_features(lots[a], lots[b], ctx))
            ij.append((a, b))
    if feats:
        pr = model.predict_proba(np.array(feats, dtype=np.float32))[:, 1]
        for (a, b), p in zip(ij, pr):
            P[a, b] = P[b, a] = p
    return P, idx


def group_pool(lots, model, threshold=0.5, min_clusters=6, seed_seller=True):
    """Seed clusters by shared consignor (a near-deterministic same-sale cue), then greedily
    merge the two clusters with the highest average same-sale probability while that average
    exceeds `threshold` and more than `min_clusters` clusters remain. Conservative by design:
    ARI rewards not merging different sales, so we only merge confident pairs."""
    n = len(lots)
    idx = [i for i in range(n) if lots[i] is not None]
    P, _ = prob_matrix(lots, model)
    clusters, seller_to = [], {}
    for i in idx:
        s = lots[i]["seller"] if seed_seller else ""
        if s:
            if s not in seller_to:
                seller_to[s] = len(clusters); clusters.append([])
            clusters[seller_to[s]].append(i)
        else:
            clusters.append([i])
    # precompute cluster-pair average affinity greedily
    while len(clusters) > min_clusters:
        best, bi, bj = -1.0, -1, -1
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                aff = np.mean([P[p, q] for p in clusters[a] for q in clusters[b]])
                if aff > best:
                    best, bi, bj = aff, a, b
        if best < threshold:
            break
        clusters[bi] += clusters[bj]
        del clusters[bj]
    labels = np.zeros(n, dtype=int)
    for ci, c in enumerate(clusters):
        for i in c:
            labels[i] = ci
    return labels
