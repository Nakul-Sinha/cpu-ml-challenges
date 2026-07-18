"""
Art-Auction Sale Reconstruction -- official, self-contained solution.

Usage: python solution.py <dataset_public_dir> <out_submission_csv>
Reads <dir>/train.csv and <dir>/test.csv, learns a pairwise same-sale model from the training
pools, groups each test pool's lots into sales (seed clusters on the shared consignor, then
agglomeratively merge only confident pairs), writes the grouping to the output path (and mirrors
./working/submission.csv).

Learned from the provided pools only; no pool ids, lot ordering, or external data.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier

THRESHOLD = 0.30   # merge clusters while average same-sale prob exceeds this (test-faithful CV peak)
AUGMENT = 2        # masked copies per train pool (robustness to the test's lower field coverage)
KEEP = dict(seller_keep=0.605 / 0.688, price_keep=0.352 / 0.497, nat_keep=0.640 / 0.719)

FIELDS = ["artist", "seller", "nationality", "object", "materials", "currency", "logprice"]
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
    "aquarell": "watercolor", "pastel": "pastel",
    "enamel": "enamel", "émail": "enamel", "email": "enamel",
    "tapestry": "tapestry", "tapisserie": "tapestry",
    "mosaic": "mosaic", "mosaïque": "mosaic", "mosaique": "mosaic",
    "furniture": "furniture", "engraving": "engraving", "gravure": "engraving",
    "print": "print", "estampe": "print",
}
BAD_NAT = {"unknown", "new", "non-unique", "", "n/a", "none"}


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
    return "other"


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
    n = int(row["n_lots"])
    return [parse_lot(row[f"lot_{i:02d}"]) for i in range(1, n + 1)]


def mask_lot(lot, rng, seller_keep=1.0, price_keep=1.0, nat_keep=1.0):
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
    ac, nc, oc, mc = Counter(), Counter(), Counter(), Counter()
    prices = []; nseller = 0; nlot = 0
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
    artist_one_seller = {a: next(iter(s)) for a, s in artist_sellers.items() if len(s) == 1}
    prices_sorted = np.sort(prices) if prices else np.array([])
    pmed = float(np.median(prices_sorted)) if len(prices_sorted) else np.nan
    return {"ac": ac, "nc": nc, "oc": oc, "mc": mc, "n": max(nlot, 1),
            "seller_cov": nseller / max(nlot, 1), "prices": prices_sorted, "pmed": pmed,
            "artist_one_seller": artist_one_seller, "artist_sellers": artist_sellers}


def eff_seller(lot, ctx):
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
    f = []
    sa, sb = a["seller"], b["seller"]
    f.append(1.0 if (sa and sb and sa == sb) else 0.0)
    f.append(1.0 if (sa and sb and sa != sb) else 0.0)
    f.append(1.0 if (not sa or not sb) else 0.0)
    aa, ab = a["artist"], b["artist"]
    f.append(1.0 if (aa and ab and aa == ab) else 0.0)
    f.append(1.0 if (a["nat_ok"] and b["nat_ok"] and a["nationality"] == b["nationality"]) else 0.0)
    f.append(1.0 if (a["nat_ok"] and b["nat_ok"] and a["nationality"] != b["nationality"]) else 0.0)
    f.append(1.0 if (not a["nat_ok"] or not b["nat_ok"]) else 0.0)
    oa, ob = a["object_n"], b["object_n"]
    f.append(1.0 if (oa and ob and oa == ob) else 0.0)
    f.append(1.0 if (oa and ob and oa != ob) else 0.0)
    ma, mb = a["materials_n"], b["materials_n"]
    f.append(1.0 if (ma and mb and ma == mb) else 0.0)
    f.append(1.0 if (ma and mb and ma != mb) else 0.0)
    f.append(1.0 if (not ma or not mb) else 0.0)
    ca, cb = a["currency"], b["currency"]
    f.append(1.0 if (ca and cb and ca == cb) else 0.0)
    f.append(1.0 if (ca and cb and ca != cb) else 0.0)
    pa, pb = a["price"], b["price"]
    if np.isnan(pa) or np.isnan(pb):
        f.append(0.0); f.append(1.0)
    else:
        f.append(abs(pa - pb)); f.append(0.0)
    if ctx is None:
        f += [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5]
    else:
        n = ctx["n"]
        f.append(1.0 / ctx["ac"].get(a["artist"], n) if (a["artist"] and a["artist"] == b["artist"]) else 0.0)
        f.append(1.0 / ctx["nc"].get(a["nationality"], n)
                 if (a["nat_ok"] and b["nat_ok"] and a["nationality"] == b["nationality"]) else 0.0)
        f.append(1.0 / ctx["oc"].get(a["object_n"], n)
                 if (a["object_n"] and a["object_n"] == b["object_n"]) else 0.0)
        f.append(1.0 / ctx["mc"].get(a["materials_n"], n)
                 if (a["materials_n"] and a["materials_n"] == b["materials_n"]) else 0.0)
        ra, rb = _rank(ctx, pa), _rank(ctx, pb)
        f.append(abs(ra - rb) if (ra is not None and rb is not None) else 0.5)
        f.append(ctx["seller_cov"])
        ea, eb = eff_seller(a, ctx), eff_seller(b, ctx)
        f.append(1.0 if (ea and eb and ea == eb) else 0.0)
        f.append(1.0 if (ea and eb and ea != eb) else 0.0)
        f.append(1.0 if (not ea or not eb) else 0.0)
        asl = ctx["artist_sellers"]
        bridge = 0.0
        if b["seller"] and a["artist"] in asl and b["seller"] in asl[a["artist"]]:
            bridge = 1.0
        if a["seller"] and b["artist"] in asl and a["seller"] in asl[b["artist"]]:
            bridge = 1.0
        f.append(bridge)
        pm = ctx["pmed"]
        if not np.isnan(pa) and not np.isnan(pb) and not np.isnan(pm):
            f.append(1.0 if ((pa >= pm) == (pb >= pm)) else 0.0)
        else:
            f.append(0.5)
    return f


def prob_matrix(lots, model):
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


def group_pool(lots, model, threshold=0.30, min_clusters=6):
    n = len(lots)
    idx = [i for i in range(n) if lots[i] is not None]
    P, _ = prob_matrix(lots, model)
    clusters, seller_to = [], {}
    for i in idx:
        s = lots[i]["seller"]
        if s:
            if s not in seller_to:
                seller_to[s] = len(clusters); clusters.append([])
            clusters[seller_to[s]].append(i)
        else:
            clusters.append([i])
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


def _pairs(lots, g, X, y):
    ctx = pool_context(lots)
    n = len(lots)
    for a in range(n):
        if lots[a] is None:
            continue
        for b in range(a + 1, n):
            if lots[b] is None:
                continue
            X.append(pair_features(lots[a], lots[b], ctx))
            y.append(1 if g[a] == g[b] else 0)


def build_pairs(df):
    X, y = [], []
    rng = np.random.default_rng(0)
    for _, row in df.iterrows():
        lots = pool_lots(row)
        g = str(row["grouping"]).split()
        _pairs(lots, g, X, y)
        for _ in range(AUGMENT):
            _pairs(mask_pool(lots, rng, **KEEP), g, X, y)
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    public_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/public"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "working/submission.csv"

    tr = pd.read_csv(os.path.join(public_dir, "train.csv"))
    te = pd.read_csv(os.path.join(public_dir, "test.csv"))

    X, y = build_pairs(tr)
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07, max_leaf_nodes=31,
                                           min_samples_leaf=40, l2_regularization=1.0,
                                           random_state=0)
    model.fit(X, y)

    ids, preds = [], []
    for _, row in te.iterrows():
        lots = pool_lots(row)
        n = int(row["n_lots"])
        lab = group_pool(lots, model, threshold=THRESHOLD)[:n]
        ids.append(row["id"])
        preds.append(" ".join(str(int(x)) for x in lab))

    out = pd.DataFrame({"id": ids, "prediction": preds})
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    os.makedirs("working", exist_ok=True)
    out.to_csv(os.path.join("working", "submission.csv"), index=False)
    print(f"wrote {len(out)} rows -> {out_path} (+ working/submission.csv)")


if __name__ == "__main__":
    main()
