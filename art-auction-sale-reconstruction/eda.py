import pandas as pd, numpy as np, io
from collections import Counter

tr = pd.read_csv("dataset/public/train.csv")
te = pd.read_csv("dataset/public/test.csv")
out = io.open("eda_out.txt", "w", encoding="utf-8")
def p(*a): print(*a); print(*a, file=out)

p("train pools:", len(tr), "test pools:", len(te))
p("n_lots: train", tr.n_lots.min(), "-", tr.n_lots.max(), "mean", round(tr.n_lots.mean(),1))
p("n_lots: test ", te.n_lots.min(), "-", te.n_lots.max(), "mean", round(te.n_lots.mean(),1))

FIELDS = ["artist", "seller", "nationality", "object", "materials", "currency", "logprice"]

def parse_lot(s):
    if not isinstance(s, str) or not s.strip():
        return None
    parts = [x.strip() for x in s.split("::")]
    parts = parts + [""] * (7 - len(parts))
    return dict(zip(FIELDS, parts[:7]))

# gather all lots from train
def iter_lots(df):
    for _, row in df.iterrows():
        n = row["n_lots"]
        grouping = str(row.get("grouping", "")).split() if "grouping" in df.columns else None
        for i in range(1, n + 1):
            lot = parse_lot(row[f"lot_{i:02d}"])
            if lot is None:
                continue
            g = grouping[i - 1] if grouping else None
            yield row["id"], i, lot, g

lots = list(iter_lots(tr))
p("\ntotal train lots:", len(lots))

# field fill rates
p("\n=== field fill rates (train) ===")
for f in FIELDS:
    filled = sum(1 for _, _, lot, _ in lots if lot[f])
    p(f"  {f:12}: {filled/len(lots):.3f} filled   distinct={len(set(lot[f] for _,_,lot,_ in lots if lot[f]))}")

# value distributions
p("\n=== nationality values ===")
p(Counter(lot['nationality'] for _,_,lot,_ in lots).most_common(20))
p("\n=== object_type values ===")
p(Counter(lot['object'] for _,_,lot,_ in lots).most_common(20))
p("\n=== currency values ===")
p(Counter(lot['currency'] for _,_,lot,_ in lots).most_common(15))
p("\n=== materials values (top) ===")
p(Counter(lot['materials'] for _,_,lot,_ in lots).most_common(20))

# log_price
prices = [float(lot['logprice']) for _,_,lot,_ in lots if lot['logprice']]
p("\n=== log_price ===")
p("filled:", len(prices), "min", round(min(prices),2), "max", round(max(prices),2),
  "mean", round(np.mean(prices),2), "std", round(np.std(prices),2))

# Per-pool: how much does seller identify the sale? Analyze one pool.
p("\n=== signal analysis: within-pool, do same-seller lots share a sale? ===")
same_seller_same_sale = 0; same_seller_pairs = 0
same_artist_same_sale = 0; same_artist_pairs = 0
tot_pairs = 0; same_sale_pairs = 0
for _, row in tr.iterrows():
    n = row["n_lots"]; grouping = str(row["grouping"]).split()
    plots = [parse_lot(row[f"lot_{i:02d}"]) for i in range(1, n+1)]
    for a in range(n):
        for b in range(a+1, n):
            if plots[a] is None or plots[b] is None: continue
            tot_pairs += 1
            same = grouping[a] == grouping[b]
            same_sale_pairs += same
            if plots[a]['seller'] and plots[a]['seller'] == plots[b]['seller']:
                same_seller_pairs += 1; same_seller_same_sale += same
            if plots[a]['artist'] and plots[a]['artist'] == plots[b]['artist']:
                same_artist_pairs += 1; same_artist_same_sale += same
p(f"total pairs: {tot_pairs}, same-sale base rate: {same_sale_pairs/tot_pairs:.3f}")
p(f"same-seller pairs: {same_seller_pairs}, P(same-sale|same-seller)={same_seller_same_sale/max(same_seller_pairs,1):.3f}")
p(f"same-artist pairs: {same_artist_pairs}, P(same-sale|same-artist)={same_artist_same_sale/max(same_artist_pairs,1):.3f}")

# seller appears in multiple sales?
p("\n=== does a seller span multiple sales within a pool? ===")
multi = 0; total_sellers = 0
for _, row in tr.iterrows():
    n = row["n_lots"]; grouping = str(row["grouping"]).split()
    plots = [parse_lot(row[f"lot_{i:02d}"]) for i in range(1, n+1)]
    seller_sales = {}
    for i in range(n):
        if plots[i] and plots[i]['seller']:
            seller_sales.setdefault(plots[i]['seller'], set()).add(grouping[i])
    for s, sales in seller_sales.items():
        total_sellers += 1
        if len(sales) > 1: multi += 1
p(f"sellers spanning >1 sale: {multi}/{total_sellers} = {multi/max(total_sellers,1):.3f}")

out.close()
print("\n[written eda_out.txt]")
