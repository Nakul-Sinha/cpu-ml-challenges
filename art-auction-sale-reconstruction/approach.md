# Approach — Art-Auction Sale Reconstruction

## Summary
Each pool mixes the lots of 6 sales and the task is to recover the grouping. I treat it as
record linkage: a learned pairwise same-sale model over lot attributes, followed by a
consignor-seeded agglomerative grouping that merges only confident pairs.

## Method
1. Parse the 7 fields and normalise object type and materials across languages (Painting,
   Peinture, Gemalde all map to one category) so a single model works on pools of any
   catalogue language.
2. A gradient-boosted classifier scores every within-pool pair for same-sale. Features: matches
   on consignor, artist, nationality, object, materials and currency; absolute log-price gap and
   within-pool price-rank gap; rarity-weighted matches, since a shared attribute that is rare in
   the pool is much stronger evidence than a common one; the pool's consignor coverage; and an
   effective-consignor feature that imputes a consignor for a lot that has none when its artist
   co-occurs with exactly one consignor elsewhere in the pool.
3. Grouping seeds one cluster per consignor (a near deterministic same-sale cue), then
   repeatedly merges the two clusters with the highest average same-sale probability while that
   average is above a threshold and more than 6 clusters remain. This is deliberately
   conservative because the metric rewards not putting lots of different sales together.

## Why this beats simple rules
The consignor rule alone recovers a large part of the structure but leaves the ~40 percent of
lots with no recorded consignor unplaced and never merges the multi-consignor sales. The model
learns which weak cues actually indicate shared provenance and, together with the rarity and
imputed-consignor features, places many of the no-consignor lots and merges consignors that
belong to the same sale.

## Validation
Group cross validation over pools, mean ARI. Crucially the test set has lower field coverage
than train (consignor 0.61 vs 0.69, price 0.35 vs 0.50, nationality 0.64 vs 0.72), so a plain
train CV is optimistic. I mask validation pools down to the test coverage rates for an honest
estimate and augment training with masked copies so the model is robust to missing fields. An
oracle-probability version of the clustering reaches ARI 0.99, confirming the grouping step is
not the bottleneck; the remaining gap is the difficulty of the no-consignor pairs.

## What did not work
Forcing exactly 6 clusters with off-the-shelf agglomerative clustering scores far lower, because
it forces false merges. Lumping all no-consignor lots into a single group is also worse than
keeping them separate. Very low merge thresholds over-merge and hurt ARI.

## Compliance
The only signal is lot content within each pool. No pool ids, no lot ordering, no row order, no
external data. The grouping comes from a model trained on the provided training pools.
