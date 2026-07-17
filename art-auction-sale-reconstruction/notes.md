# Art-Auction Sale Reconstruction — notes

## Task
Each pool = lots of 6 sales shuffled; recover the 6-way grouping. Metric: mean ARI over pools.
AI baseline 0.55. Lot = `artist :: seller :: nationality :: object :: materials :: currency :: log_price`.

## Key dataset facts (EDA)
- 449 train pools, 256 test pools, ~43 lots/pool, exactly 6 sales/pool.
- Field fill (train): artist 1.00, seller 0.69, nationality 0.72 (Unknown/NEW/NON-UNIQUE are
  non-informative), object 1.00 (multilingual: Painting/Peinture/Gemälde), materials 0.27
  (multilingual), currency 0.58, log_price 0.50.
- **Consignor is the dominant cue**: P(same-sale | same-seller) = 0.975; a seller spans >1 sale
  in only 1.7% of cases. But seller is recorded for only ~69% of lots.
- P(same-sale | same-artist) = 0.25 (weak; artists span sales).
- **Test is harder than train**: coverage seller 0.605 vs 0.688, price 0.352 vs 0.497,
  nationality 0.640 vs 0.719. So a train-only CV is optimistic; we mask validation pools (and
  augment training) to the test rates for an honest estimate.

## Method
1. Cross-lingual normalisation of object/materials to canonical categories so one model
   transfers across pools of different catalogue languages.
2. Pairwise same-sale model (HistGradientBoosting) over 25 features: seller/artist/nationality/
   object/materials/currency matches, |price diff| and within-pool price-rank gap, **rarity-
   weighted** matches (a shared attribute rare in the pool is stronger evidence), pool seller
   coverage, and an **effective-seller** feature that imputes a consignor for no-seller lots
   whose artist co-occurs with exactly one consignor in the pool.
3. Grouping: seed clusters by consignor (near-deterministic), then agglomeratively merge the two
   clusters with the highest average same-sale probability while it exceeds a threshold and >6
   clusters remain. Conservative by design — ARI rewards not merging different sales.

## Validation
Group CV over pools (mean ARI). ORACLE-prob clustering ceiling = 0.988 (the clustering is not
the bottleneck; the no-seller pairwise prediction is, AUC ~0.69). Baselines reproduced:
all-one-group 0.00, all-singletons 0.00, share-consignor 0.70-0.71 (higher than the problem's
0.43 because we keep no-seller lots separate rather than lumping them).

## Progress
- Basic pair features + forced-6 agglomerative: 0.57 (false merges hurt ARI).
- Consignor-seed + confident-merge (threshold ~0.25): 0.716 train CV.
- + rarity features: no-seller AUC 0.66 -> 0.69. + effective-seller: 0.717 train CV.
- **Test-faithful (masked-to-test-coverage) CV**: 0.614 (no aug) -> 0.620 (masked-augment
  training). This is the honest estimate for the real test; AI baseline 0.55.
- Seeding clusters by the artist-imputed consignor HURTS (0.60): imputation errors cause false
  merges. Kept only as a model feature.
- Locked: masked-augment training (2 copies/pool), merge threshold 0.28.

## Compliance
Learned pairwise model over lot content only; no pool ids, ordering, or external data.
`solution.py` reads dataset/public/ and writes working/submission.csv.
