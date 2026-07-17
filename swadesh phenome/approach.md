# Approach — Swadesh Phoneme Cipher Decoding

## Summary
The task is to invert a single global substitution cipher: every IPA segment of one hidden
Uralic language was replaced by an opaque token, and I must recover the token to segment map
and decode the target lexicon. My solution is an unsupervised statistical decipherment that
learns the map from regular sound correspondences between the enciphered target and its
true-IPA Uralic relatives, then commits to one strict one-to-one map.

## Method
1. Crib = the Uralic relatives (the target is stated to be Uralic). Candidate target segments
   are the Uralic segments seen in at least 2 languages.
2. For every enciphered target word I run a monotonic Needleman-Wunsch alignment against each
   same-concept relative form and accumulate soft co-occurrence counts between tokens and
   segments, weighted by length similarity (a cognate proxy) and by how related the language
   is to the target.
3. The counts are turned into pointwise mutual information, which corrects for the fact that a
   few segments (t, k, a) co-occur with everything. The global token to segment map is the
   Hungarian assignment that maximises PMI plus a frequency prior. The frequency prior is
   important: raw PMI prefers rare marked variants, whereas the true segment for a frequent
   token is almost always the plain common one, and the prior is weighted toward the segments
   of the closest relatives, i.e. the target's own likely inventory.
4. The whole thing is an EM loop: the current map re-estimates language relatedness and refines
   the alignments. Damped counts and a strong gap penalty keep it stable. I keep the iteration
   whose decoded words best resemble their nearest cognate, a label-free selection criterion.
5. Final prediction blends a small ensemble of configurations by averaging their assignment
   score matrices before the single global assignment, which lowers variance on the near
   neighbour vowel and consonant confusions.

## Validation
Because there are no target labels, I validate by leave-one-Uralic-language-out simulation:
take a known Uralic language, encipher it with a random bijection, recover it from the other
relatives, and score against truth. This mirrors the real task exactly. The decoding
self-consistency (mean similarity of each decoded word to its nearest cognate) correlates with
the true score at r = 0.90 across folds and is computable on the real test, giving an honest
score estimate for the hidden target.

## What the model found
On the real test the map locks onto Finnic relatives, closest to Estonian and Veps, and the
decoded words are recognisable Finnic core vocabulary (berry = marja, eleven = üksteist,
ring = sormus). The target is a divergent large-inventory Finnic language. Finnic targets
recover well in cross validation.

## What did not work
Assigning tokens by raw co-occurrence counts collapses onto the few most frequent segments.
Hard gating of cognate pairs by decoded similarity oscillates and destroys the signal. Pure
PMI without a frequency prior systematically picks rare marked segment variants. Over
concentrating on a single closest relative (very high relatedness power) loses the consensus
benefit of several relatives.

## Compliance
Fully unsupervised and derived only from the provided files. No external lexicons, no language
model memory, no hardcoded answers. The map is a single global bijection recovered from the
data.
