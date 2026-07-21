# Docstring Gap Restoration. Approach

Recommended time spent value: 15 hours.

## Submission form text

My solution is a learned hybrid of a retrieval system and a code pretrained language model, combined by a trained reranker that decides per row which source to trust.

Retrieval side. From the training set I build anchored context indexes (the 1 or 2 words on each side of the gap, plus skip gram variants, mapped to counters of observed target spans), a gated fuzzy nearest neighbor view over hashed character ngram term frequency vectors (no idf anywhere, per the rules I followed), code derived candidates (function name splits, argument names, return expression identifiers, each scored by a learned prior of how often such strings appear as targets), a word trigram language model with stupid backoff that both generates 1 to 4 word bridge candidates between the left and right context and scores fill fluency, and global frequent spans. The union pool holds about 68 candidates per row with an oracle of 0.63.

Generator side. Salesforce codet5 small (60M parameters, public pretrained weights loaded in script) fills the gap zero shot: the sentence with the gap replaced by the sentinel token, prefixed by a compact code hint built from the def line and the last return line. Measured facts drove this choice: codet5 small zero shot scores 0.415 on my held out bucket versus 0.199 for t5 small zero shot and about 0.31 for t5 small after 15 minutes of CPU fine tuning; fine tuning codet5 at the t5 learning rate regresses it, so the shipped configuration uses the base pretrained weights unmodified with int8 dynamic quantization for inference speed (33 rows per second at 10 threads, quality delta under 0.005).

Fusion. A LightGBM LambdaRank reranker, trained in script on 40k training rows with strict parity hygiene (candidates for a training row are generated from index fits that exclude that row's half of the data, so the model never sees a row through its own anchors), scores every candidate with 83 features: anchored conditional probabilities and ranks, provenance one hots, fuzzy cosine, code priors, fill fluency language model scores, centrality against the pool, length priors, and the codet5 candidate's sequence log probability, length and pool agreement. Decode is pure argmax of the reranker. This trained model materially drives every prediction (the sanctioned frozen retrieval plus trained ranking pattern); measured alternatives (hard confidence overrides, codet5 primary with retrieval rescue) all scored lower.

Validation. All development used an md5 bucket split of the masked sentences: bucket 0 for evaluation, buckets 2 to 19 for fitting, and bucket 1 locked and scored exactly once on the final artifact. Honest numbers: retrieval chassis alone 0.319, codet5 zero shot alone 0.409, shipped hybrid 0.438 on bucket 0 and 0.4392 on the locked bucket 1, an optimism gap of about zero, so I expect roughly 0.44 on the grader. The 2 way oracle between the retrieval pick and codet5 is 0.492, of which the reranker realizes 89 percent. The final timed solo run at 10 threads took 52.4 minutes for the full pipeline including all index builds, reranker training, and 50k row inference, leaving 37 minutes of margin; cumulative wall clock checkpoints degrade gracefully (reduce codet5 coverage, then skip codet5 to the pure retrieval path, then a best constant fallback) so a valid submission is always written.

What worked: the code pretrained generator (the single decisive lever), injecting it as a candidate for all rows rather than gating to weak retrieval rows (it beats retrieval on every anchor tier), the fill fluency language model features, candidate centrality, and the parity hygiene that kept holdout numbers honest (bucket 1 confirmed them). What did not work, measured and dropped: TFIDF free replacements were required anyway, but also BM25 style scoring was never used; retrieval posterior decoding; minimum Bayes risk decoding once centrality was a feature; a neural span classifier whose confidence was anti calibrated; hard confidence overrides; fine tuning codet5; and larger models (codet5 base needs about 80 minutes for inference alone, over budget).

## Compliance notes

- No TFIDF, BM25 or idf weighting anywhere (hashed raw term frequency and count based conditional probabilities only). All CSV reads use keep_default_na=False. Nothing is fit on test rows; test inputs are only read for per row inference.
- Pretrained weights: Salesforce codet5 small, a public, commonly available Hugging Face model, loaded in script (offline cache first, then download), used zero shot; the challenge explicitly allows lightweight CPU language models and bans only large LLM fine tuning, GPU, and hosted APIs. The trained in script component that drives predictions is the LightGBM reranker; if the model cannot load, the script degrades to the pure retrieval pipeline.
- Runtime measured 52.4 minutes at 10 CPU threads end to end; peak memory about 12 GB. Deterministic seeds throughout.
- The script accepts python3 solution.py public_dir submission_out, auto detects the data directory if arguments are missing, asserts 50000 rows, exact id order and zero empty predictions before writing.

## Artifacts

- solution.py (self contained ship artifact, md5 80df5a02b9379f8188b0c3d3d71034d3, verified identical on box and local mirror)
- submission.csv = runs/D3/submission_v3.csv from the final timed run
- Held out predictions and per row dumps: runs/D3/val_pred_v3.csv (bucket 0, 0.4380), runs/D3/val_pred_v3_bucket1.csv (locked bucket 1, 0.4392)
- Full logs and evaluation ladder: runs/D3/final_run.log, runs/D3/d3_eval_full.log, notes.md
