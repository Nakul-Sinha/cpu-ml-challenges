# solution.py boilerplate contract (both challenges)

Runtime command: `python3 solution.py <public_dir> <submission_out>`

Required scaffolding for the final scripts:

1. Path handling:
   - public_dir = sys.argv[1] if len(sys.argv) > 1 else autodetect
   - submission_out = sys.argv[2] if len(sys.argv) > 2 else "working/submission.csv"
   - autodetect candidates: ["dataset/public", "dataset", ".", "/kaggle/input"], then walk
     for a dir containing train.csv+test.csv (grader-reproduction memory).
   - submission_out.parent.mkdir(parents=True, exist_ok=True)

2. Wall-clock safeguard: T0=time.time() at import; budget checks before each expensive
   stage; degrade gracefully (skip enrichment stages, shrink models) so a valid CSV is
   ALWAYS written < 80 min. Final fallback writer wrapped in try/except producing a
   trivial-but-valid submission if the main path dies.

3. CSV hygiene: keep_default_na=False everywhere (docgap targets like 'nan'; ins ids);
   UTF-8 output; exact column names/order; quoting handled by pandas default (QUOTE_MINIMAL).

4. Version robustness: grader pandas may be 2.x (Box 2 dev used 3.0.3) - use only stable
   APIs (no .map on DataFrame, no deprecated groupby.apply patterns, include_groups only
   if available -> avoid entirely; plain dict/loop code preferred in hot paths anyway).

5. Threads: respect ~10 cores - set OMP_NUM_THREADS via os.environ BEFORE numpy import
   inside solution.py top; lightgbm n_jobs=10; torch.set_num_threads(10).

6. Determinism: fixed seeds (numpy/random/torch/lgbm seed + deterministic feature order).

7. In-script validation before writing: row count == len(test), id match, schema checks
   (ins: validate_edits per row; docgap: non-empty string predictions), then write, then
   re-read + re-validate the written file.

8. No reading of sample_submission as content source (schema reference only), no test
   fitting, no caching artifacts between runs, single self-contained file.
