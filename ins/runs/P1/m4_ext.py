"""M4 COMPOSER -- integrate M2 (German) + M3 (Italian/EN/Deletions) plug-ins onto
the M1 base pipeline, resolving conflicts, WITHOUT forking pipeline.py core.

Conflicts resolved here (both specialists were written to be registered ALONE and
both do a DESTRUCTIVE `register(P)` assignment; M4 composes them by hand):

  STORE_BUILDERS  order = [stash_transducer, m2.build_stores, m3.build_stores]
      * stash_transducer runs first so an exact/norm-memory transducer is available
        to exact_first_hook (leak-free: fit on the SAME fold-train frame).
      * m2.build_stores installs stores['span_scorer'] (de fem_strict gate) + M2G.
      * m3.build_stores installs stores['m3'] + _ACTIVE; it only overwrites
        span_scorer when USE_NPGEN (default off), so M2's gate survives -> no clash.

  TOKEN_FEATURE_EXTRAS = [m3.it_en_feats]         (it/en features; +0.010 nested)
      * M2's token_extras stay OFF (measured -0.005 de regression in M2's own runs).
      * Feature-name keys do not collide (de_* vs it_*/en_*); reset FEAT/EXTRA names.

  REPLACEMENT_HOOKS order (task-mandated): exact-memory FIRST, then paired-collapse,
      then NP/slash rewrites, then A2 defaults (the transducer fallback inside
      pipeline.build_edits):
        [exact_first_hook, m2.collapse_hook, m3.it_slash_hook, m3.en_norm_hook, m3.del_hook]
      Each hook is language-gated (m2 de-only, m3 it/en-only) so within a language
      only its own hooks fire; exact_first_hook guarantees a verified train memory
      wins over any heuristic collapse/rewrite.  del_hook stays inert (USE_DEL off).

  SPAN_CANDIDATE_GENERATORS = [m2.span_generator, m3.np_generator]
      * np_generator is inert (USE_NPGEN off); span_generator emits de paired forms.
      * In BASE mode these feed stores['span_scorer'] (M2 gate).  In RERANK mode the
        M4 reranker (see reranker.py) supersedes span_scorer and scores ALL candidates.

Everything remains learned-from-train at runtime (no literal encoded content strings).
Toggle exact_first_hook with env M4_EXACTFIRST=0 (default 1) for ablation.
"""
import os
import pipeline
import m2_ext
import m3_ext
from transducer import Transducer, _norm as _tnorm


# ---------------------------------------------------------------------------
# STORE BUILDER: stash a leak-free exact/norm-memory transducer for the hook.
# (pipeline fits its own per-fold transducer for transduction; this parallel
#  copy is fit on the identical fold-train frame, so its memory is identical.)
# ---------------------------------------------------------------------------
def stash_transducer(train_df, stores):
    if "_transducer" not in stores:
        stores["_transducer"] = Transducer().fit(train_df)


# ---------------------------------------------------------------------------
# (c) exact-memory-FIRST replacement hook.  Returns the verified train memory
#     for this exact span (or its normalized form) so it wins over the M2/M3
#     heuristic collapse/rewrite hooks that follow.  Defers (None) otherwise.
# ---------------------------------------------------------------------------
def exact_first_hook(lang, src, context, stores):
    if os.environ.get("M4_EXACTFIRST", "1") != "1":
        return None
    T = stores.get("_transducer")
    if T is None:
        return None
    k = (lang, src)
    if k in T.exact:
        return T.exact[k]
    nk = (lang, _tnorm(src))
    if nk in T.norm:
        return T.norm[nk]
    return None


# ---------------------------------------------------------------------------
# Registration: compose both specialists onto the pipeline module object.
# ---------------------------------------------------------------------------
def register(P=pipeline):
    P.STORE_BUILDERS = [stash_transducer, m2_ext.build_stores, m3_ext.build_stores]
    P.TOKEN_FEATURE_EXTRAS = [m3_ext.it_en_feats]              # m2 token_extras OFF
    P.REPLACEMENT_HOOKS = [exact_first_hook, m2_ext.collapse_hook,
                           m3_ext.it_slash_hook, m3_ext.en_norm_hook, m3_ext.del_hook]
    P.SPAN_CANDIDATE_GENERATORS = [m2_ext.span_generator, m3_ext.np_generator]
    P.FEAT_NAMES = None
    P.EXTRA_NAMES = None
    return P
