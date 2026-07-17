"""
Swadesh Phoneme Cipher Decoding -- official solution.

Reads ./dataset/public/{train,test}.csv, recovers the global token->IPA-segment
bijection for the hidden Uralic target by cognate-alignment decipherment against the
true-IPA Uralic relatives, and writes ./working/submission.csv.

Unsupervised: no target labels are used; the map is derived only from the provided
files. See notes.md / approach.md for the method.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import pandas as pd
from decipher import Decipherer, forms_by_lang_concept, load_train

# Configuration locked from leave-one-Uralic-out cross-validation (see sweeps / notes.md).
CONFIG = dict(
    n_iter=18, gap=-6.0, pmi_k=0.5, beta=1.5, tau=8.0, lensim_pow=2.0,
    rel_pow=3.0, align_scale=0.5, seg_min_langs=2, aff_keep=0.15,
    damp=0.5, freq_prior=0.7, cog_floor=0.5, vc_weight=1.0,
)


def main():
    public_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/public"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "working/submission.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tr = load_train(os.path.join(public_dir, "train.csv"))
    te = pd.read_csv(os.path.join(public_dir, "test.csv"))

    # Crib = all Uralic relatives (the target belongs to the Uralic family).
    crib = forms_by_lang_concept(tr[tr.family == "Uralic"])

    target_words = [(c, str(cip).split()) for c, cip in zip(te.concept.values, te.cipher.values)]

    dec = Decipherer(**CONFIG).fit(target_words, crib)

    preds = [" ".join(dec.decode(str(cip).split())) for cip in te.cipher.values]
    out = pd.DataFrame({"id": te["id"].values, "ipa": preds})
    # never leave a row blank (blank ~ 0 similarity); fall back to raw tokens if any empty
    out["ipa"] = out["ipa"].where(out["ipa"].str.strip().astype(bool), te["cipher"].values)
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows -> {out_path}")
    print("recovered inventory sample:", dict(list(dec.sigma.items())[:12]))


if __name__ == "__main__":
    main()
