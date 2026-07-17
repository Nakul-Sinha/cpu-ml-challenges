#!/bin/bash
# Sweep 2 (new code: relatedness-weighted prior). Test the key hypothesis: concentrating
# on the closest relatives (higher rel_pow) recovers Saami targets better, because close
# relatives share the target's sound changes (identity correspondences).
cd ~/swadesh
PY=~/venv/bin/python
LOG=sweep2_results.log
: > $LOG
run () { tag="$1"; shift
  line=$($PY cv_parallel.py --procs 16 --tag "$tag" --n_iter 18 --gap -6 --align_scale 0.5 \
        --seg_min_langs 2 --damp 0.5 "$@" 2>&1 | grep "mean_all")
  echo "$tag | $* | $line" | tee -a $LOG
}
run rp1_fp0.7  --rel_pow 1.0 --freq_prior 0.7
run rp2_fp0.7  --rel_pow 2.0 --freq_prior 0.7
run rp3_fp0.7  --rel_pow 3.0 --freq_prior 0.7
run rp5_fp0.7  --rel_pow 5.0 --freq_prior 0.7
run rp2_fp1.0  --rel_pow 2.0 --freq_prior 1.0
run rp3_fp1.0  --rel_pow 3.0 --freq_prior 1.0
run rp4_fp1.0  --rel_pow 4.0 --freq_prior 1.0
run rp2_fp0.85 --rel_pow 2.0 --freq_prior 0.85
run rp3_fp0.85 --rel_pow 3.0 --freq_prior 0.85
run rp2_fp1.3  --rel_pow 2.0 --freq_prior 1.3
run rp3_fp1.0_g9 --rel_pow 3.0 --freq_prior 1.0 --gap -9
run rp3_fp1.0_as0.65 --rel_pow 3.0 --freq_prior 1.0 --align_scale 0.65
echo "DONE" | tee -a $LOG
