#!/bin/bash
# Coordinate sweep around the current baseline. Logs one line per config.
cd ~/swadesh
PY=~/venv/bin/python
LOG=sweep_results.log
: > $LOG
run () {  # $1 = tag, rest = args
  tag="$1"; shift
  line=$($PY cv_parallel.py --procs 16 --tag "$tag" --n_iter 16 "$@" 2>&1 | grep "mean_all")
  echo "$tag | args: $* | $line" | tee -a $LOG
}
B="--freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 1.0 --damp 0.5"
run baseline           $B
run fp0.3   --freq_prior 0.3 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 1.0 --damp 0.5
run fp0.7   --freq_prior 0.7 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 1.0 --damp 0.5
run fp1.0   --freq_prior 1.0 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 1.0 --damp 0.5
run sml1    --freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 1 --rel_pow 1.0 --damp 0.5
run sml3    --freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 3 --rel_pow 1.0 --damp 0.5
run sml4    --freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 4 --rel_pow 1.0 --damp 0.5
run gap4    --freq_prior 0.5 --gap -4 --align_scale 0.5 --seg_min_langs 2 --rel_pow 1.0 --damp 0.5
run gap9    --freq_prior 0.5 --gap -9 --align_scale 0.5 --seg_min_langs 2 --rel_pow 1.0 --damp 0.5
run as0.3   --freq_prior 0.5 --gap -6 --align_scale 0.3 --seg_min_langs 2 --rel_pow 1.0 --damp 0.5
run as0.7   --freq_prior 0.5 --gap -6 --align_scale 0.7 --seg_min_langs 2 --rel_pow 1.0 --damp 0.5
run rp0.5   --freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 0.5 --damp 0.5
run rp2.0   --freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 2.0 --damp 0.5
run rp0.0   --freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 0.0 --damp 0.5
run damp0.3 --freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 1.0 --damp 0.3
run damp0.7 --freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 1.0 --damp 0.7
run lsp1    --freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 1.0 --damp 0.5 --lensim_pow 1.0
run lsp3    --freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 1.0 --damp 0.5 --lensim_pow 3.0
run pmik1   --freq_prior 0.5 --gap -6 --align_scale 0.5 --seg_min_langs 2 --rel_pow 1.0 --damp 0.5 --pmi_k 1.0
run combo1  --freq_prior 0.7 --gap -6 --align_scale 0.5 --seg_min_langs 3 --rel_pow 1.0 --damp 0.5
run combo2  --freq_prior 0.7 --gap -9 --align_scale 0.6 --seg_min_langs 3 --rel_pow 1.5 --damp 0.5
run combo3  --freq_prior 0.4 --gap -6 --align_scale 0.4 --seg_min_langs 3 --rel_pow 1.5 --damp 0.4
echo "DONE" | tee -a $LOG
