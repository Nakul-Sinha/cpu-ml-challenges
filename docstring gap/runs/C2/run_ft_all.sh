#!/bin/bash
# C2 fine-tune sweep: deep plain (projection anchor + ckpt), doc_first hint compare, lr sweep.
cd ~/docgap
PY=~/venv/bin/python
echo "=== FT SWEEP START $(date) ==="
nice -n 10 $PY runs/C2/t5_finetune.py --lr 3e-4 --mode plain     --budget 1500 --eval_every 300 --tag main --save_ckpt runs/C2/ft_main
echo "=== main done $(date) ==="
nice -n 10 $PY runs/C2/t5_finetune.py --lr 3e-4 --mode doc_first --budget 900  --eval_every 300 --tag hint
echo "=== hint done $(date) ==="
nice -n 10 $PY runs/C2/t5_finetune.py --lr 1e-4 --mode plain     --budget 600  --eval_every 300 --tag lr1e4
echo "=== ALL_FT_DONE $(date) ==="
