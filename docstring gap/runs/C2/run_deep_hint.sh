#!/bin/bash
# Deep doc_first (code-context) FT with checkpoint, then integration dump + union-oracle.
cd ~/docgap
PY=~/venv/bin/python
echo "=== DEEP DOC_FIRST START $(date) ==="
df -h / | tail -1
# free disk: drop the weaker plain ckpt (doc_first is strictly better for integration)
rm -rf runs/C2/ft_main
nice -n 10 $PY runs/C2/t5_finetune.py --lr 3e-4 --mode doc_first --budget 600 --eval_every 300 --tag hintdeep --save_ckpt runs/C2/ft_hint
echo "=== deep hint done -> DUMP $(date) ==="
nice -n 10 $PY -c "import sys; sys.path.insert(0,'runs/C2'); from t5_probe import run_dump; run_dump(500, ckpt='runs/C2/ft_hint', mode='doc_first')"
echo "=== dump done -> UNION ORACLE $(date) ==="
nice -n 10 $PY runs/C2/union_oracle.py
echo "=== ALL_DEEP_DONE $(date) ==="
