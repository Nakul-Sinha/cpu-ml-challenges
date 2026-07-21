"""M4 ship entry point: composes M2+M3 plug-ins onto the M1 base, selects leak-free
operating points, applies the group-consistency vote (de+en), fits full-train, writes
submission_v2.csv + OOF + cv_report.json.  Reranker machinery lives in run_m4.py
(built + verified; measured marginal on honest nested CV, so NOT in the ship config).

Run:  cd ~/insled && OMP_NUM_THREADS=5 nice -n 10 ~/venv/bin/python runs/M4/solution_m4.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_m4

if __name__ == "__main__":
    run_m4.ship()
