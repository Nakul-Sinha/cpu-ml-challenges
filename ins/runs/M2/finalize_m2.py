"""Finalize M2 (German specialist, gen-only): register plug-ins, run full pipeline,
write deliverables (oof_edits.csv, oof_token_probs.csv, submission_v2.csv, cv_report.json)
into runs/M2/. Also emit de per-type recall vs the loss map."""
import os, sys, json, collections
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["M2_GEN"] = "1"; os.environ["M2_FEAT"] = "0"
import pipeline as P
import m2_ext
m2_ext.register(P)          # gen-only: store builder + span gen + collapse hook
P.main()                    # writes M2 deliverables into HERE and prints CV report
