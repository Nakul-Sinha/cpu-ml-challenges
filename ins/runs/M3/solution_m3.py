"""M3 deliverable: register M3 plug-ins onto the M1 pipeline and run its full
main() (leak-free per-fold CV + full-train fit -> submission + OOF).  Self-contained:
paths auto-detect via pipeline.ROOT.  Nothing in core pipeline.py is modified."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P
import m3_ext as M

# ship config: features + replacement hooks ON; deletion forfeited; npgen inert-off
M.USE_FEATS = M.USE_IT_REPL = M.USE_EN_REPL = True
M.USE_DEL = M.USE_NPGEN = False
M.register(P)

if __name__ == "__main__":
    P.main()
