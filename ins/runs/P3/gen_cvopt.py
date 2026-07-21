"""Generate submission_cvopt.csv (de_thr=DE_THR_CVOPT=0.19) reusing solution.py's own
ship functions so the alternate is identical-pipeline except the de threshold."""
import os, sys, importlib.util
P3 = os.path.expanduser("~/insled/runs/P3")
spec = importlib.util.spec_from_file_location("solution", os.path.join(P3, "solution.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

train, test = S.load_frames(os.path.expanduser("~/insled/dataset"))
art = S.ship_artifacts(train, test)
sub = S.assemble_submission(art, de_thr=S.DE_THR_CVOPT)
S.validate_submission(sub, test)
out = os.path.join(P3, "submission_cvopt.csv")
S.write_submission(sub, test, out)
S.verify_written(out, test)
rates = S._edit_rates(sub, test)
print("submission_cvopt.csv (de_thr=%.2f):" % S.DE_THR_CVOPT)
for L in ("de", "en", "it"):
    ed, tot, frac, ratio, flag = rates[L]
    print(f"  {L}: {ed}/{tot} frac={frac} ratio={ratio}" + ("  <<FLAG" if flag else ""))
print("edited total:", sum(1 for i in sub if sub[i]), "/", len(sub))
print("wrote", out)
