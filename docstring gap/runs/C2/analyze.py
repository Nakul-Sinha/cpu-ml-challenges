"""Consume C2 probe/FT artifacts -> projection (30/40 min) + grader budget table."""
import sys, os, json, glob
import numpy as np, pandas as pd
HERE = os.path.join("runs", "C2")

# ---- learning curves ----
curves = {}
for p in sorted(glob.glob(os.path.join(HERE, "curve_*.csv"))):
    df = pd.read_csv(p)
    curves[df.tag.iloc[0]] = df
    print(f"\n== curve {df.tag.iloc[0]} (lr={df.lr.iloc[0]} mode={df['mode'].iloc[0]} sps={df.samples_per_s.iloc[0]:.1f}) ==")
    for _, r in df.iterrows():
        print(f"   t={r.train_min:5.1f}min seen={int(r.seen):6d} chrF={r.chrf:.4f}")

# ---- project main curve to 30/40 min ----
def fit_project(df, targets=(30, 40)):
    t = df.train_min.values.astype(float)
    y = df.chrf.values.astype(float)
    out = {}
    # saturating exponential: y = a - b*exp(-t/tau)
    try:
        from scipy.optimize import curve_fit
        def sat(t, a, b, tau):
            return a - b * np.exp(-t / tau)
        p0 = [max(y) + 0.15, max(y) - y[0], 12.0]
        popt, _ = curve_fit(sat, t, y, p0=p0, maxfev=20000,
                            bounds=([y[-1], 0, 1], [0.9, 1.0, 200]))
        for T in targets:
            out[f"sat_{T}min"] = float(sat(T, *popt))
        out["sat_asymptote"] = float(popt[0])
        out["sat_params"] = [float(x) for x in popt]
    except Exception as e:
        out["sat_err"] = repr(e)
    # log fit: y = a + b*ln(t)  (fit on t>0)
    m = t > 0
    if m.sum() >= 2:
        b, a = np.polyfit(np.log(t[m]), y[m], 1)[0], None
        coef = np.polyfit(np.log(t[m]), y[m], 1)
        for T in targets:
            out[f"log_{T}min"] = float(coef[0] * np.log(T) + coef[1])
    return out

if "main" in curves:
    proj = fit_project(curves["main"])
    print("\n== PROJECTION (main curve) ==")
    for k, v in proj.items():
        print(f"   {k}: {v}")

# ---- speed ----
sp = json.load(open(os.path.join(HERE, "speed_results.json")))
print("\n== SPEED (box, 7 threads, code_first inputs, 1024 rows) ==")
for prec in ["fp32", "int8"]:
    for bs, d in sp[prec].items():
        print(f"   {prec} bs={bs}: {d['rows_per_s']:.1f} rows/s  50k->{d['min_50k']:.1f} min")
print(f"   fp32 chrF={sp['fp32_chrf']:.4f}  int8 chrF={sp['int8_chrf']:.4f}  (int8 delta {sp['int8_chrf']-sp['fp32_chrf']:+.4f})")

# ---- grader budget table ----
# grader: 10 cores @ ~0.8x per-core vs box 7 threads. Net throughput factor (central est).
GF = 0.9          # grader throughput / box-7thread throughput (central); range ~0.85-1.0
LOAD_MIN = 2.0    # t5-small download+load on grader
best_inf_rps = sp["int8"]["64"]["rows_per_s"]  # int8 bs64 fastest
inf_50k_box = 50000 / best_inf_rps / 60
inf_50k_grader = inf_50k_box / GF
sps_main = curves["main"].samples_per_s.iloc[0] if "main" in curves else 11.6
sps_grader = sps_main * GF
print("\n== GRADER BUDGET (90 min, GF=%.2f, load=%.1f min) ==" % (GF, LOAD_MIN))
print(f"   int8 bs64 inference 50k: box {inf_50k_box:.1f} min -> grader {inf_50k_grader:.1f} min")
print(f"   FT throughput: box {sps_main:.1f} sps -> grader {sps_grader:.1f} sps")
budget_ft = 90 - LOAD_MIN - inf_50k_grader
ft_samples = budget_ft * 60 * sps_grader
print(f"   remaining for FT: {budget_ft:.1f} min -> ~{ft_samples/1000:.1f}k samples")
print(f"   RECOMMEND: FT {budget_ft:.0f} min ({ft_samples/1000:.0f}k samples), int8 bs64 inference {inf_50k_grader:.0f} min, load {LOAD_MIN:.0f} min")
