"""Generator: assemble the self-contained runs/P3/solution.py.

Embeds the VERBATIM sources of the 8 base modules (as readable r'''...''' blocks,
exec'd into synthetic modules in dependency order) + a scorer-free elru shim
(validate_edits only) + the P3 ship runtime (_runtime.py) + a robust main wrapper
(argv/autodetect, wall-clock guard, strict pre/post validation, all-empty fallback).

Run ON THE BOX so solution.py embeds the exact box module code:
    ~/venv/bin/python ~/insled/runs/P3/build_solution.py
"""
import os

RUNS = os.path.expanduser("~/insled/runs")
OUT = os.path.expanduser("~/insled/runs/P3/solution.py")
RUNTIME_SRC = os.path.expanduser("~/insled/runs/P3/_runtime.py")

# module name -> source file (dependency order preserved in ORDER below)
MODFILES = {
    "transducer": os.path.join(RUNS, "N3", "transducer_p2.py"),
    "pipeline":   os.path.join(RUNS, "M4", "pipeline.py"),
    "m2_ext":     os.path.join(RUNS, "M4", "m2_ext.py"),
    "m3_ext":     os.path.join(RUNS, "M4", "m3_ext.py"),
    "m4_ext":     os.path.join(RUNS, "M4", "m4_ext.py"),
    "n2_ext":     os.path.join(RUNS, "N2", "n2_ext.py"),
    "run_m4":     os.path.join(RUNS, "M4", "run_m4.py"),
    "run_n1":     os.path.join(RUNS, "N1", "run_n1.py"),
}
ORDER = ["elru", "transducer", "pipeline", "m2_ext", "m3_ext", "m4_ext", "n2_ext", "run_m4", "run_n1"]

# scorer-free elru shim: ONLY validate_edits (the submission validity check from the
# spec).  No ELRU scoring logic -> solution.py is scorer-free.
ELRU_SHIM = '''"""Scorer-free elru shim: submission validity check only (validate_edits).
No ELRU scoring in the shipped solution (the grader scores independently)."""


def validate_edits(edits, text_len):
    if not isinstance(edits, list) or len(edits) > 8:
        return False
    prev_end = -1
    for e in edits:
        if set(e.keys()) != {"start", "end", "replacement"}:
            return False
        s, en, rep = e["start"], e["end"], e["replacement"]
        if not (isinstance(s, int) and isinstance(en, int) and isinstance(rep, str)):
            return False
        if not (0 <= s < en <= text_len):
            return False
        if len(rep) > 160:
            return False
        if s < prev_end:
            return False
        prev_end = en
    return True
'''

HEADER = '''#!/usr/bin/env python3
"""Institutional Edit Ledger Recovery -- SHIP solution (P3 v4, self-contained).

ONE self-contained file: no imports from runs/, scorer-free.  Fits every model on
train.csv AT RUNTIME and emits a validated edit-ledger submission for test.csv.

Pipeline (honest nested CV 0.5709; see runs/P3/cv_report_v4.json):
  * A1 LightGBM per-token edit detector (81 feats, learned lexicon)   [pipeline]
  * A2 transducer w/ IT multi-token decomp + append rules (P2)        [transducer]
  * German paired-form collapse + marked-run generator (M2/N2)        [m2_ext/n2_ext]
  * Italian NP-gate assembly + slash reorder (N1) + IT LGBM re-scorer boost (P1)
  * German BiGRU detector ensemble (P1 lever 2, a=0.6)                [runtime]
  * de/en group-consistency vote (hi.60/lo.40)                        [run_m4]

Usage:   python3 solution.py [public_dir] [submission_out]
  public_dir      dir containing train.csv + test.csv (autodetected if omitted)
  submission_out  output CSV (default: working/submission.csv)

Everything learned from train.csv at runtime; no literal encoded content strings;
no tfidf; a real LightGBM + BiGRU materially drive predictions; deterministic.
"""
import os, sys, json, time, types, glob, traceback

# ---- threads (deterministic; ~10-core grader friendly) ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

_T0 = time.time()
_WALL_GUARD = 3000.0   # safeguard; this pipeline runs in ~1-2 min

'''

BOOT = '''
# ======================================================================
#  Embedded-module bootstrap: exec the verbatim sources into synthetic
#  modules registered in sys.modules, in dependency order, so their normal
#  `import pipeline` / `from transducer import ...` cross-references resolve
#  WITHOUT any imports from runs/.  Behavior is byte-identical to the box code.
# ======================================================================
def _boot_modules():
    for _name in _ORDER:
        _m = types.ModuleType(_name)
        _m.__file__ = _name + ".py"
        sys.modules[_name] = _m
        exec(compile(_MODS[_name], _name + ".py", "exec"), _m.__dict__)


_boot_modules()

'''

MAIN = r'''
# ======================================================================
#  Path autodetect + IO
# ======================================================================
def _has_data(d):
    return d and os.path.isfile(os.path.join(d, "train.csv")) and os.path.isfile(os.path.join(d, "test.csv"))


def find_data_dir(arg=None):
    cands = []
    if arg:
        cands += [arg, os.path.join(arg, "public"), os.path.join(arg, "dataset"),
                  os.path.join(arg, "dataset", "public")]
    cands += [os.path.join("dataset", "public"), "dataset", ".",
              os.path.join("..", "dataset", "public"), os.path.join("..", "dataset"),
              os.path.expanduser("~/insled/dataset")]
    cands += glob.glob("/kaggle/input/*") + ["/kaggle/input"]
    for d in cands:
        if _has_data(d):
            return os.path.abspath(d)
    # last resort: walk cwd (and a couple of parents) for a dir with both csvs
    for base in (".", "..", os.path.expanduser("~")):
        for root, _dirs, files in os.walk(base):
            if "train.csv" in files and "test.csv" in files:
                return os.path.abspath(root)
            if root.count(os.sep) - base.count(os.sep) > 4:
                _dirs[:] = []
    raise FileNotFoundError("could not locate a directory containing train.csv + test.csv")


def load_frames(data_dir):
    import pandas as pd
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), keep_default_na=False)
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), keep_default_na=False)
    train["edits"] = train.edits_json.apply(json.loads)
    return train, test


# ======================================================================
#  Strict validation
# ======================================================================
def validate_submission(sub, test):
    import elru
    assert len(sub) == len(test), f"row count {len(sub)} != {len(test)}"
    assert set(sub) == set(test.id), "id set mismatch vs test"
    tl = {r.id: len(r.text) for r in test.itertuples()}
    for i in sub:
        e = sub[i]
        assert elru.validate_edits(e, tl[i]), f"invalid edits row {i}: {e}"
        # belt-and-suspenders explicit checks
        assert isinstance(e, list) and len(e) <= 8, f"len>8 row {i}"
        pe = -1
        for ed in e:
            assert set(ed) == {"start", "end", "replacement"}, f"keys row {i}"
            assert 0 <= ed["start"] < ed["end"] <= tl[i], f"bounds row {i}"
            assert len(ed["replacement"]) <= 160, f"rep>160 row {i}"
            assert ed["start"] >= pe, f"overlap/unsorted row {i}"
            pe = ed["end"]
    return True


def write_submission(sub, test, out_path):
    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    rows = [{"id": i, "edits_json": json.dumps(sub[i], ensure_ascii=False)} for i in test.id]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def verify_written(out_path, test):
    """Re-read the written file and re-validate (post-write check)."""
    import pandas as pd
    import elru
    df = pd.read_csv(out_path, keep_default_na=False)
    assert len(df) == len(test), f"written row count {len(df)} != {len(test)}"
    assert set(df.id) == set(test.id), "written id mismatch"
    tl = {r.id: len(r.text) for r in test.itertuples()}
    for r in df.itertuples():
        e = json.loads(r.edits_json)
        assert elru.validate_edits(e, tl[r.id]), f"written invalid row {r.id}"
    return True


def _edit_rates(sub, test):
    import collections
    lang = {r.id: r.language for r in test.itertuples()}
    ed = collections.Counter(); tot = collections.Counter()
    for i in sub:
        tot[lang[i]] += 1
        if sub[i]:
            ed[lang[i]] += 1
    tr = {"de": 0.577, "en": 0.470, "it": 0.704}
    out = {}
    for L in ("de", "en", "it"):
        frac = ed[L] / max(tot[L], 1); ratio = frac / tr[L]
        out[L] = (ed[L], tot[L], round(frac, 3), round(ratio, 2), not (0.45 <= ratio <= 1.80))
    return out


def write_empty(test, out_path):
    """Fallback: a valid all-empty-ledger submission (never crashes the grader)."""
    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    pd.DataFrame([{"id": i, "edits_json": "[]"} for i in test.id]).to_csv(out_path, index=False)


def main():
    arg_pub = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("working", "submission.csv")
    data_dir = find_data_dir(arg_pub)
    print(f"[data] {data_dir}", flush=True)
    train, test = load_frames(data_dir)
    print(f"[load] train={len(train)} test={len(test)}", flush=True)

    sub, _test_rows = build_submission(train, test, de_thr=DE_THR)

    validate_submission(sub, test)                  # pre-write strict validation
    rates = _edit_rates(sub, test)
    print("[edit-rates] " + "  ".join(
        f"{L}={rates[L][0]}/{rates[L][1]} frac={rates[L][2]} ratio={rates[L][3]}"
        + ("  <<FLAG" if rates[L][4] else "") for L in ("de", "en", "it")), flush=True)

    write_submission(sub, test, out_path)
    verify_written(out_path, test)                  # post-write re-validation
    elapsed = time.time() - _T0
    assert elapsed < _WALL_GUARD, f"wall-clock {elapsed:.0f}s exceeded guard {_WALL_GUARD:.0f}s"
    n_ed = sum(1 for i in sub if sub[i])
    print(f"[done] wrote {out_path}  ({n_ed}/{len(sub)} edited)  [{elapsed:.0f}s]", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as _exc:
        # LOUD fallback: emit a valid all-empty submission only if the main path threw.
        sys.stderr.write("\n!!!!!! SOLUTION MAIN PATH FAILED -- writing empty fallback !!!!!!\n")
        traceback.print_exc()
        try:
            _out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("working", "submission.csv")
            import pandas as pd
            _dd = None
            try:
                _dd = find_data_dir(sys.argv[1] if len(sys.argv) > 1 else None)
            except Exception:
                pass
            if _dd is not None:
                _test = pd.read_csv(os.path.join(_dd, "test.csv"), keep_default_na=False)
                write_empty(_test, _out)
                sys.stderr.write(f"[fallback] wrote all-empty submission to {_out} ({len(_test)} rows)\n")
            else:
                sys.stderr.write("[fallback] could not locate test.csv; NO submission written\n")
        except Exception:
            traceback.print_exc()
        sys.exit(1)
'''


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def emit_block(name, src):
    # guard: r'''...''' delimiter safety
    assert "'''" not in src, f"module {name} contains ''' -- unsafe to embed"
    # a raw string may not end in a backslash; ensure a trailing newline sits before '''
    if not src.endswith("\n"):
        src = src + "\n"
    return f"_MODS[{name!r}] = r'''\n{src}'''\n\n"


def main():
    parts = [HEADER]
    parts.append("# ======================================================================\n")
    parts.append("#  Embedded verbatim module sources (readable; grep-able)\n")
    parts.append("# ======================================================================\n")
    parts.append(f"_ORDER = {ORDER!r}\n_MODS = {{}}\n\n")
    parts.append(emit_block("elru", ELRU_SHIM))
    for name in ORDER:
        if name == "elru":
            continue
        parts.append(emit_block(name, read(MODFILES[name])))
    parts.append(BOOT)
    parts.append("\n# ======================================================================\n")
    parts.append("#  P3 ship runtime\n")
    parts.append("# ======================================================================\n")
    parts.append(read(RUNTIME_SRC))
    parts.append(MAIN)
    out = "".join(parts)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {OUT}  ({out.count(chr(10))} lines, {len(out)} bytes)")


if __name__ == "__main__":
    main()
