"""Parse and summarise ESP32 test / benchmark console logs.

The board applications (firmware/apps/pv_selftest_app, pv_benchmark_app)
print PVINFO, PVTEST and PVBENCH JSON lines on the serial console. This
module turns a saved log into a report and decides what the log can count
as:

  NOT_EXECUTED          no benchmark lines (the default: no board has run it)
  HOST_ONLY             lines come from a host build (platform "host-...");
                        code check only, never an ESP32 result
  INVALID               self-test missing or failed, or lines inconsistent
  ESP32_MEASURED        an ESP-IDF target name, self-test PASS, both backends

Flash and static RAM are not in the log; they come from `idf.py size`
(passed in as the parsed JSON or text and stored verbatim).
"""
import json

PREFIXES = ("PVINFO ", "PVTEST ", "PVBENCH ")
STAGES = ("preprocessing", "neural", "post", "total")
STAT_KEYS = ("n", "mean_us", "median_us", "p95_us", "p99_us", "min_us", "max_us")
NOT_EXECUTED = {"measurement_status": "NOT_EXECUTED",
                "note": "No ESP32 board has run the benchmark. Flash, RAM, stack, heap and latency are not "
                        "measured; no performance result is claimed."}


def parse_log(text):
    out = {"info": [], "selftest": [], "bench": [], "errors": []}
    for line in text.splitlines():
        line = line.strip()
        for key, prefix in (("info", "PVINFO "), ("selftest", "PVTEST "), ("bench", "PVBENCH ")):
            if line.startswith(prefix):
                try:
                    out[key].append(json.loads(line[len(prefix):]))
                except ValueError:
                    out["errors"].append(f"bad JSON in {prefix.strip()} line")
        if line.startswith("PVBENCH_ERROR") or line.startswith("PVTEST_FAIL"):
            out["errors"].append(line)
    return out


def _check_bench(b):
    errs = []
    if b.get("schema") != "pv-bench-1":
        errs.append("bench schema")
    for s in STAGES:
        st = b.get(s, {})
        if any(k not in st for k in STAT_KEYS):
            errs.append(f"{s} stats incomplete")
            continue
        if not (st["min_us"] <= st["median_us"] <= st["p95_us"] <= st["p99_us"] <= st["max_us"]):
            errs.append(f"{s} percentiles not ordered")
    return errs


def summarize(text, size_report=None):
    p = parse_log(text)
    if not p["bench"]:
        return dict(NOT_EXECUTED, parsed_errors=p["errors"])
    platforms = {b.get("platform", "") for b in p["bench"]} | {t.get("platform", "") for t in p["selftest"]}
    errors = list(p["errors"])
    for b in p["bench"]:
        errors += _check_bench(b)
    selftest_ok = bool(p["selftest"]) and all(t.get("result") == "PASS" and t.get("failures") == 0
                                              for t in p["selftest"])
    backends = sorted({b.get("backend") for b in p["bench"]})
    if any(pl.startswith("host") or not pl for pl in platforms):
        status = "HOST_ONLY"
    elif not selftest_ok or errors or len(platforms) != 1 or backends != ["FLOAT32", "INT8"]:
        status = "INVALID"
    else:
        status = "ESP32_MEASURED"
    return {"measurement_status": status, "platforms": sorted(platforms), "selftest_pass": selftest_ok,
            "backends": backends, "info": p["info"], "selftest": p["selftest"],
            "results": {b["backend"]: {s: b[s] for s in STAGES} | {k: b[k] for k in b if k not in STAGES}
                        for b in p["bench"]},
            "size_report": size_report if size_report is not None else "NOT_PROVIDED (run idf.py size)",
            "errors": errors,
            "note": ("Counts as an ESP32 measurement only with status ESP32_MEASURED, a saved console log, the "
                     "ESP-IDF version and the board model recorded.")}
