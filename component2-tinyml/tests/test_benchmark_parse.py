"""Benchmark log parser: what a console log may count as. The ESP32-like log
below is a SYNTHETIC parser fixture with made-up numbers, not a measurement."""
import json

from tinyml import benchmark


def stage(m):
    return {"n": 1000, "mean_us": m, "median_us": m, "p95_us": m * 1.1, "p99_us": m * 1.2, "min_us": m * 0.9,
            "max_us": m * 1.5}


def bench_line(platform, backend, **kw):
    b = {"schema": "pv-bench-1", "platform": platform, "backend": backend, "model_version": "c2-local-v1",
         "iterations": 1000, **{s: stage(1.0) for s in benchmark.STAGES}, **kw}
    return "PVBENCH " + json.dumps(b)


def selftest_line(platform, result="PASS", failures=0):
    return "PVTEST " + json.dumps({"schema": "pv-selftest-1", "platform": platform, "result": result,
                                   "failures": failures})


def test_no_log_means_not_executed():
    s = benchmark.summarize("")
    assert s["measurement_status"] == "NOT_EXECUTED" and "no performance result" in s["note"]


def test_host_lines_are_host_only():
    log = "\n".join([selftest_line("host-stub"), bench_line("host-stub", "FLOAT32"), bench_line("host-stub", "INT8")])
    assert benchmark.summarize(log)["measurement_status"] == "HOST_ONLY"


def test_synthetic_target_log_is_classified_as_measured_only_when_complete():
    good = [selftest_line("esp32s3"), bench_line("esp32s3", "FLOAT32"), bench_line("esp32s3", "INT8")]
    s = benchmark.summarize("noise\n" + "\n".join(good), size_report={"flash_code": 1})
    assert s["measurement_status"] == "ESP32_MEASURED" and s["size_report"] == {"flash_code": 1}
    assert benchmark.summarize("\n".join(good[1:]))["measurement_status"] == "INVALID"          # no self-test
    assert benchmark.summarize("\n".join([selftest_line("esp32s3", "FAIL", 3)] + good[1:]))[
        "measurement_status"] == "INVALID"
    assert benchmark.summarize("\n".join(good[:2]))["measurement_status"] == "INVALID"         # one backend only
    broken = bench_line("esp32s3", "INT8", total={**stage(1.0), "p95_us": 0.1})
    assert benchmark.summarize("\n".join(good[:2] + [broken]))["measurement_status"] == "INVALID"


def test_size_report_is_not_invented():
    s = benchmark.summarize("\n".join([selftest_line("esp32"), bench_line("esp32", "FLOAT32"),
                                       bench_line("esp32", "INT8")]))
    assert s["size_report"].startswith("NOT_PROVIDED")
