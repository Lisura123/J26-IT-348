"""Firmware runtime modules on the host: public API, sensor adapter, fault
confirmation, mock relay, shadow pipeline (the C message checks against the
Python pv-msg-1 validator are in test_messaging.py), on-device
self-test and benchmark cores, and the two ESP-IDF applications built
against host stubs. Host checks only: nothing here ran on an ESP32."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tinyml import benchmark
from tinyml.contract import LABELS
from tinyml.export import file_sha256
from tinyml.hostc import (APPS, FIRMWARE, RUNTIME_SOURCES, SOURCES, compile_and_run_app, compile_and_run_runtime,
                          compiler)
from tinyml.quantize import QuantParams
from tinyml.selfcheck import selfcheck_header

ROOT = Path(__file__).resolve().parents[1]
needs_cc = pytest.mark.skipif(compiler() is None, reason="no C compiler")


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    if compiler() is None:
        pytest.skip("no C compiler")
    return compile_and_run_runtime(tmp_path_factory.mktemp("rt"))


@needs_cc
def test_runtime_c_tests_pass(runtime):
    assert runtime["passed"], runtime["output"][-3000:]
    assert "-fsanitize=undefined,address" in " ".join(runtime["flags"])


@needs_cc
def test_on_device_selftest_suite_passes_on_host(runtime):
    line = next(l for l in runtime["output"].splitlines() if l.startswith("PVTEST "))
    r = json.loads(line[len("PVTEST "):])
    assert r["result"] == "PASS" and r["platform"] == "host-gcc"
    assert r["f32_label_ok"] == r["f32_label_total"] == 105 and r["int8_exact_ok"] == r["int8_total"] == 105
    assert r["gate_ok"] == r["gate_total"] and r["max_logit_abs_diff"] < 1e-4


@needs_cc
def test_host_benchmark_lines_are_never_esp32_results(runtime):
    s = benchmark.summarize(runtime["output"])
    assert s["measurement_status"] == "HOST_ONLY" and s["backends"] == ["FLOAT32", "INT8"]


@needs_cc
@pytest.mark.parametrize("app", ["pv_selftest_app", "pv_benchmark_app"])
def test_esp_applications_build_and_run_against_host_stubs(tmp_path, app):
    r = compile_and_run_app(app, tmp_path)
    assert r["ran"] and r["passed"], r["output"][-3000:]
    out = r["output"]
    assert '"target":"host-stub"' in out and '"result":"PASS"' in out
    if app == "pv_benchmark_app":
        s = benchmark.summarize(out)
        assert s["measurement_status"] == "HOST_ONLY" and s["selftest_pass"]
        assert "PVBENCH_DONE" in out
    else:
        assert "PVTEST_DONE PASS" in out


@needs_cc
def test_physical_relay_build_switch_stops_the_build(tmp_path):
    r = subprocess.run([compiler(), "-std=c99", "-DPV_RELAY_ENABLE_PHYSICAL", f"-I{FIRMWARE}",
                        f"-I{FIRMWARE / 'generated'}", "-c", str(FIRMWARE / "pv_relay.c"), "-o", str(tmp_path / "x.o")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "Physical relay actuation is disabled" in r.stderr


def test_no_gpio_or_actuation_code_in_firmware():
    pattern = re.compile(r"gpio_|GPIO_|digitalWrite|ledc_|rtc_gpio|esp_restart|esp_ota_", re.I)
    files = [p for p in (ROOT / "firmware").rglob("*") if p.suffix in (".c", ".h")]
    assert files
    hits = [str(p) for p in files if pattern.search(p.read_text(encoding="utf-8"))]
    assert hits == []


@pytest.mark.skipif(shutil.which("cmake") is None or compiler() is None, reason="no cmake")
def test_component_builds_as_a_host_cmake_library(tmp_path):
    b = tmp_path / "build"
    cfg = subprocess.run(["cmake", "-S", str(FIRMWARE), "-B", str(b)], capture_output=True, text=True)
    assert cfg.returncode == 0, cfg.stderr
    build = subprocess.run(["cmake", "--build", str(b)], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    assert (b / "libpv_fault.a").exists()


def test_esp_idf_build_files_are_consistent():
    cm = (FIRMWARE / "CMakeLists.txt").read_text()
    listed = re.findall(r"(pv_\w+\.c)", cm)
    assert sorted(listed) == sorted(SOURCES + RUNTIME_SOURCES)
    assert "idf_component_register" in cm and "-ffp-contract=off" in cm
    kconfig = (FIRMWARE / "Kconfig").read_text()
    for app in ("pv_selftest_app", "pv_benchmark_app"):
        main = (APPS / app / "main" / "main.c").read_text()
        for opt in re.findall(r"CONFIG_PV_FAULT_(\w+)", main):
            assert re.search(rf"config PV_FAULT_{opt}\b", kconfig), opt
        top = (APPS / app / "CMakeLists.txt").read_text()
        assert "project.cmake" in top and "pv_fault_test" in top
        defaults = (APPS / app / "sdkconfig.defaults").read_text()
        assert "CONFIG_IDF_TARGET" not in defaults                       # no board is assumed


def test_selfcheck_header_is_reproducible_from_frozen_artifacts():
    contract = json.loads((ROOT / "models" / "model_contract.json").read_text())
    q = QuantParams.from_json((ROOT / "models" / "int8" / "quant_params.json").read_text())
    text = selfcheck_header(contract, file_sha256(ROOT / "models" / "model_contract.json"), q)
    assert (FIRMWARE / "selfcheck" / "pv_selfcheck.h").read_text() == text
    labels = re.search(r"PV_SC_LABEL_INT8\[16\] = \{([^}]*)\}", text).group(1)
    assert sorted(set(int(v) for v in labels.split(","))) == list(range(len(LABELS)))   # every class covered


def test_existing_generated_headers_and_golden_vectors_unchanged():
    manifest = json.loads((ROOT / "models" / "int8" / "manifest.json").read_text())
    assert all(file_sha256(ROOT / k) == v for k, v in manifest["files"].items())
    assert len(manifest["files"]) == 5
