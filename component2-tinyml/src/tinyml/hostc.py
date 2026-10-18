"""Compile and run the host C tests (gcc) for the portable firmware."""
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIRMWARE = ROOT / "firmware" / "pv_fault"
HOST_TEST = ROOT / "tests" / "host_c" / "test_main.c"
SOURCES = ("pv_fault.c", "pv_mlp_f32.c", "pv_mlp_int8.c")
# runtime modules (public API, sensor adapter, confirmation, mock relay, messages)
RUNTIME_SOURCES = ("pv_api.c", "pv_sensor.c", "pv_confirm.c", "pv_relay.c", "pv_message.c", "pv_pipeline.c")
FIRMWARE_TEST = ROOT / "firmware" / "pv_fault_test"
TEST_SUITE_SOURCES = ("pv_selftest.c", "pv_bench.c")
SELFCHECK_DIR = FIRMWARE / "selfcheck"
RUNTIME_TEST = ROOT / "tests" / "host_c" / "test_runtime.c"
GOLDEN_DIR = ROOT / "tests" / "host_c"
GEN_DIR = FIRMWARE / "generated"
CFLAGS = ["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O1", "-ffp-contract=off"]
SANITIZE = ["-fsanitize=undefined,address", "-fno-sanitize-recover=all"]


def compiler():
    return shutil.which("gcc") or shutil.which("cc")


def compile_and_run(gen_dir, golden_dir=None, out_dir=None, sanitize=True, test_source=HOST_TEST):
    """Build firmware + test_main.c against generated headers and run it.
    Returns {"ran", "passed", "compiler", "flags", "output"}."""
    cc = compiler()
    if cc is None:
        return {"ran": False, "passed": False, "compiler": None, "flags": None,
                "output": "no C compiler found"}
    gen_dir = Path(gen_dir)
    out = Path(out_dir or gen_dir)
    exe = out / "pv_host_test"
    flags = CFLAGS + (SANITIZE if sanitize else [])
    # A quoted #include is searched first in the including file's own folder, so the
    # test source is compiled from an empty temporary folder: golden_vectors.h then
    # comes only from golden_dir, never from a copy that sits next to test_main.c.
    with tempfile.TemporaryDirectory() as build_dir:
        src = Path(build_dir) / Path(test_source).name
        shutil.copyfile(test_source, src)
        cmd = [cc, *flags, f"-I{gen_dir}", f"-I{golden_dir or gen_dir}", f"-I{FIRMWARE}",
               *[str(FIRMWARE / s) for s in SOURCES], str(src), "-lm", "-o", str(exe)]
        build = subprocess.run(cmd, capture_output=True, text=True)
    version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    if build.returncode != 0:
        return {"ran": False, "passed": False, "compiler": version, "flags": flags,
                "output": "compile failed:\n" + build.stderr[-4000:]}
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    text = (run.stdout + run.stderr)[-4000:]
    return {"ran": True, "passed": run.returncode == 0 and "ALL PASSED" in run.stdout,
            "compiler": version, "flags": flags, "output": text.strip()}


def compile_and_run_runtime(out_dir, gen_dir=GEN_DIR, golden_dir=GOLDEN_DIR, sanitize=True, test_source=RUNTIME_TEST,
                            extra_flags=()):
    """Build the runtime modules, the self-test and benchmark cores and
    test_runtime.c (host gcc) and run it. Same result keys as compile_and_run."""
    cc = compiler()
    if cc is None:
        return {"ran": False, "passed": False, "compiler": None, "flags": None, "output": "no C compiler found"}
    out = Path(out_dir)
    exe = out / "pv_runtime_test"
    flags = CFLAGS + (SANITIZE if sanitize else []) + list(extra_flags)
    with tempfile.TemporaryDirectory() as build_dir:
        src = Path(build_dir) / Path(test_source).name
        shutil.copyfile(test_source, src)
        cmd = [cc, *flags, f"-I{gen_dir}", f"-I{golden_dir}", f"-I{FIRMWARE}", f"-I{SELFCHECK_DIR}",
               f"-I{FIRMWARE_TEST}", *[str(FIRMWARE / s) for s in SOURCES + RUNTIME_SOURCES],
               *[str(FIRMWARE_TEST / s) for s in TEST_SUITE_SOURCES], str(src), "-lm", "-o", str(exe)]
        build = subprocess.run(cmd, capture_output=True, text=True)
    version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    if build.returncode != 0:
        return {"ran": False, "passed": False, "compiler": version, "flags": flags,
                "output": "compile failed:\n" + build.stderr[-4000:]}
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    text = run.stdout + run.stderr
    return {"ran": True, "passed": run.returncode == 0 and "RUNTIME ALL PASSED" in run.stdout,
            "compiler": version, "flags": flags, "output": text}


ESP_STUBS = ROOT / "tests" / "esp_stubs"
APPS = ROOT / "firmware" / "apps"


def compile_and_run_app(app, out_dir, sanitize=True):
    """Compile firmware/apps/<app>/main/main.c against the host stubs of the
    ESP-IDF APIs (tests/esp_stubs) and run it. Checks the application code on
    the host only; the output is never an ESP32 result."""
    cc = compiler()
    if cc is None:
        return {"ran": False, "passed": False, "output": "no C compiler found"}
    exe = Path(out_dir) / f"{app}_host"
    flags = CFLAGS + (SANITIZE if sanitize else [])
    cmd = [cc, *flags, f"-I{ESP_STUBS}", f"-I{GEN_DIR}", f"-I{GOLDEN_DIR}", f"-I{FIRMWARE}", f"-I{SELFCHECK_DIR}",
           f"-I{FIRMWARE_TEST}", *[str(FIRMWARE / s) for s in SOURCES + RUNTIME_SOURCES],
           *[str(FIRMWARE_TEST / s) for s in TEST_SUITE_SOURCES], str(ESP_STUBS / "esp_stubs.c"),
           str(APPS / app / "main" / "main.c"), "-lm", "-o", str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    if build.returncode != 0:
        return {"ran": False, "passed": False, "output": "compile failed:\n" + build.stderr[-4000:]}
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    return {"ran": True, "passed": run.returncode == 0, "output": run.stdout + run.stderr}
