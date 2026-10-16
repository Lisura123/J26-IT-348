"""Compile and run the host C tests (gcc) for the portable firmware."""
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIRMWARE = ROOT / "firmware" / "pv_fault"
HOST_TEST = ROOT / "tests" / "host_c" / "test_main.c"
SOURCES = ("pv_fault.c", "pv_mlp_f32.c", "pv_mlp_int8.c")
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
