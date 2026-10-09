
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

LUX_PER_WM2 = 120.0          
LUX_SATURATION = 100_000     

PROBE_INVALID = (-127.0, -999.0)   

FIT_MIN_G = 200.0 