
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

LUX_PER_WM2 = 120.0          
LUX_SATURATION = 100_000  

PROBE_INVALID = (-127.0, -999.0)   

FIT_MIN_G = 200.0   

CONFORMAL_ALPHA = 0.10 
CONFORMAL_MIN_G = 50.0 


SC_V_RESID_PCT = -80.0          
SC_I_FRACTION_OF_ISC = 0.70     
OC_I_FRACTION = 0.05            
OC_V_FRACTION_OF_VOC = 0.90     
DUST_SENSOR_V = 0.25            
DUST_V_RESID_MIN_PCT = -10.0    
SHADE_V_RESID_PCT = -5.0        
SHADE_TEMP_SPREAD_C = 1.0       

def load_datasheet() -> dict:
    
    return pd.read_csv(DATA / "panel_datasheet.csv").iloc[0].to_dict()


def load_calibration(path: Path = OUT / "twin_calibration.json") -> dict:
    
    return json.loads(Path(path).read_text())