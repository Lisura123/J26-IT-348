
import numpy as np
import pandas as pd

from . import config as C


def residuals(df: pd.DataFrame) -> pd.DataFrame:
    def pct(actual, expected):
        return 100 * (actual - expected) / expected.where(expected.abs() > 1e-9)
    return pd.DataFrame({
        "voltage_residual_pct": pct(df.voltage_v, df.expected_voltage_v),
        "current_residual_pct": pct(df.current_ma, df.expected_current_ma),
        "power_residual_pct": pct(df.power_mw, df.expected_power_mw),
        "panel_temp_residual_c": df.panel_temp_mean_c - df.expected_panel_temp_c,
    }, index=df.index)


def classify(df: pd.DataFrame) -> pd.Series:
    inside = df.within_expected_band
    short = ((df.voltage_residual_pct <= C.SC_V_RESID_PCT)
             & (df.current_ma >= C.SC_I_FRACTION_OF_ISC * df.expected_isc_ma))
    open_ = ((df.current_ma <= C.OC_I_FRACTION * df.expected_current_ma)
             & (df.voltage_v >= C.OC_V_FRACTION_OF_VOC * df.expected_voc_v))
    dust = ((df.dust_vo_v >= C.DUST_SENSOR_V)
            & (df.voltage_residual_pct > C.DUST_V_RESID_MIN_PCT)
            & (df.power_residual_pct < 0))
    shade = ((df.voltage_residual_pct <= C.SHADE_V_RESID_PCT)
             & (df.panel_temp_spread_c >= C.SHADE_TEMP_SPREAD_C)
             & (df.power_residual_pct < 0))
    cause = np.select([inside, short, open_, dust, shade],
                      ["NORMAL", "SHORT_CIRCUIT", "OPEN_CIRCUIT", "DUST", "PARTIAL_SHADING"],
                      default="UNEXPLAINED_LOSS")
    return pd.Series(cause, index=df.index)


def explain(r) -> str:
    head = (f"Actual {r.power_mw:.0f} mW vs twin expected {r.expected_power_mw:.0f} mW "
            f"(90% band {r.power_band_low_mw:.0f}-{r.power_band_high_mw:.0f} mW) "
            f"at ~{r.irradiance_wm2_est:.0f} W/m²; ")
    why = {
        "NORMAL": "output is inside the expected band.",
        "SHORT_CIRCUIT": f"voltage collapsed to {r.voltage_v:.2f} V while current rose to "
                         f"{r.current_ma:.0f} mA (near short-circuit current).",
        "OPEN_CIRCUIT": f"current is ~0 mA while voltage sits at open-circuit level ({r.voltage_v:.2f} V).",
        "DUST": f"dust sensor reads {r.dust_vo_v:.2f} V and current is "
                f"{abs(r.current_residual_pct):.0f}% below expected with voltage near normal.",
        "PARTIAL_SHADING": f"voltage is {abs(r.voltage_residual_pct):.0f}% below expected and panel probes "
                           f"differ by {r.panel_temp_spread_c:.1f} °C, consistent with a shaded-cell hotspot.",
        "UNEXPLAINED_LOSS": "output is outside the expected band but matches no known fault signature.",
    }[r.physics_cause]
    verdict = "agrees" if r.ml_physics_agree else f"disagrees with physics ({r.physics_cause}); flag for review"
    ml = f" TinyML {r.model_version}: {r.predicted_fault} ({r.confidence:.0%}) {verdict}."
    if str(r.model_up_to_date) == "False":
        ml += f" Site model is out of date (latest {r.latest_published_version})."
    return head + why + ml