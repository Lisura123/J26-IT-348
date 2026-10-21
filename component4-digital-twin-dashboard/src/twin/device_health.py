import math

FLAGS = {
    "OK": dict(sensor="All sensors", severity="OK",
               what="Every sensor reported a valid value.",
               check="No action needed.",
               twin="Twin uses this reading normally."),
    "CURRENT_SIGN_REVERSED": dict(sensor="INA219 (current)", severity="WARNING",
               what="The current sensor reported a negative current, so current flows the wrong way through the shunt.",
               check="Check the INA219 wiring: VIN+ must go to the panel side and VIN- to the load side. Swap the two wires if they are reversed.",
               twin="Power and current residuals are not reliable for this reading."),
    "LOAD_SHORTED": dict(sensor="Load / wiring", severity="CRITICAL",
               what="Voltage collapsed to almost zero while current is high: the load or the wiring is shorted.",
               check="Disconnect the panel, inspect the load resistor, terminal block and cables for a short, then reconnect.",
               twin="Twin will classify this as SHORT_CIRCUIT."),
    "TEMP1_SENTINEL_-127": dict(sensor="DS18B20 panel probe", severity="WARNING",
               what="The panel temperature probe returned -127 °C, the DS18B20 code for 'device not found on the 1-Wire bus'.",
               check="Check the probe's data wire, the 4.7 kΩ pull-up resistor between DATA and 3.3 V, and the connector.",
               twin="Panel temperature residual is unavailable; shading cannot be thermally confirmed."),
    "TEMP1_MISSING_-999": dict(sensor="DS18B20 panel probe", severity="WARNING",
               what="No panel temperature was recorded for this reading (firmware placeholder -999).",
               check="Check the firmware log for a 1-Wire read timeout; if it repeats, reseat the probe.",
               twin="Panel temperature residual is unavailable for this reading."),
    "AMBIENT_TEMP_SENTINEL_-127": dict(sensor="DS18B20 air probe", severity="WARNING",
               what="The air temperature probe returned -127 °C (not found on the 1-Wire bus).",
               check="Check the air probe's wiring and pull-up resistor; both probes share the same bus.",
               twin="Faiman model has no ambient temperature, so expected panel temperature and power are less accurate."),
    "AMBIENT_TEMP_MISSING_-999": dict(sensor="DS18B20 air probe", severity="WARNING",
               what="No air temperature was recorded for this reading (firmware placeholder -999).",
               check="Check the firmware log for a 1-Wire read timeout on the air probe.",
               twin="Faiman model has no ambient temperature for this reading."),
    "SPIKE_VOLTAGE": dict(sensor="INA219 (voltage)", severity="INFO",
               what="A sudden one-off jump in voltage that does not match the readings around it.",
               check="Ignore if isolated. If spikes repeat, check for loose connections or electrical noise near the INA219.",
               twin="Single reading may fall outside the band; not treated as a fault on its own."),
    "SPIKE_CURRENT": dict(sensor="INA219 (current)", severity="INFO",
               what="A sudden one-off jump in current that does not match the readings around it.",
               check="Ignore if isolated. If spikes repeat, check the shunt connections and cable routing.",
               twin="Single reading may fall outside the band; not treated as a fault on its own."),
    "SPIKE_IRRADIANCE": dict(sensor="TEMT6000 light sensor", severity="INFO",
               what="A sudden one-off jump in light level (for example a reflection or a passing object).",
               check="Ignore if isolated. If it repeats, check the light sensor's mounting and shading.",
               twin="Expected power for this reading may be wrong."),
    "DUPLICATE_RECEIVED": dict(sensor="Network", severity="INFO",
               what="The same reading arrived twice (the device re-sent it after a missed acknowledgement).",
               check="No action unless frequent; then check WiFi signal strength at the site.",
               twin="Duplicate is ignored by the backend upsert."),
    "LOST_IN_TRANSIT": dict(sensor="Network", severity="WARNING",
               what="A reading was lost between the device and the server.",
               check="Check WiFi / LoRa signal strength and gateway uptime at the site.",
               twin="Gap in the timeline; no twin output for the missing reading."),
    "DUST_BASELINE_DRIFT": dict(sensor="Dust sensor", severity="WARNING",
               what="The dust sensor's clean-air baseline has drifted, so dust readings are biased.",
               check="Clean the dust sensor's optical window and re-run its baseline calibration.",
               twin="Dust rule may raise false DUST alerts until recalibrated."),
    "TEST_MODE": dict(sensor="Device firmware", severity="INFO",
               what="The device was running in test mode (bench test, not real operation).",
               check="Make sure test mode is switched off before field deployment.",
               twin="Test-mode readings should not be used for fault detection."),
}

STATUS = {
    "HEALTHY": "All checks passed.",
    "DEGRADED": "The device is working but some readings are unreliable.",
    "FAULTY": "The device or panel has a fault that needs attention.",
    "NOT_GRADED": "Not graded: usually no sunlight (night) or the device is in test mode.",
}
RANK = {"OK": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3}


def split_flags(flag) -> list[str]:
    if flag is None or (isinstance(flag, float) and math.isnan(flag)) or str(flag).strip() == "":
        return []
    return [f.strip() for f in str(flag).split("|") if f.strip()]


def explain_flag(code: str) -> dict:
    info = FLAGS.get(code)
    if info is None:
        info = dict(sensor="Unknown", severity="WARNING", what=f"Unrecognised quality flag '{code}'.",
                    check="Ask the sensing team (Member 1) what this flag means.", twin="Unknown impact.")
    return {"code": code, **info}


def explain_reading(doc: dict) -> dict:
    issues = [explain_flag(c) for c in split_flags(doc.get("quality_flag")) if c != "OK"]
    lux, power = doc.get("irradiance_lux"), doc.get("power_mw")
    dark = isinstance(lux, (int, float)) and lux < 50 and (power or 0) < 1
    status = doc.get("health_status") or ("NOT_GRADED" if dark else "UNKNOWN")
    worst = max((i["severity"] for i in issues), key=RANK.get, default="OK")
    note = "No sunlight (night or heavy cloud): zero output is expected." if dark else ""
    return {"health_status": status,
            "status_text": STATUS.get(status, "No health status was sent with this reading."),
            "worst_severity": worst, "issues": issues, "context": note}