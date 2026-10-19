/* Sensor adapter: calibrated channels -> eight model inputs (see pv_sensor.h). */
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include "pv_sensor.h"

static float pv_take(const pv_channel_t *ch, pv_channel_state_t *state)
{
    if (ch->state != PV_CH_VALID) {
        *state = ch->state;
        return NAN;
    }
    if (!isfinite(ch->value)) {
        *state = PV_CH_SENSOR_ERROR;
        return NAN;
    }
    *state = PV_CH_VALID;
    return ch->value;
}

int pv_sensor_adapt(const pv_sensor_sample_t *in, pv_adapted_t *out)
{
    int i;
    if (in == NULL || out == NULL) return -1;
    out->raw[PV_RAW_VOLTAGE_V] = pv_take(&in->voltage_v, &out->state[PV_RAW_VOLTAGE_V]);
    out->raw[PV_RAW_CURRENT_MA] = pv_take(&in->current_ma, &out->state[PV_RAW_CURRENT_MA]);
    out->raw[PV_RAW_PANEL_TEMP1_C] = pv_take(&in->panel_temp1_c, &out->state[PV_RAW_PANEL_TEMP1_C]);
    out->raw[PV_RAW_PANEL_TEMP2_C] = pv_take(&in->panel_temp2_c, &out->state[PV_RAW_PANEL_TEMP2_C]);
    out->raw[PV_RAW_AMBIENT_TEMP_C] = pv_take(&in->ambient_temp_c, &out->state[PV_RAW_AMBIENT_TEMP_C]);
    out->raw[PV_RAW_IRRADIANCE_LUX] = pv_take(&in->irradiance_lux, &out->state[PV_RAW_IRRADIANCE_LUX]);
    out->raw[PV_RAW_DUST_VO_V] = pv_take(&in->dust_vo_v, &out->state[PV_RAW_DUST_VO_V]);

    /* power_mw = voltage_v * current_ma (V x mA = mW), from the same sample. */
    if (out->state[PV_RAW_VOLTAGE_V] == PV_CH_VALID && out->state[PV_RAW_CURRENT_MA] == PV_CH_VALID) {
        out->raw[PV_RAW_POWER_MW] = out->raw[PV_RAW_VOLTAGE_V] * out->raw[PV_RAW_CURRENT_MA];
        out->state[PV_RAW_POWER_MW] = isfinite(out->raw[PV_RAW_POWER_MW]) ? PV_CH_VALID : PV_CH_SENSOR_ERROR;
        if (out->state[PV_RAW_POWER_MW] != PV_CH_VALID) out->raw[PV_RAW_POWER_MW] = NAN;
    } else {
        out->raw[PV_RAW_POWER_MW] = NAN;
        out->state[PV_RAW_POWER_MW] = out->state[PV_RAW_VOLTAGE_V] != PV_CH_VALID ? out->state[PV_RAW_VOLTAGE_V]
                                                                                : out->state[PV_RAW_CURRENT_MA];
    }

    out->invalid_mask = 0;
    for (i = 0; i < PV_N_RAW; i++)
        if (isnan(out->raw[i])) out->invalid_mask |= (uint16_t)(1u << i);
    out->timestamp_unix_ms = in->timestamp_unix_ms;
    out->timestamp_valid = in->timestamp_valid && in->timestamp_unix_ms >= PV_MIN_VALID_UNIX_MS;
    out->sequence = in->sequence;
    if (out->invalid_mask) out->adapter_status = PV_ADAPTER_INCOMPLETE;
    else if (!out->timestamp_valid) out->adapter_status = PV_ADAPTER_BAD_TIMESTAMP;
    else out->adapter_status = PV_ADAPTER_COMPLETE;
    return 0;
}

pv_channel_state_t pv_ds18b20_state(float value_c, int first_conversion)
{
    if (!isfinite(value_c) || value_c == -127.0f) return PV_CH_SENSOR_ERROR;
    if (first_conversion && value_c == 85.0f) return PV_CH_SENSOR_ERROR;
    return PV_CH_VALID;
}

/* Days since 1970-01-01 -> civil date (proleptic Gregorian; H. Hinnant). */
static void pv_civil_from_days(int64_t z, int *y, unsigned *m, unsigned *d)
{
    int64_t era, yy;
    unsigned doe, yoe, doy, mp;
    z += 719468;
    era = (z >= 0 ? z : z - 146096) / 146097;
    doe = (unsigned)(z - era * 146097);
    yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    yy = (int64_t)yoe + era * 400;
    doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    mp = (5 * doy + 2) / 153;
    *d = doy - (153 * mp + 2) / 5 + 1;
    *m = mp < 10 ? mp + 3 : mp - 9;
    *y = (int)(yy + (*m <= 2));
}

int pv_format_utc(int64_t unix_ms, char *buf, unsigned buf_len)
{
    int64_t secs, days, rem;
    int y;
    unsigned mo, d;
    if (buf == NULL || buf_len < 21 || unix_ms < PV_MIN_VALID_UNIX_MS) return -1;
    secs = unix_ms / 1000;
    days = secs / 86400;
    rem = secs % 86400;
    pv_civil_from_days(days, &y, &mo, &d);
    if (y > 9999) return -1;
    snprintf(buf, buf_len, "%04d-%02u-%02uT%02d:%02d:%02dZ", y, mo, d, (int)(rem / 3600), (int)(rem % 3600 / 60),
             (int)(rem % 60));
    return 0;
}

const char *pv_channel_state_name(pv_channel_state_t s)
{
    switch (s) {
    case PV_CH_VALID: return "VALID";
    case PV_CH_MISSING: return "MISSING";
    case PV_CH_NOT_INSTALLED: return "NOT_INSTALLED";
    case PV_CH_SENSOR_ERROR: return "SENSOR_ERROR";
    case PV_CH_SATURATED: return "SATURATED";
    case PV_CH_UNCALIBRATED: return "UNCALIBRATED";
    }
    return "UNKNOWN";
}
