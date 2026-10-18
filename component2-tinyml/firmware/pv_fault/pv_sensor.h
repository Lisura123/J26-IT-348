/* Sensor-adapter contract (Component 2 side).
 *
 * Member 1's drivers convert raw sensor values into CALIBRATED physical
 * units and fill one pv_sensor_sample_t per reading, with a state for every
 * channel. pv_sensor_adapt() turns it into the eight model inputs:
 *
 *   raw[0] voltage_v       V    INA219 bus voltage
 *   raw[1] current_ma      mA   INA219 current
 *   raw[2] power_mw        mW   COMPUTED here: voltage_v * current_ma
 *   raw[3] panel_temp1_c   degC panel DS18B20 #1
 *   raw[4] panel_temp2_c   degC panel DS18B20 #2 (NOT INSTALLED on the stated hardware)
 *   raw[5] ambient_temp_c  degC air DS18B20
 *   raw[6] irradiance_lux  lx   TEMT6000 (lux conversion UNRESOLVED)
 *   raw[7] dust_vo_v       V    GP2Y10 (definition UNRESOLVED)
 *
 * Rules: a channel that is not PV_CH_VALID, or not finite, becomes NaN, so
 * the model's quality gate reports INVALID_INPUT (a skipped reading, never a
 * fault). Nothing is filled, copied or estimated: in particular a missing
 * panel_temp2_c is NOT replaced by panel_temp1_c. Range checks stay in the
 * model's quality gate (pv_input_status); the adapter does not change them.
 * Portable C99. */
#ifndef PV_SENSOR_H
#define PV_SENSOR_H

#include <stdint.h>

#include "pv_fault.h"

typedef enum {
    PV_CH_VALID = 0,         /* calibrated value in physical units          */
    PV_CH_MISSING = 1,       /* no sample this cycle (timeout, bus error)   */
    PV_CH_NOT_INSTALLED = 2, /* sensor not fitted on this device            */
    PV_CH_SENSOR_ERROR = 3,  /* driver reported an error code               */
    PV_CH_SATURATED = 4,     /* ADC at full scale / sensor range exceeded    */
    PV_CH_UNCALIBRATED = 5   /* conversion to physical units not available  */
} pv_channel_state_t;

typedef struct {
    float value;
    pv_channel_state_t state;
} pv_channel_t;

/* One reading from Member 1's drivers. power_mw is not an input: it is
 * computed from voltage and current of the same INA219 sample. */
typedef struct {
    pv_channel_t voltage_v, current_ma, panel_temp1_c, panel_temp2_c, ambient_temp_c, irradiance_lux, dust_vo_v;
    int64_t timestamp_unix_ms;   /* UTC milliseconds since 1970-01-01          */
    int timestamp_valid;         /* 1 only if the clock is synchronised (NTP)  */
    uint32_t sequence;           /* reading counter from the sampling loop     */
} pv_sensor_sample_t;

/* Adapter status (not a fault status). */
enum {
    PV_ADAPTER_COMPLETE = 0,     /* all eight inputs present                    */
    PV_ADAPTER_INCOMPLETE = 1,   /* at least one input NaN -> INVALID_INPUT      */
    PV_ADAPTER_BAD_TIMESTAMP = 2 /* inputs complete but time not usable          */
};

/* Earliest accepted timestamp (2020-01-01T00:00:00Z): an unsynchronised RTC
 * usually starts at 1970. */
#define PV_MIN_VALID_UNIX_MS 1577836800000LL

typedef struct {
    float raw[PV_N_RAW];                      /* model inputs, NaN if unavailable  */
    pv_channel_state_t state[PV_N_RAW];        /* per input, power derived from V, I */
    uint16_t invalid_mask;                     /* bit i set: raw[i] is NaN           */
    int adapter_status;                        /* PV_ADAPTER_*                       */
    int64_t timestamp_unix_ms;
    int timestamp_valid;
    uint32_t sequence;
} pv_adapted_t;

/* Returns 0 on success, -1 on NULL arguments. */
int pv_sensor_adapt(const pv_sensor_sample_t *in, pv_adapted_t *out);

/* Helper for DS18B20 drivers: the state of a temperature conversion.
 * -127.0 (DEVICE_DISCONNECTED_C) and non-finite values are sensor errors.
 * 85.0 is the power-on register value: it is a sensor error when it is the
 * first conversion after power-up or reset (first_conversion != 0); the
 * driver must track that flag. */
pv_channel_state_t pv_ds18b20_state(float value_c, int first_conversion);

/* "YYYY-MM-DDTHH:MM:SSZ" (21 bytes incl. NUL). Returns 0, or -1 if the
 * time is before PV_MIN_VALID_UNIX_MS or the buffer is too small. */
int pv_format_utc(int64_t unix_ms, char *buf, unsigned buf_len);

const char *pv_channel_state_name(pv_channel_state_t s);

#endif /* PV_SENSOR_H */
