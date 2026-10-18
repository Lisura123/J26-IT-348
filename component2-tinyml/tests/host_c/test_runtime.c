/* Host test of the runtime modules: pv_api, pv_sensor, pv_confirm, pv_relay,
 * pv_message, plus the on-device self-test and benchmark cores. Synthetic
 * inputs and the committed golden vectors only. Prints "PVMSG <json>" lines
 * for the Python message-schema tests, "PVTEST"/"PVBENCH" lines from the
 * suites, one line per failure, and "RUNTIME ALL PASSED <n>" on success. */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "golden_vectors.h"
#include "pv_api.h"
#include "pv_bench.h"
#include "pv_confirm.h"
#include "pv_message.h"
#include "pv_pipeline.h"
#include "pv_relay.h"
#include "pv_selftest.h"
#include "pv_sensor.h"

static int failures = 0, checks = 0;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static const float *ok_raw(int nth)
{
    int i, k = 0;
    for (i = 0; i < GV_N_RAW; i++)
        if (GV_RAW[i].status == PV_STATUS_OK && GV_RAW[i].exact_label && k++ == nth) return GV_RAW[i].raw;
    return NULL;
}

/* ---------------------------------------------------------------- API */

static void test_api(void)
{
    pv_prediction_t p;
    const float *raw = ok_raw(0);
    const pv_model_info_t *m = pv_api_model_info();
    CHECK(pv_api_classify(PV_BACKEND_INT8, raw, &p) == PV_ERR_NOT_INITIALIZED, "classify before init must fail");
    CHECK(p.label == -1 && p.label_name == NULL, "no label before init");
    CHECK(pv_api_init() == PV_OK, "init failures 0x%x", pv_api_last_check_failures());
    CHECK(pv_api_init() == PV_OK && pv_api_is_initialized(), "init idempotent");
    CHECK(strcmp(m->model_version, "c2-local-v1") == 0 && strcmp(m->feature_version, "pv8-v1") == 0 &&
          strcmp(m->quant_version, "int8-v1") == 0 && m->n_params == 229, "model info");
    CHECK(pv_api_classify(PV_BACKEND_INT8, NULL, &p) == PV_ERR_NULL_ARG, "NULL raw");
    CHECK(pv_api_classify(PV_BACKEND_INT8, raw, NULL) == PV_ERR_NULL_ARG, "NULL out");
    CHECK(pv_api_classify((pv_backend_t)7, raw, &p) == PV_ERR_BAD_BACKEND, "bad backend");
    CHECK(strcmp(pv_api_strerror(PV_ERR_MODEL_CHECK), "MODEL_CHECK_FAILED") == 0, "strerror");
    {
        int i;
        for (i = 0; i < GV_N_RAW; i++) {
            const gv_raw_t *g = &GV_RAW[i];
            pv_prediction_t pf, p8;
            CHECK(pv_api_classify(PV_BACKEND_FLOAT32, g->raw, &pf) == PV_OK, "f32 rc %d", i);
            CHECK(pv_api_classify(PV_BACKEND_INT8, g->raw, &p8) == PV_OK, "int8 rc %d", i);
            CHECK(pf.input_status == g->status && p8.input_status == g->status, "%s status", g->id);
            if (g->status != PV_STATUS_OK) {
                CHECK(pf.label == -1 && p8.label == -1 && pf.confidence == 0.0f, "%s skipped has no label", g->id);
            } else if (g->exact_label) {
                CHECK(pf.label == g->label_f32 && p8.label == g->label_int8, "%s labels", g->id);
                CHECK(strcmp(p8.label_name, pv_label_names[p8.label]) == 0, "%s label name", g->id);
                CHECK(pf.backend == PV_BACKEND_FLOAT32 && p8.backend == PV_BACKEND_INT8, "%s backend", g->id);
                CHECK(pf.input_clips == 0 && pf.lut_clips == 0, "%s f32 has no clip counts", g->id);
            }
        }
    }
    CHECK(strcmp(pv_prediction_status_name(PV_STATUS_LOW_LIGHT), "SKIPPED_LOW_LIGHT") == 0, "status names");
}

/* ---------------------------------------------------------------- sensor adapter */

static void fill_sample(pv_sensor_sample_t *s, const float *raw)
{
    memset(s, 0, sizeof(*s));
    s->voltage_v.value = raw[PV_RAW_VOLTAGE_V];
    s->current_ma.value = raw[PV_RAW_CURRENT_MA];
    s->panel_temp1_c.value = raw[PV_RAW_PANEL_TEMP1_C];
    s->panel_temp2_c.value = raw[PV_RAW_PANEL_TEMP2_C];
    s->ambient_temp_c.value = raw[PV_RAW_AMBIENT_TEMP_C];
    s->irradiance_lux.value = raw[PV_RAW_IRRADIANCE_LUX];
    s->dust_vo_v.value = raw[PV_RAW_DUST_VO_V];
    s->timestamp_unix_ms = 1789000000000LL;   /* 2026-09-10 */
    s->timestamp_valid = 1;
    s->sequence = 7;
}

static void test_sensor(void)
{
    pv_sensor_sample_t s;
    pv_adapted_t a;
    pv_prediction_t p;
    const float *raw = ok_raw(0);
    char ts[24];
    int i;
    fill_sample(&s, raw);
    CHECK(pv_sensor_adapt(&s, &a) == 0 && a.adapter_status == PV_ADAPTER_COMPLETE && a.invalid_mask == 0,
          "complete sample");
    CHECK(a.raw[PV_RAW_POWER_MW] == raw[PV_RAW_VOLTAGE_V] * raw[PV_RAW_CURRENT_MA], "power = V * mA");
    for (i = 0; i < PV_N_RAW; i++)
        if (i != PV_RAW_POWER_MW) CHECK(a.raw[i] == raw[i], "channel %d copied unchanged", i);

    s.panel_temp2_c.state = PV_CH_NOT_INSTALLED;        /* stated hardware: no second panel probe */
    pv_sensor_adapt(&s, &a);
    CHECK(isnan(a.raw[PV_RAW_PANEL_TEMP2_C]) && a.raw[PV_RAW_PANEL_TEMP1_C] == raw[PV_RAW_PANEL_TEMP1_C],
          "missing panel_temp2 is NaN, never copied from panel_temp1");
    CHECK(a.state[PV_RAW_PANEL_TEMP2_C] == PV_CH_NOT_INSTALLED && a.adapter_status == PV_ADAPTER_INCOMPLETE &&
          a.invalid_mask == (1u << PV_RAW_PANEL_TEMP2_C), "missing panel_temp2 reported");
    CHECK(pv_api_classify(PV_BACKEND_INT8, a.raw, &p) == PV_OK && p.input_status == PV_STATUS_INVALID_INPUT &&
          p.label == -1, "missing panel_temp2 -> INVALID_INPUT, not a fault");

    fill_sample(&s, raw);
    s.current_ma.state = PV_CH_MISSING;
    pv_sensor_adapt(&s, &a);
    CHECK(isnan(a.raw[PV_RAW_CURRENT_MA]) && isnan(a.raw[PV_RAW_POWER_MW]) &&
          a.state[PV_RAW_POWER_MW] == PV_CH_MISSING, "missing current -> no power");
    CHECK(a.invalid_mask == ((1u << PV_RAW_CURRENT_MA) | (1u << PV_RAW_POWER_MW)), "mask current+power");

    fill_sample(&s, raw);
    s.irradiance_lux.state = PV_CH_SATURATED;
    s.dust_vo_v.value = NAN;
    pv_sensor_adapt(&s, &a);
    CHECK(isnan(a.raw[PV_RAW_IRRADIANCE_LUX]) && a.state[PV_RAW_IRRADIANCE_LUX] == PV_CH_SATURATED,
          "saturated ADC is not converted to a number");
    CHECK(a.state[PV_RAW_DUST_VO_V] == PV_CH_SENSOR_ERROR, "VALID but NaN -> SENSOR_ERROR");

    fill_sample(&s, raw);
    s.timestamp_unix_ms = 5000;                          /* unsynchronised RTC */
    pv_sensor_adapt(&s, &a);
    CHECK(a.adapter_status == PV_ADAPTER_BAD_TIMESTAMP && !a.timestamp_valid, "1970 timestamp rejected");
    s.timestamp_valid = 0;
    s.timestamp_unix_ms = 1789000000000LL;
    pv_sensor_adapt(&s, &a);
    CHECK(!a.timestamp_valid, "unsynchronised flag respected");
    CHECK(pv_sensor_adapt(NULL, &a) == -1 && pv_sensor_adapt(&s, NULL) == -1, "NULL args");

    CHECK(pv_ds18b20_state(-127.0f, 0) == PV_CH_SENSOR_ERROR, "DS18B20 -127");
    CHECK(pv_ds18b20_state(85.0f, 1) == PV_CH_SENSOR_ERROR, "DS18B20 85 on first conversion");
    CHECK(pv_ds18b20_state(85.0f, 0) == PV_CH_VALID, "85 later is a real value");
    CHECK(pv_ds18b20_state(NAN, 0) == PV_CH_SENSOR_ERROR && pv_ds18b20_state(31.5f, 1) == PV_CH_VALID, "DS18B20");

    CHECK(pv_format_utc(1789000000000LL, ts, sizeof(ts)) == 0 && strcmp(ts, "2026-09-10T00:26:40Z") == 0,
          "utc format %s", ts);
    CHECK(pv_format_utc(1709251199000LL, ts, sizeof(ts)) == 0 && strcmp(ts, "2024-02-29T23:59:59Z") == 0,
          "leap day %s", ts);
    CHECK(pv_format_utc(1577836800000LL, ts, sizeof(ts)) == 0 && strcmp(ts, "2020-01-01T00:00:00Z") == 0,
          "minimum %s", ts);
    CHECK(pv_format_utc(1577836799999LL, ts, sizeof(ts)) == -1, "before minimum");
    CHECK(pv_format_utc(1789000000000LL, ts, 20) == -1, "small buffer");
}

/* ---------------------------------------------------------------- confirmation */

static pv_prediction_t pred(int status, int label, float conf)
{
    pv_prediction_t p;
    memset(&p, 0, sizeof(p));
    p.input_status = status;
    p.label = status == PV_STATUS_OK ? label : -1;
    p.confidence = status == PV_STATUS_OK ? conf : 0.0f;
    return p;
}

static pv_confirm_status_t feed(pv_confirm_t *st, int status, int label, float conf, int64_t ts,
                                pv_confirm_output_t *o)
{
    pv_prediction_t p = pred(status, label, conf);
    pv_confirm_update(st, &p, NULL, ts, o);
    return o->status;
}

static void test_confirm(void)
{
    pv_confirm_config_t cfg;
    pv_confirm_t st;
    pv_confirm_output_t o;
    int i;
    int64_t t = 1000000;
    pv_confirm_default_config(&cfg);
    CHECK(cfg.confirm_count == 5 && cfg.recovery_count == 5 && cfg.min_confidence == 0.90f &&
          isnan(cfg.open_max_current_ma) && isnan(cfg.short_max_voltage_v), "conservative defaults");
    CHECK(pv_confirm_init(&st, &cfg) == 0, "init");

    /* ML_PREDICTED -> PENDING x3 -> CONFIRMED on the 5th consecutive reading */
    CHECK(feed(&st, PV_STATUS_OK, 4, 0.99f, t += 5000, &o) == PV_CONF_ML_PREDICTED &&
          o.event == PV_EVENT_CANDIDATE_STARTED, "first SHORT is ML_PREDICTED");
    for (i = 0; i < 3; i++)
        CHECK(feed(&st, PV_STATUS_OK, 4, 0.99f, t += 5000, &o) == PV_CONF_CONFIRMATION_PENDING, "pending %d", i);
    CHECK(feed(&st, PV_STATUS_OK, 4, 0.99f, t += 5000, &o) == PV_CONF_CONFIRMED_BY_CONFIGURED_RULES &&
          o.event == PV_EVENT_CONFIRMED && o.confirmed_label == 4 && o.consecutive == 5, "confirmed");
    CHECK(o.electrical == PV_ELEC_NOT_CONFIGURED, "electrical check disabled by default");

    /* persistence: skipped readings keep a confirmed fault; hysteresis on recovery */
    CHECK(feed(&st, PV_STATUS_LOW_LIGHT, -1, 0, t += 5000, &o) == PV_CONF_CONFIRMED_BY_CONFIGURED_RULES &&
          o.evidence == PV_EVID_SKIPPED_BY_QUALITY_GATE, "skipped keeps confirmed");
    for (i = 0; i < 4; i++)
        CHECK(feed(&st, PV_STATUS_OK, 0, 0.99f, t += 5000, &o) == PV_CONF_CONFIRMED_BY_CONFIGURED_RULES &&
              o.recovery_progress == i + 1, "recovering %d", i);
    CHECK(feed(&st, PV_STATUS_OK, 4, 0.99f, t += 5000, &o) == PV_CONF_CONFIRMED_BY_CONFIGURED_RULES &&
          o.recovery_progress == 0, "fault seen again resets recovery");
    for (i = 0; i < 4; i++) feed(&st, PV_STATUS_OK, 0, 0.99f, t += 5000, &o);
    CHECK(feed(&st, PV_STATUS_OK, 0, 0.99f, t += 5000, &o) == PV_CONF_NO_FAULT && o.event == PV_EVENT_CLEARED &&
          o.confirmed_label == -1, "cleared after 5 normal readings");

    /* false-alarm diagnostics: a short burst is cancelled, never confirmed */
    pv_confirm_init(&st, &cfg);
    feed(&st, PV_STATUS_OK, 2, 0.95f, t += 5000, &o);
    feed(&st, PV_STATUS_OK, 2, 0.95f, t += 5000, &o);
    CHECK(feed(&st, PV_STATUS_OK, 0, 0.95f, t += 5000, &o) == PV_CONF_NO_FAULT &&
          o.event == PV_EVENT_CANDIDATE_CANCELLED && st.diag.candidates_cancelled == 1, "burst cancelled");
    /* label flips restart the sequence */
    feed(&st, PV_STATUS_OK, 1, 0.95f, t += 5000, &o);
    CHECK(feed(&st, PV_STATUS_OK, 2, 0.95f, t += 5000, &o) == PV_CONF_ML_PREDICTED && st.diag.label_flips == 1,
          "flip restarts");
    /* low confidence is not evidence; more gaps than allowed cancel */
    CHECK(feed(&st, PV_STATUS_OK, 2, 0.5f, t += 5000, &o) == PV_CONF_INSUFFICIENT_EVIDENCE &&
          o.evidence == PV_EVID_LOW_CONFIDENCE && o.candidate_label == 2, "one gap tolerated");
    CHECK(feed(&st, PV_STATUS_INVALID_INPUT, -1, 0, t += 5000, &o) == PV_CONF_INSUFFICIENT_EVIDENCE &&
          o.candidate_label == -1 && o.event == PV_EVENT_CANDIDATE_CANCELLED, "second gap cancels");
    /* stale timestamps reset a pending sequence */
    feed(&st, PV_STATUS_OK, 3, 0.95f, t += 5000, &o);
    feed(&st, PV_STATUS_OK, 3, 0.95f, t += 5000, &o);
    CHECK(feed(&st, PV_STATUS_OK, 3, 0.95f, t += 60000, &o) == PV_CONF_ML_PREDICTED && o.evidence == PV_EVID_STALE &&
          o.consecutive == 1, "stale gap restarts");
    CHECK(st.diag.confirmations == 0, "no confirmation from interrupted sequences");

    /* fresh flag and expiry: a confirmed fault needs fresh evidence */
    pv_confirm_init(&st, &cfg);
    for (i = 0; i < 5; i++) feed(&st, PV_STATUS_OK, 3, 0.99f, t += 5000, &o);
    CHECK(o.fresh == 1 && o.reading_seq == st.diag.readings, "confirmation is fresh");
    CHECK(feed(&st, PV_STATUS_INVALID_INPUT, -1, 0, t += 5000, &o) == PV_CONF_CONFIRMED_BY_CONFIGURED_RULES &&
          o.fresh == 0, "one invalid reading tolerated, not fresh");
    CHECK(feed(&st, PV_STATUS_INVALID_INPUT, -1, 0, t += 5000, &o) == PV_CONF_INSUFFICIENT_EVIDENCE &&
          o.event == PV_EVENT_EXPIRED && o.expired_label == 3 && o.confirmed_label == -1 && st.diag.expirations == 1,
          "second invalid reading expires the confirmation");
    for (i = 0; i < 118; i++) feed(&st, PV_STATUS_INVALID_INPUT, -1, 0, t += 5000, &o);
    CHECK(o.status == PV_CONF_INSUFFICIENT_EVIDENCE && o.confirmed_label == -1, "10 min of invalid readings: not confirmed");
    pv_confirm_init(&st, &cfg);
    for (i = 0; i < 5; i++) feed(&st, PV_STATUS_OK, 4, 0.99f, t += 5000, &o);
    CHECK(feed(&st, PV_STATUS_OK, 4, 0.99f, t += 60000, &o) == PV_CONF_ML_PREDICTED && o.event == PV_EVENT_EXPIRED &&
          o.expired_label == 4 && o.candidate_label == 4 && o.fresh == 0, "a long time gap expires it; restart");
    pv_confirm_init(&st, &cfg);
    for (i = 0; i < 5; i++) feed(&st, PV_STATUS_OK, 4, 0.99f, t += 5000, &o);
    CHECK(feed(&st, PV_STATUS_OK, 4, 0.5f, t += 5000, &o) == PV_CONF_CONFIRMED_BY_CONFIGURED_RULES && o.fresh == 0,
          "low confidence is not fresh");
    CHECK(feed(&st, PV_STATUS_OK, 4, 0.99f, t += 5000, &o) == PV_CONF_CONFIRMED_BY_CONFIGURED_RULES && o.fresh == 1,
          "fresh again with usable matching evidence");

    /* transition: a confirmed fault cleared by another fault label starts that candidate */
    pv_confirm_init(&st, &cfg);
    for (i = 0; i < 5; i++) feed(&st, PV_STATUS_OK, 4, 0.99f, t += 5000, &o);
    for (i = 0; i < 4; i++)
        CHECK(feed(&st, PV_STATUS_OK, 2, 0.99f, t += 5000, &o) == PV_CONF_CONFIRMED_BY_CONFIGURED_RULES && o.fresh == 0,
              "DUST while SHORT confirmed %d", i);
    CHECK(feed(&st, PV_STATUS_OK, 2, 0.99f, t += 5000, &o) == PV_CONF_ML_PREDICTED && o.event == PV_EVENT_CLEARED &&
          o.candidate_label == 2 && o.consecutive == 1 && o.confirmed_label == -1, "cleared reading is ML_PREDICTED DUST");
    CHECK(feed(&st, PV_STATUS_OK, 2, 0.99f, t += 5000, &o) == PV_CONF_CONFIRMATION_PENDING && o.consecutive == 2,
          "DUST sequence continues");

    /* optional electrical check (thresholds here are TEST VALUES, not hardware values) */
    cfg.short_max_voltage_v = 0.5f;
    pv_confirm_init(&st, &cfg);
    {
        float raw[PV_N_RAW] = {3.2f, 80.0f, 256.0f, 30.0f, 30.0f, 25.0f, 50000.0f, 0.1f};
        pv_prediction_t p = pred(PV_STATUS_OK, 4, 0.99f);
        pv_confirm_update(&st, &p, raw, -1, &o);
        CHECK(o.status == PV_CONF_INSUFFICIENT_EVIDENCE && o.electrical == PV_ELEC_CONTRADICTED &&
              o.evidence == PV_EVID_ELECTRICAL_CONTRADICTION, "SHORT at 3.2 V contradicted");
        raw[0] = 0.06f;
        pv_confirm_update(&st, &p, raw, -1, &o);
        CHECK(o.status == PV_CONF_ML_PREDICTED && o.electrical == PV_ELEC_CONSISTENT, "SHORT at 0.06 V consistent");
    }

    cfg.confirm_count = 1;
    CHECK(pv_confirm_init(&st, &cfg) == -1, "confirm_count 1 refused");
    pv_confirm_default_config(&cfg);
    cfg.min_confidence = 0.0f;
    CHECK(pv_confirm_init(&st, &cfg) == -1, "confidence 0 refused");
    pv_confirm_default_config(&cfg);
    cfg.open_max_current_ma = -1.0f;
    CHECK(pv_confirm_init(&st, &cfg) == -1, "negative threshold refused");
}

/* ---------------------------------------------------------------- relay (mock only) */

static int mock_calls = 0, mock_fail = 0;
static int mock_isolate(void *ctx) { (void)ctx; mock_calls++; return mock_fail; }
static int mock_reconnect(void *ctx) { (void)ctx; mock_calls++; return mock_fail; }

/* Drives a real confirmation state to a fresh confirmation of `label`. */
static void confirm_fault(pv_confirm_t *st, int label, int64_t *t, pv_confirm_output_t *o)
{
    pv_confirm_config_t cfg;
    int i;
    pv_confirm_default_config(&cfg);
    pv_confirm_init(st, &cfg);
    for (i = 0; i < 5; i++) feed(st, PV_STATUS_OK, label, 0.99f, *t += 5000, o);
}

static void test_relay(void)
{
    pv_relay_t r;
    pv_relay_driver_t physical = {0, mock_isolate, mock_reconnect, NULL};
    pv_relay_driver_t mock = {1, mock_isolate, mock_reconnect, NULL};
    pv_confirm_t st;
    pv_confirm_output_t conf, older;
    pv_relay_log_entry_t log[PV_RELAY_LOG_SIZE];
    int64_t t = 2000000;
    confirm_fault(&st, 3, &t, &conf);                   /* OPEN_CIRCUIT, fresh */
    CHECK(conf.status == PV_CONF_CONFIRMED_BY_CONFIGURED_RULES && conf.fresh == 1, "fresh confirmation");

    CHECK(pv_relay_init(&r, &physical) == PV_RELAY_ERR_PHYSICAL_DISABLED && !r.initialized, "physical refused");
    CHECK(pv_relay_request_isolation(&r, &st, &conf, 1) == PV_RELAY_ERR_NOT_INITIALIZED && mock_calls == 0, "no init");
    CHECK(pv_relay_init(&r, &mock) == PV_RELAY_OK && r.state == PV_RELAY_STATE_CONNECTED, "mock accepted");
    CHECK(pv_relay_request_isolation(&r, &st, &conf, 42) == PV_RELAY_ERR_NOT_AUTHORIZED && mock_calls == 0,
          "confirmed fault alone is not enough");
    pv_relay_authorize(&r, 41);
    CHECK(pv_relay_request_isolation(&r, &st, &conf, 42) == PV_RELAY_ERR_NOT_AUTHORIZED, "wrong token");
    CHECK(pv_relay_request_isolation(&r, &st, &conf, 41) == PV_RELAY_ERR_NOT_AUTHORIZED, "token is one-shot");
    CHECK(pv_relay_authorize(&r, 0) == PV_RELAY_ERR_NOT_AUTHORIZED, "zero token");

    {   /* pending, and an eligible-looking but forged status */
        pv_confirm_t p2;
        pv_confirm_output_t pend;
        int64_t t2 = 0;
        confirm_fault(&p2, 3, &t2, &pend);
        pv_confirm_reset(&p2);
        feed(&p2, PV_STATUS_OK, 3, 0.99f, t2 += 5000, &pend);
        feed(&p2, PV_STATUS_OK, 3, 0.99f, t2 += 5000, &pend);
        pv_relay_authorize(&r, 7);
        CHECK(pend.status == PV_CONF_CONFIRMATION_PENDING &&
              pv_relay_request_isolation(&r, &p2, &pend, 7) == PV_RELAY_ERR_NOT_CONFIRMED && mock_calls == 0, "pending");
        pend.status = PV_CONF_CONFIRMED_BY_CONFIGURED_RULES;   /* forged: the state machine never confirmed it */
        pend.confirmed_label = 3;
        pend.fresh = 1;
        pv_relay_authorize(&r, 7);
        CHECK(pv_relay_request_isolation(&r, &p2, &pend, 7) == PV_RELAY_ERR_STALE_EVIDENCE && mock_calls == 0,
              "forged confirmation refused");
    }
    {   /* DUST never isolates */
        pv_confirm_t d;
        pv_confirm_output_t dc;
        int64_t t3 = 0;
        confirm_fault(&d, 2, &t3, &dc);
        pv_relay_authorize(&r, 7);
        CHECK(pv_relay_request_isolation(&r, &d, &dc, 7) == PV_RELAY_ERR_LABEL_NOT_ELIGIBLE && mock_calls == 0, "DUST");
    }

    /* stale evidence: no new isolation decision from old or invalid readings */
    older = conf;
    feed(&st, PV_STATUS_INVALID_INPUT, -1, 0, t += 5000, &conf);       /* confirmed, but this reading is invalid */
    CHECK(conf.status == PV_CONF_CONFIRMED_BY_CONFIGURED_RULES && conf.fresh == 0, "invalid reading is not fresh");
    pv_relay_authorize(&r, 7);
    CHECK(pv_relay_request_isolation(&r, &st, &conf, 7) == PV_RELAY_ERR_STALE_EVIDENCE && mock_calls == 0,
          "invalid reading cannot trigger isolation");
    pv_relay_authorize(&r, 7);
    CHECK(pv_relay_request_isolation(&r, &st, &older, 7) == PV_RELAY_ERR_STALE_EVIDENCE && mock_calls == 0,
          "an older output cannot trigger isolation");
    feed(&st, PV_STATUS_OK, 0, 0.99f, t += 5000, &conf);              /* disagreeing usable reading */
    pv_relay_authorize(&r, 7);
    CHECK(conf.fresh == 0 && pv_relay_request_isolation(&r, &st, &conf, 7) == PV_RELAY_ERR_STALE_EVIDENCE,
          "a disagreeing reading cannot trigger isolation");
    feed(&st, PV_STATUS_LOW_LIGHT, -1, 0, t += 5000, &conf);
    feed(&st, PV_STATUS_LOW_LIGHT, -1, 0, t += 5000, &conf);           /* second unusable: expired */
    pv_relay_authorize(&r, 7);
    CHECK(conf.event == PV_EVENT_EXPIRED && pv_relay_request_isolation(&r, &st, &conf, 7) == PV_RELAY_ERR_NOT_CONFIRMED,
          "expired confirmation cannot trigger isolation");

    confirm_fault(&st, 3, &t, &conf);
    pv_relay_authorize(&r, 7);
    CHECK(pv_relay_request_isolation(&r, &st, &conf, 7) == PV_RELAY_OK && mock_calls == 1 &&
          r.state == PV_RELAY_STATE_ISOLATED_SIMULATED, "simulated isolation on fresh evidence");
    feed(&st, PV_STATUS_OK, 3, 0.99f, t += 5000, &conf);
    pv_relay_authorize(&r, 7);
    CHECK(pv_relay_request_isolation(&r, &st, &conf, 7) == PV_RELAY_ERR_DUPLICATE && mock_calls == 1, "duplicate");
    pv_relay_authorize(&r, 9);
    CHECK(pv_relay_request_reconnect(&r, 9) == PV_RELAY_OK && r.state == PV_RELAY_STATE_CONNECTED &&
          mock_calls == 2, "manual reconnect");
    pv_relay_authorize(&r, 9);
    CHECK(pv_relay_request_reconnect(&r, 9) == PV_RELAY_ERR_DUPLICATE && mock_calls == 2, "reconnect duplicate");

    mock_fail = 1;
    feed(&st, PV_STATUS_OK, 3, 0.99f, t += 5000, &conf);
    pv_relay_authorize(&r, 5);
    CHECK(pv_relay_request_isolation(&r, &st, &conf, 5) == PV_RELAY_ERR_DRIVER && r.state == PV_RELAY_STATE_LOCKED_OUT,
          "driver failure locks out");
    mock_fail = 0;
    pv_relay_authorize(&r, 5);
    CHECK(pv_relay_request_isolation(&r, &st, &conf, 5) == PV_RELAY_ERR_LOCKED_OUT, "locked out");
    CHECK(pv_relay_reset_lockout(&r, 5) == PV_RELAY_ERR_NOT_AUTHORIZED, "reset needs authorisation");
    pv_relay_authorize(&r, 5);
    CHECK(pv_relay_reset_lockout(&r, 5) == PV_RELAY_OK && r.state == PV_RELAY_STATE_CONNECTED, "reset");
    CHECK(pv_relay_log_read(&r, log, PV_RELAY_LOG_SIZE) == PV_RELAY_LOG_SIZE && log[0].action == PV_RELAY_ACT_RESET &&
          log[0].result == PV_RELAY_OK && log[0].seq == r.seq && log[1].result == PV_RELAY_ERR_NOT_AUTHORIZED, "log");
    {
        int i;
        for (i = 0; i < PV_RELAY_LOG_SIZE; i++) pv_relay_request_reconnect(&r, 1);   /* unauthorised, logged */
        CHECK(r.log_count > PV_RELAY_LOG_SIZE && pv_relay_log_read(&r, log, PV_RELAY_LOG_SIZE) == PV_RELAY_LOG_SIZE &&
              log[0].seq == r.seq && log[PV_RELAY_LOG_SIZE - 1].seq == r.seq - PV_RELAY_LOG_SIZE + 1 &&
              r.state == PV_RELAY_STATE_CONNECTED && mock_calls == 3, "log ring wraps, newest first");
    }
}

/* ---------------------------------------------------------------- messages */

static void print_msg(const pv_msg_meta_t *meta, const pv_prediction_t *p, const pv_confirm_output_t *c)
{
    char buf[1024];
    int rc = pv_message_json(meta, p, c, buf, sizeof(buf));
    CHECK(rc == PV_MSG_OK, "message rc %d", rc);
    if (rc == PV_MSG_OK) printf("PVMSG %s\n", buf);
}

static void test_message(void)
{
    pv_msg_meta_t meta = {"REC-T1", "SITE-A", "ESP32-A01", 1789000000000LL, 1, 0, 0};
    pv_prediction_t p8, pf, ps;
    pv_confirm_output_t c;
    pv_confirm_t st;
    pv_confirm_config_t cfg;
    float raw[PV_N_RAW];
    char buf[1024], small[64], topic[64];
    pv_api_classify(PV_BACKEND_INT8, ok_raw(0), &p8);
    pv_api_classify(PV_BACKEND_FLOAT32, ok_raw(1), &pf);
    memcpy(raw, ok_raw(0), sizeof(raw));
    raw[PV_RAW_IRRADIANCE_LUX] = 100.0f;
    pv_api_classify(PV_BACKEND_INT8, raw, &ps);
    pv_confirm_default_config(&cfg);
    pv_confirm_init(&st, &cfg);
    pv_confirm_update(&st, &p8, ok_raw(0), meta.timestamp_unix_ms, &c);

    print_msg(&meta, &p8, &c);                           /* INT8, classified, with confirmation */
    meta.record_id = "REC-T2";
    meta.device_id = NULL;
    print_msg(&meta, &pf, NULL);                         /* Float32, no confirmation, no device id */
    meta.record_id = "REC-T3";
    meta.has_sensor_mask = 1;
    meta.sensor_invalid_mask = 0;
    print_msg(&meta, &ps, NULL);                         /* skipped (LOW_LIGHT) */

    meta.record_id = "bad id";
    CHECK(pv_message_json(&meta, &p8, NULL, buf, sizeof(buf)) == PV_MSG_ERR_BAD_ID, "bad id refused");
    meta.record_id = "REC-T4";
    meta.timestamp_valid = 0;
    CHECK(pv_message_json(&meta, &p8, NULL, buf, sizeof(buf)) == PV_MSG_ERR_NO_TIMESTAMP, "no timestamp");
    meta.timestamp_valid = 1;
    CHECK(pv_message_json(&meta, &p8, NULL, small, sizeof(small)) == PV_MSG_ERR_BUFFER && small[0] == '\0',
          "small buffer");
    CHECK(pv_message_topic("SITE-A", topic, sizeof(topic)) == PV_MSG_OK && strcmp(topic, "solar/SITE-A/prediction") == 0,
          "topic");
    CHECK(pv_message_topic("SITE/A", topic, sizeof(topic)) == PV_MSG_ERR_BAD_ID, "topic injection refused");
}

/* ---------------------------------------------------------------- shadow pipeline */

static void test_pipeline(void)
{
    pv_pipeline_t pl;
    pv_pipeline_output_t o;
    pv_confirm_config_t cfg;
    pv_sensor_sample_t s;
    char json[1024];
    pv_confirm_default_config(&cfg);
    CHECK(pv_pipeline_init(&pl, PV_BACKEND_INT8, &cfg, "SITE/A", "ESP32-A01") == -1, "bad site refused");
    CHECK(pv_pipeline_init(&pl, (pv_backend_t)3, &cfg, "SITE-A", "ESP32-A01") == -1, "bad backend refused");
    CHECK(pv_pipeline_init(&pl, PV_BACKEND_INT8, &cfg, "SITE-A", "ESP32-A01") == 0, "pipeline init");
    fill_sample(&s, ok_raw(0));
    CHECK(pv_pipeline_step(&pl, &s, "REC-P1", &o, json, sizeof(json)) == 0 && o.message_rc == PV_MSG_OK &&
          o.prediction.input_status == PV_STATUS_OK && o.confirmation.evidence == PV_EVID_USABLE, "complete reading");
    s.panel_temp2_c.state = PV_CH_NOT_INSTALLED;
    s.timestamp_unix_ms += 5000;
    CHECK(pv_pipeline_step(&pl, &s, "REC-P2", &o, json, sizeof(json)) == 0 && o.message_rc == PV_MSG_OK &&
          o.prediction.input_status == PV_STATUS_INVALID_INPUT && o.prediction.label == -1 &&
          o.confirmation.status == PV_CONF_INSUFFICIENT_EVIDENCE, "no second panel probe -> skipped, not a fault");
    printf("PVMSG %s\n", json);
    s.timestamp_valid = 0;
    CHECK(pv_pipeline_step(&pl, &s, "REC-P3", &o, json, sizeof(json)) == 0 &&
          o.message_rc == PV_MSG_ERR_NO_TIMESTAMP, "no message without a UTC timestamp");
    CHECK(pl.readings == 3 && pv_pipeline_step(&pl, NULL, "X", &o, json, sizeof(json)) == -1, "counts");
}

/* ---------------------------------------------------------------- suites */

static uint32_t host_ticks(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec);
}

static uint32_t scratch[4 * 200];

static void test_suites(void)
{
    pv_selftest_report_t rep;
    pv_bench_result_t b8, bf;
    CHECK(pv_selftest_run(&rep, "host-gcc") == 0 && rep.failures == 0, "self-test suite failures %d", rep.failures);
    CHECK(rep.f32_label_ok == rep.f32_label_total && rep.int8_exact_ok == rep.int8_total && rep.gate_ok == rep.gate_total,
          "self-test totals");
    CHECK(pv_bench_run(PV_BACKEND_INT8, 200, 20, host_ticks, 1000.0, scratch, &b8) == 0, "bench int8");
    CHECK(pv_bench_run(PV_BACKEND_FLOAT32, 200, 20, host_ticks, 1000.0, scratch, &bf) == 0, "bench f32");
    CHECK(b8.pre.n == 200 && b8.total.median_us <= b8.total.p95_us && b8.total.p95_us <= b8.total.p99_us &&
          b8.total.p99_us <= b8.total.max_us && b8.total.min_us <= b8.total.median_us, "stats ordered");
    CHECK(pv_bench_run(PV_BACKEND_INT8, 0, 0, host_ticks, 1000.0, scratch, &b8) == -1, "zero iterations refused");
    pv_bench_print_json(&b8, "host-gcc", "\"note\":\"host run, not an ESP32 measurement\"");
    pv_bench_print_json(&bf, "host-gcc", "");
    {
        uint32_t v[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        CHECK(pv_bench_percentile(v, 10, 50) == 5 && pv_bench_percentile(v, 10, 95) == 10 &&
              pv_bench_percentile(v, 10, 99) == 10 && pv_bench_percentile(v, 10, 0) == 1, "nearest rank");
    }
}

int main(void)
{
    test_api();
    test_sensor();
    test_confirm();
    test_relay();
    test_message();
    test_pipeline();
    test_suites();
    if (failures) {
        printf("RUNTIME FAILED %d of %d checks\n", failures, checks);
        return 1;
    }
    printf("RUNTIME ALL PASSED %d\n", checks);
    return 0;
}
