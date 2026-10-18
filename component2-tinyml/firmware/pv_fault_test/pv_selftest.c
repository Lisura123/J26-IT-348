/* On-device golden-vector test suite (see pv_selftest.h). Portable C99. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "golden_vectors.h"
#include "pv_api.h"
#include "pv_feature_params.h"
#include "pv_selftest.h"

#define PV_MAX_FAIL_LINES 20

static pv_selftest_report_t *R;

static void check(int ok, const char *what, int idx)
{
    R->checks++;
    if (ok) return;
    if (R->failures++ < PV_MAX_FAIL_LINES) printf("PVTEST_FAIL %s %d\n", what, idx);
}

static int close_f(float a, float b, float rel, float abs_tol)
{
    return fabsf(a - b) <= abs_tol + rel * fabsf(b);
}

static void test_z_cases(void)
{
    int i, c;
    for (i = 0; i < GV_N_Z; i++) {
        const gv_z_t *g = &GV_Z[i];
        float lf[PV_N_OUT], deq[PV_N_OUT];
        int8_t qx[PV_N_FEAT];
        int32_t acc2[PV_N_OUT];
        int in_clips = 0, lut = 0, ok8;
        pv_mlp_f32(g->z, lf);
        for (c = 0; c < PV_N_OUT; c++) {
            float d = fabsf(lf[c] - g->logits_f32[c]);
            if (d > R->max_logit_abs_diff) R->max_logit_abs_diff = d;
            check(close_f(lf[c], g->logits_f32[c], 1e-5f, 1e-4f), "z_case_f32_logit", i);
        }
        R->f32_label_total++;
        if (pv_argmax_f32(lf) == g->label_f32) R->f32_label_ok++;
        check(pv_argmax_f32(lf) == g->label_f32, "z_case_f32_label", i);
        ok8 = pv_mlp_int8(g->z, qx, acc2, deq, &in_clips, &lut) == 0 && memcmp(qx, g->qx, sizeof(qx)) == 0 &&
              memcmp(acc2, g->acc2, sizeof(acc2)) == 0 && pv_argmax_i32(acc2) == g->label_int8 &&
              in_clips == g->input_clips;
        R->int8_total++;
        if (ok8) R->int8_exact_ok++;
        check(ok8, "z_case_int8_bit_exact", i);
    }
}

static void test_raw_cases(void)
{
    int i, j;
    for (i = 0; i < GV_N_RAW; i++) {
        const gv_raw_t *g = &GV_RAW[i];
        pv_prediction_t pf, p8;
        R->gate_total++;
        if (pv_input_status(g->raw) == g->status) R->gate_ok++;
        check(pv_input_status(g->raw) == g->status, "raw_gate_status", i);
        check(pv_api_classify(PV_BACKEND_FLOAT32, g->raw, &pf) == PV_OK, "raw_classify_f32_rc", i);
        check(pv_api_classify(PV_BACKEND_INT8, g->raw, &p8) == PV_OK, "raw_classify_int8_rc", i);
        if (g->status != PV_STATUS_OK) {
            check(pf.label == -1 && p8.label == -1 && pf.label_name == NULL && p8.label_name == NULL,
                  "raw_skipped_has_no_label", i);
            continue;
        }
        {
            float feat[PV_N_FEAT], z[PV_N_FEAT];
            pv_features(g->raw, feat);
            pv_scale(feat, z);
            for (j = 0; j < PV_N_FEAT; j++) {
                float d = fabsf(z[j] - g->z[j]);
                if (d > R->max_z_abs_diff) R->max_z_abs_diff = d;
                check(close_f(z[j], g->z[j], 1e-5f, 1e-5f), "raw_preprocessing_z", i);
            }
        }
        {
            pv_result_t r8;
            pv_classify_int8(g->raw, &r8);
            for (j = 0; j < PV_N_FEAT; j++) {
                int d = r8.qx[j] - g->qx[j];
                if (d != 0) R->qx_off_by_one++;
                check(d >= -1 && d <= 1, "raw_int8_qx", i);
            }
        }
        if (g->exact_label) {
            R->raw_label_total += 2;
            R->raw_label_ok += (pf.label == g->label_f32) + (p8.label == g->label_int8);
            check(pf.label == g->label_f32, "raw_label_f32", i);
            check(p8.label == g->label_int8, "raw_label_int8", i);
        }
        check(pf.confidence > 0.0f && pf.confidence <= 1.0f && p8.confidence > 0.0f && p8.confidence <= 1.0f,
              "raw_confidence_range", i);
    }
}

/* Explicit quality-gate checks built from the first status-OK golden reading. */
static void test_gates(void)
{
    int i, base = -1;
    float raw[PV_N_RAW];
    pv_prediction_t p;
    for (i = 0; i < GV_N_RAW; i++) if (GV_RAW[i].status == PV_STATUS_OK) { base = i; break; }
    check(base >= 0, "gate_base_reading_found", 0);
    if (base < 0) return;
#define GATE(field, value, expected, id) do { \
        memcpy(raw, GV_RAW[base].raw, sizeof(raw)); raw[field] = (value); \
        R->gate_total++; if (pv_input_status(raw) == (expected)) R->gate_ok++; \
        check(pv_input_status(raw) == (expected), "gate_" id, base); \
        pv_api_classify(PV_BACKEND_INT8, raw, &p); \
        check((expected) == PV_STATUS_OK || p.label == -1, "gate_no_label_" id, base); } while (0)
    GATE(PV_RAW_PANEL_TEMP2_C, NAN, PV_STATUS_INVALID_INPUT, "missing_panel_temp2");
    GATE(PV_RAW_PANEL_TEMP1_C, -127.0f, PV_STATUS_INVALID_INPUT, "ds18b20_error");
    GATE(PV_RAW_VOLTAGE_V, INFINITY, PV_STATUS_INVALID_INPUT, "non_finite_voltage");
    GATE(PV_RAW_CURRENT_MA, -1.0f, PV_STATUS_INVALID_INPUT, "negative_current");
    GATE(PV_RAW_IRRADIANCE_LUX, PV_LOW_LIGHT_LUX * 0.5f, PV_STATUS_LOW_LIGHT, "low_light");
    GATE(PV_RAW_IRRADIANCE_LUX, PV_LUX_SATURATION, PV_STATUS_LUX_SATURATED, "lux_saturated");
#undef GATE
}

int pv_selftest_run(pv_selftest_report_t *rep, const char *platform)
{
    const pv_model_info_t *m = pv_api_model_info();
    pv_err_t init;
    memset(rep, 0, sizeof(*rep));
    R = rep;
    init = pv_api_init();
    check(init == PV_OK, "api_init_selfcheck", (int)pv_api_last_check_failures());
    if (init == PV_OK) {
        test_z_cases();
        test_raw_cases();
        test_gates();
    }
    printf("PVTEST {\"schema\":\"pv-selftest-1\",\"platform\":\"%s\",\"api_version\":\"%s\","
           "\"model_version\":\"%s\",\"feature_version\":\"%s\",\"quant_version\":\"%s\",\"weights_sha256\":\"%s\","
           "\"checks\":%d,\"failures\":%d,\"f32_label_ok\":%d,\"f32_label_total\":%d,\"int8_exact_ok\":%d,"
           "\"int8_total\":%d,\"raw_label_ok\":%d,\"raw_label_total\":%d,\"gate_ok\":%d,\"gate_total\":%d,"
           "\"max_logit_abs_diff\":%.3g,\"max_z_abs_diff\":%.3g,\"qx_off_by_one\":%d,\"result\":\"%s\"}\n",
           platform ? platform : "unknown", m->api_version, m->model_version, m->feature_version, m->quant_version,
           m->weights_sha256, rep->checks, rep->failures, rep->f32_label_ok, rep->f32_label_total, rep->int8_exact_ok,
           rep->int8_total, rep->raw_label_ok, rep->raw_label_total, rep->gate_ok, rep->gate_total,
           (double)rep->max_logit_abs_diff, (double)rep->max_z_abs_diff, rep->qx_off_by_one,
           rep->failures ? "FAIL" : "PASS");
    return rep->failures ? 1 : 0;
}
