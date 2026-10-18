/* Public inference API: metadata, initialisation self-check, backend
 * selection and result packaging over pv_fault.c. Portable C99. */
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "pv_api.h"
#include "pv_feature_params.h"
#include "pv_selfcheck.h"

/* Self-check tolerance for Float32 logits: logf/expf/tanhf may differ by an
 * ulp between C libraries; the INT8 accumulators are compared exactly. */
#define PV_SC_LOGIT_ABS_TOL 1e-4f
#define PV_SC_LOGIT_REL_TOL 1e-5f

/* The INT8 quantisation version string lives in pv_model_int8.h, which is
 * included only by pv_mlp_int8.c; the self-check header carries the same value. */
static const pv_model_info_t pv_info = {
    PV_API_VERSION, PV_MODEL_VERSION, PV_FEATURE_VERSION, PV_SC_QUANT_VERSION,
    PV_CONTRACT_SHA256, PV_WEIGHTS_SHA256, PV_N_FEAT, PV_N_HID, PV_N_OUT,
    PV_N_FEAT * PV_N_HID + PV_N_HID + PV_N_HID * PV_N_OUT + PV_N_OUT
};

static int pv_initialized = 0;
static unsigned pv_check_failures = 0;

static unsigned pv_check_labels(void)
{
    static const char *const expected[PV_N_OUT] = PV_SC_LABEL_NAMES_INIT;
    int c;
    for (c = 0; c < PV_N_OUT; c++)
        if (pv_label_names[c] == NULL || strcmp(pv_label_names[c], expected[c]) != 0) return PV_CHECK_LABELS;
    return 0;
}

static unsigned pv_check_scaler(void)
{
    int i;
    for (i = 0; i < PV_N_FEAT; i++)
        if (!isfinite(PV_SCALER_STD[i]) || !(PV_SCALER_STD[i] > 0.0f) || !isfinite(PV_SCALER_MEAN[i]))
            return PV_CHECK_SCALER;
    return 0;
}

static unsigned pv_check_metadata(void)
{
    if (strcmp(PV_MODEL_VERSION, PV_SC_MODEL_VERSION) != 0 || strcmp(PV_FEATURE_VERSION, PV_SC_FEATURE_VERSION) != 0 ||
        strcmp(PV_CONTRACT_SHA256, PV_SC_CONTRACT_SHA256) != 0 || strcmp(PV_WEIGHTS_SHA256, PV_SC_WEIGHTS_SHA256) != 0 ||
        pv_info.n_params != 229)
        return PV_CHECK_METADATA;
    return 0;
}

static unsigned pv_check_probes(void)
{
    unsigned fail = 0;
    int n, c;
    for (n = 0; n < PV_SC_N; n++) {
        float logits[PV_N_OUT], deq[PV_N_OUT];
        int8_t qx[PV_N_FEAT];
        int32_t acc2[PV_N_OUT];
        int in_clips = 0, lut_clips = 0;
        pv_mlp_f32(PV_SC_Z[n], logits);
        for (c = 0; c < PV_N_OUT; c++)
            if (!(fabsf(logits[c] - PV_SC_LOGITS_F32[n][c]) <=
                  PV_SC_LOGIT_ABS_TOL + PV_SC_LOGIT_REL_TOL * fabsf(PV_SC_LOGITS_F32[n][c])))
                fail |= PV_CHECK_F32_PROBES;
        if (pv_argmax_f32(logits) != PV_SC_LABEL_F32[n]) fail |= PV_CHECK_F32_PROBES;
        if (pv_mlp_int8(PV_SC_Z[n], qx, acc2, deq, &in_clips, &lut_clips) != 0 ||
            memcmp(acc2, PV_SC_ACC2[n], sizeof(acc2)) != 0 || pv_argmax_i32(acc2) != PV_SC_LABEL_INT8[n])
            fail |= PV_CHECK_INT8_PROBES;
    }
    return fail;
}

pv_err_t pv_api_init(void)
{
    if (pv_initialized) return PV_OK;
    pv_check_failures = pv_check_labels() | pv_check_scaler() | pv_check_metadata() | pv_check_probes();
    if (pv_check_failures != 0) return PV_ERR_MODEL_CHECK;
    pv_initialized = 1;
    return PV_OK;
}

int pv_api_is_initialized(void) { return pv_initialized; }
unsigned pv_api_last_check_failures(void) { return pv_check_failures; }
const pv_model_info_t *pv_api_model_info(void) { return &pv_info; }

pv_err_t pv_api_classify(pv_backend_t backend, const float raw[PV_N_RAW], pv_prediction_t *out)
{
    pv_result_t r;
    int c;
    if (out == NULL || raw == NULL) return PV_ERR_NULL_ARG;
    memset(out, 0, sizeof(*out));
    out->label = -1;
    out->input_status = PV_STATUS_INVALID_INPUT;
    if (!pv_initialized) return PV_ERR_NOT_INITIALIZED;
    if (backend == PV_BACKEND_FLOAT32) pv_classify_f32(raw, &r);
    else if (backend == PV_BACKEND_INT8) pv_classify_int8(raw, &r);
    else return PV_ERR_BAD_BACKEND;
    out->backend = backend;
    out->input_status = r.status;
    out->input_clips = r.input_clips;
    out->lut_clips = r.lut_clips;
    if (r.status != PV_STATUS_OK || r.label < 0 || r.label >= PV_N_OUT) {
        out->label = -1;                    /* skipped: no label, never NORMAL */
        return PV_OK;
    }
    out->label = r.label;
    out->label_name = pv_label_names[r.label];
    out->confidence = r.confidence;
    for (c = 0; c < PV_N_OUT; c++) out->prob[c] = r.prob[c];
    return PV_OK;
}

const char *pv_api_strerror(pv_err_t err)
{
    switch (err) {
    case PV_OK: return "OK";
    case PV_ERR_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case PV_ERR_NULL_ARG: return "NULL_ARG";
    case PV_ERR_BAD_BACKEND: return "BAD_BACKEND";
    case PV_ERR_MODEL_CHECK: return "MODEL_CHECK_FAILED";
    }
    return "UNKNOWN_ERROR";
}

const char *pv_backend_name(pv_backend_t backend)
{
    return backend == PV_BACKEND_FLOAT32 ? "FLOAT32" : backend == PV_BACKEND_INT8 ? "INT8" : "UNKNOWN";
}

const char *pv_input_status_name(int status)
{
    switch (status) {
    case PV_STATUS_OK: return "OK";
    case PV_STATUS_LOW_LIGHT: return "LOW_LIGHT";
    case PV_STATUS_LUX_SATURATED: return "LUX_SATURATED";
    case PV_STATUS_INVALID_INPUT: return "INVALID_INPUT";
    }
    return "UNKNOWN";
}

const char *pv_prediction_status_name(int status)
{
    switch (status) {
    case PV_STATUS_OK: return "CLASSIFIED";
    case PV_STATUS_LOW_LIGHT: return "SKIPPED_LOW_LIGHT";
    case PV_STATUS_LUX_SATURATED: return "SKIPPED_LUX_SATURATED";
    case PV_STATUS_INVALID_INPUT: return "SKIPPED_INVALID_INPUT";
    }
    return "SKIPPED_INVALID_INPUT";
}
