/* Float32 preprocessing: quality gates, pv8-v1 features, frozen scaler.
 * Same order of operations as src/tinyml/device_reference.py. */
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "pv_fault.h"
#include "pv_feature_params.h"

const char *const pv_label_names[PV_N_OUT] = PV_LABEL_NAMES_INIT;
const char *const pv_feature_names[PV_N_FEAT] = PV_FEATURE_NAMES_INIT;

/* Precedence: INVALID > SATURATED > LOW_LIGHT > OK. */
int pv_input_status(const float raw[PV_N_RAW])
{
    int i;
    for (i = 0; i < PV_N_RAW; i++)
        if (!isfinite(raw[i])) return PV_STATUS_INVALID_INPUT;
    for (i = 0; i < PV_N_TEMP_SENTINELS; i++)
        if (raw[PV_RAW_PANEL_TEMP1_C] == PV_TEMP_SENTINELS[i] ||
            raw[PV_RAW_PANEL_TEMP2_C] == PV_TEMP_SENTINELS[i])
            return PV_STATUS_INVALID_INPUT;
    for (i = 0; i < PV_N_RAW; i++)
        if (raw[i] < PV_LIMIT_LO[i] || raw[i] > PV_LIMIT_HI[i]) return PV_STATUS_INVALID_INPUT;
    if (raw[PV_RAW_IRRADIANCE_LUX] >= PV_LUX_SATURATION) return PV_STATUS_LUX_SATURATED;
    if (raw[PV_RAW_IRRADIANCE_LUX] < PV_LOW_LIGHT_LUX) return PV_STATUS_LOW_LIGHT;
    return PV_STATUS_OK;
}

/* Only call for status-OK readings (klx > 0). */
void pv_features(const float raw[PV_N_RAW], float feat[PV_N_FEAT])
{
    const float v = raw[PV_RAW_VOLTAGE_V], i_ma = raw[PV_RAW_CURRENT_MA], p = raw[PV_RAW_POWER_MW];
    const float t1 = raw[PV_RAW_PANEL_TEMP1_C], t2 = raw[PV_RAW_PANEL_TEMP2_C];
    const float klx = raw[PV_RAW_IRRADIANCE_LUX] / PV_LUX_PER_UNIT;
    const float tp = (t1 + t2) / 2.0f;
    const float vref = (PV_VREF_A + PV_VREF_B * logf(klx)) + PV_VREF_C * (tp - PV_T_REF_C);
    feat[0] = vref - v;                      /* v_deficit   */
    feat[1] = i_ma / klx;                    /* i_norm      */
    feat[2] = p / klx;                       /* p_norm      */
    feat[3] = raw[PV_RAW_DUST_VO_V];         /* dust_vo_v   */
    feat[4] = fabsf(t1 - t2);                /* temp_spread */
    feat[5] = tp - raw[PV_RAW_AMBIENT_TEMP_C]; /* temp_rise */
    feat[6] = v;                             /* voltage_v   */
    feat[7] = i_ma;                          /* current_ma  */
}

void pv_scale(const float feat[PV_N_FEAT], float z[PV_N_FEAT])
{
    int i;
    for (i = 0; i < PV_N_FEAT; i++) z[i] = (feat[i] - PV_SCALER_MEAN[i]) / PV_SCALER_STD[i];
}

void pv_softmax(const float logits[PV_N_OUT], float prob[PV_N_OUT])
{
    int c;
    float m = logits[0], s = 0.0f;
    for (c = 1; c < PV_N_OUT; c++) if (logits[c] > m) m = logits[c];
    for (c = 0; c < PV_N_OUT; c++) { prob[c] = expf(logits[c] - m); s += prob[c]; }
    for (c = 0; c < PV_N_OUT; c++) prob[c] /= s;
}

/* Ties go to the lowest index (LABELS order), as in quantize.argmax_first. */
int pv_argmax_f32(const float v[PV_N_OUT])
{
    int c, best = 0;
    for (c = 1; c < PV_N_OUT; c++) if (v[c] > v[best]) best = c;
    return best;
}

int pv_argmax_i32(const int32_t v[PV_N_OUT])
{
    int c, best = 0;
    for (c = 1; c < PV_N_OUT; c++) if (v[c] > v[best]) best = c;
    return best;
}

static void pv_finish(pv_result_t *out)
{
    int c;
    pv_softmax(out->logits, out->prob);
    out->confidence = out->prob[0];
    for (c = 1; c < PV_N_OUT; c++) if (out->prob[c] > out->confidence) out->confidence = out->prob[c];
}

static int pv_prepare(const float raw[PV_N_RAW], pv_result_t *out, float z[PV_N_FEAT])
{
    float feat[PV_N_FEAT];
    memset(out, 0, sizeof(*out));
    out->label = -1;
    out->status = pv_input_status(raw);
    if (out->status != PV_STATUS_OK) return 0;
    pv_features(raw, feat);
    pv_scale(feat, z);
    return 1;
}

void pv_classify_f32(const float raw[PV_N_RAW], pv_result_t *out)
{
    float z[PV_N_FEAT];
    if (!pv_prepare(raw, out, z)) return;
    pv_mlp_f32(z, out->logits);
    out->label = pv_argmax_f32(out->logits);
    pv_finish(out);
}

void pv_classify_int8(const float raw[PV_N_RAW], pv_result_t *out)
{
    float z[PV_N_FEAT];
    if (!pv_prepare(raw, out, z)) return;
    if (pv_mlp_int8(z, out->qx, out->acc2, out->logits, &out->input_clips, &out->lut_clips) != 0) {
        out->status = PV_STATUS_INVALID_INPUT;
        return;
    }
    out->label = pv_argmax_i32(out->acc2);
    pv_finish(out);
}
