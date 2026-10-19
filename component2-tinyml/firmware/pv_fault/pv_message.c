/* Prediction message formatter (see pv_message.h). */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "pv_message.h"
#include "pv_sensor.h"

static const char *const pv_prob_fields[PV_N_OUT] = {
    "prob_normal", "prob_partial_shading", "prob_dust", "prob_open_circuit", "prob_short_circuit"
};

typedef struct { char *buf; size_t len, pos; int err, first; } pv_jw_t;

static void pv_put(pv_jw_t *w, const char *fmt, ...)
{
    va_list ap;
    int n;
    if (w->err) return;
    va_start(ap, fmt);
    n = vsnprintf(w->buf + w->pos, w->len - w->pos, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= w->len - w->pos) { w->err = 1; return; }
    w->pos += (size_t)n;
}

static void pv_key(pv_jw_t *w, const char *key)
{
    pv_put(w, "%s\"%s\":", w->first ? "" : ",", key);
    w->first = 0;
}

/* Values are IDs or fixed enum names: no character needs escaping. */
static void pv_str(pv_jw_t *w, const char *key, const char *val) { pv_key(w, key); pv_put(w, "\"%s\"", val); }
static void pv_int(pv_jw_t *w, const char *key, long val) { pv_key(w, key); pv_put(w, "%ld", val); }
static void pv_num(pv_jw_t *w, const char *key, float val) { pv_key(w, key); pv_put(w, "%.7g", (double)val); }

int pv_message_id_ok(const char *id)
{
    size_t i, n;
    if (id == NULL) return 0;
    n = strlen(id);
    if (n == 0 || n > PV_MSG_MAX_ID) return 0;
    for (i = 0; i < n; i++) {
        char c = id[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' ||
              c == ':' || c == '-'))
            return 0;
    }
    return 1;
}

int pv_message_topic(const char *site_id, char *buf, size_t buf_len)
{
    int n;
    if (site_id == NULL || buf == NULL) return PV_MSG_ERR_NULL_ARG;
    if (!pv_message_id_ok(site_id)) return PV_MSG_ERR_BAD_ID;
    n = snprintf(buf, buf_len, "solar/%s/prediction", site_id);
    return (n < 0 || (size_t)n >= buf_len) ? PV_MSG_ERR_BUFFER : PV_MSG_OK;
}

int pv_message_json(const pv_msg_meta_t *meta, const pv_prediction_t *pred, const pv_confirm_output_t *conf,
                    char *buf, size_t buf_len)
{
    pv_jw_t w;
    char ts[24];
    const pv_model_info_t *info = pv_api_model_info();
    int classified, c;
    if (meta == NULL || pred == NULL || buf == NULL || buf_len == 0) return PV_MSG_ERR_NULL_ARG;
    if (!pv_message_id_ok(meta->record_id) || !pv_message_id_ok(meta->site_id) ||
        (meta->device_id != NULL && !pv_message_id_ok(meta->device_id)))
        return PV_MSG_ERR_BAD_ID;
    if (!meta->timestamp_valid || pv_format_utc(meta->timestamp_unix_ms, ts, sizeof(ts)) != 0)
        return PV_MSG_ERR_NO_TIMESTAMP;
    classified = pred->input_status == PV_STATUS_OK && pred->label >= 0 && pred->label < PV_N_OUT;

    w.buf = buf; w.len = buf_len; w.pos = 0; w.err = 0; w.first = 1;
    buf[0] = '\0';
    pv_put(&w, "{");
    pv_str(&w, "record_id", meta->record_id);
    pv_str(&w, "timestamp_utc", ts);
    pv_str(&w, "site_id", meta->site_id);
    if (meta->device_id != NULL) pv_str(&w, "device_id", meta->device_id);
    pv_str(&w, "message_schema", PV_MSG_SCHEMA);
    pv_str(&w, "model_version", info->model_version);
    pv_str(&w, "feature_version", info->feature_version);
    pv_str(&w, "inference_backend", pv_backend_name(pred->backend));
    if (pred->backend == PV_BACKEND_INT8) pv_str(&w, "quant_version", info->quant_version);
    pv_str(&w, "prediction_status", pv_prediction_status_name(pred->input_status));
    pv_str(&w, "input_status", pv_input_status_name(pred->input_status));
    pv_str(&w, "decision_source", classified ? "ML" : "NONE");
    if (classified) {
        pv_str(&w, "predicted_label", pv_label_names[pred->label]);
        pv_num(&w, "confidence", pred->confidence);
        for (c = 0; c < PV_N_OUT; c++) pv_num(&w, pv_prob_fields[c], pred->prob[c]);
        if (pred->backend == PV_BACKEND_INT8) {
            pv_int(&w, "input_clips", pred->input_clips);
            pv_int(&w, "lut_clips", pred->lut_clips);
        }
    }
    if (meta->has_sensor_mask) pv_int(&w, "sensor_invalid_mask", meta->sensor_invalid_mask);
    if (conf != NULL) {
        pv_str(&w, "confirmation_status", pv_confirm_status_name(conf->status));
        pv_str(&w, "confirmation_evidence", pv_evidence_name(conf->evidence));
        pv_int(&w, "confirmation_count", conf->consecutive);
        if (conf->candidate_label >= 0 && conf->candidate_label < PV_N_OUT)
            pv_str(&w, "candidate_label", pv_label_names[conf->candidate_label]);
        if (conf->confirmed_label >= 0 && conf->confirmed_label < PV_N_OUT)
            pv_str(&w, "confirmed_label", pv_label_names[conf->confirmed_label]);
    }
    pv_put(&w, "}");
    if (w.err) {
        buf[0] = '\0';
        return PV_MSG_ERR_BUFFER;
    }
    return PV_MSG_OK;
}
