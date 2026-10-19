/* Fault confirmation state machine (see pv_confirm.h). Non-actuating. */
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "pv_confirm.h"

#define PV_LABEL_NORMAL 0
#define PV_LABEL_OPEN 3
#define PV_LABEL_SHORT 4

void pv_confirm_default_config(pv_confirm_config_t *cfg)
{
    if (cfg == NULL) return;
    cfg->confirm_count = 5;
    cfg->recovery_count = 5;
    cfg->max_gap_readings = 1;
    cfg->min_confidence = 0.90f;
    cfg->max_interval_ms = 15000;
    cfg->open_max_current_ma = NAN;
    cfg->short_max_voltage_v = NAN;
}

static int pv_config_ok(const pv_confirm_config_t *c)
{
    if (c->confirm_count < 2 || c->recovery_count < 1) return 0;
    if (!(c->min_confidence > 0.0f && c->min_confidence <= 1.0f)) return 0;
    if (c->max_interval_ms < 0) return 0;
    if (!isnan(c->open_max_current_ma) && !(c->open_max_current_ma >= 0.0f && isfinite(c->open_max_current_ma)))
        return 0;
    if (!isnan(c->short_max_voltage_v) && !(c->short_max_voltage_v >= 0.0f && isfinite(c->short_max_voltage_v)))
        return 0;
    return 1;
}

void pv_confirm_reset(pv_confirm_t *st)
{
    if (st == NULL) return;
    st->candidate_label = -1;
    st->consecutive = 0;
    st->gap = 0;
    st->confirmed_label = -1;
    st->recovery = 0;
    st->unusable_since_evidence = 0;
    st->has_last_ts = 0;
    st->last_ts_ms = 0;
}

int pv_confirm_init(pv_confirm_t *st, const pv_confirm_config_t *cfg)
{
    if (st == NULL || cfg == NULL || !pv_config_ok(cfg)) return -1;
    memset(st, 0, sizeof(*st));
    st->cfg = *cfg;
    pv_confirm_reset(st);
    return 0;
}

static pv_electrical_check_t pv_electrical(const pv_confirm_config_t *c, int label, const float *raw)
{
    float limit, value;
    if (label == PV_LABEL_OPEN) { limit = c->open_max_current_ma; value = raw ? raw[PV_RAW_CURRENT_MA] : NAN; }
    else if (label == PV_LABEL_SHORT) { limit = c->short_max_voltage_v; value = raw ? raw[PV_RAW_VOLTAGE_V] : NAN; }
    else return PV_ELEC_NOT_APPLICABLE;
    if (isnan(limit)) return PV_ELEC_NOT_CONFIGURED;
    return (isfinite(value) && value <= limit) ? PV_ELEC_CONSISTENT : PV_ELEC_CONTRADICTED;
}

static void pv_cancel_candidate(pv_confirm_t *st, pv_confirm_output_t *out)
{
    if (st->candidate_label >= 0) {
        st->diag.candidates_cancelled++;
        out->event = PV_EVENT_CANDIDATE_CANCELLED;
    }
    st->candidate_label = -1;
    st->consecutive = 0;
    st->gap = 0;
}

static void pv_fill(const pv_confirm_t *st, pv_confirm_output_t *out)
{
    out->candidate_label = st->candidate_label;
    out->confirmed_label = st->confirmed_label;
    out->consecutive = st->consecutive;
    out->recovery_progress = st->recovery;
}

/* A confirmed fault that is no longer backed by fresh evidence is dropped;
 * confirming it again needs a complete new sequence. */
static void pv_expire(pv_confirm_t *st, pv_confirm_output_t *out)
{
    if (st->confirmed_label < 0) return;
    out->expired_label = st->confirmed_label;
    st->confirmed_label = -1;
    st->recovery = 0;
    st->diag.expirations++;
    out->event = PV_EVENT_EXPIRED;
}

/* An unusable reading is no evidence either way: it counts against a pending
 * candidate and against the freshness of a confirmed fault. A confirmed fault
 * survives at most max_gap_readings of them (reported, never fresh). */
static void pv_unusable(pv_confirm_t *st, pv_confirm_output_t *out)
{
    if (st->candidate_label >= 0 && ++st->gap > st->cfg.max_gap_readings) pv_cancel_candidate(st, out);
    if (st->confirmed_label >= 0 && ++st->unusable_since_evidence > st->cfg.max_gap_readings) pv_expire(st, out);
    out->status = st->confirmed_label >= 0 ? PV_CONF_CONFIRMED_BY_CONFIGURED_RULES : PV_CONF_INSUFFICIENT_EVIDENCE;
}

int pv_confirm_update(pv_confirm_t *st, const pv_prediction_t *pred, const float *raw,
                      int64_t timestamp_ms, pv_confirm_output_t *out)
{
    int label;
    if (st == NULL || pred == NULL || out == NULL) return -1;
    memset(out, 0, sizeof(*out));
    out->event = PV_EVENT_NONE;
    out->electrical = PV_ELEC_NOT_APPLICABLE;
    st->diag.readings++;
    out->reading_seq = st->diag.readings;
    out->expired_label = -1;

    /* Temporal consistency: a long gap invalidates a pending sequence. */
    if (timestamp_ms >= 0) {
        if (st->has_last_ts && st->cfg.max_interval_ms > 0 &&
            (timestamp_ms - st->last_ts_ms > st->cfg.max_interval_ms || timestamp_ms < st->last_ts_ms)) {
            st->diag.stale++;
            out->evidence = PV_EVID_STALE;
            pv_cancel_candidate(st, out);
            pv_expire(st, out);                         /* old evidence cannot keep a fault confirmed */
        }
        st->last_ts_ms = timestamp_ms;
        st->has_last_ts = 1;
    }

    if (pred->input_status != PV_STATUS_OK || pred->label < 0 || pred->label >= PV_N_OUT) {
        st->diag.skipped_by_quality_gate++;
        out->evidence = PV_EVID_SKIPPED_BY_QUALITY_GATE;
        pv_unusable(st, out);
        pv_fill(st, out);
        return 0;
    }
    label = pred->label;
    out->confidence = pred->confidence;
    if (!(pred->confidence >= st->cfg.min_confidence)) {
        st->diag.low_confidence++;
        out->evidence = PV_EVID_LOW_CONFIDENCE;
        pv_unusable(st, out);
        pv_fill(st, out);
        return 0;
    }
    out->electrical = pv_electrical(&st->cfg, label, raw);
    if (out->electrical == PV_ELEC_CONTRADICTED) {
        st->diag.electrical_contradictions++;
        out->evidence = PV_EVID_ELECTRICAL_CONTRADICTION;
        if (st->candidate_label == label) pv_cancel_candidate(st, out);
        pv_unusable(st, out);
        pv_fill(st, out);
        return 0;
    }
    if (out->evidence != PV_EVID_STALE) out->evidence = PV_EVID_USABLE;
    st->diag.usable++;

    st->unusable_since_evidence = 0;

    if (st->confirmed_label >= 0) {                     /* hysteresis on a confirmed fault */
        if (label == st->confirmed_label) {
            st->recovery = 0;
            out->status = PV_CONF_CONFIRMED_BY_CONFIGURED_RULES;
            out->fresh = 1;
            pv_fill(st, out);
            return 0;
        }
        if (++st->recovery < st->cfg.recovery_count) {
            out->status = PV_CONF_CONFIRMED_BY_CONFIGURED_RULES;   /* not fresh: this reading disagrees */
            pv_fill(st, out);
            return 0;
        }
        st->confirmed_label = -1;
        st->recovery = 0;
        st->diag.clears++;
        out->event = PV_EVENT_CLEARED;
        /* fall through: this reading is evaluated as new evidence (NORMAL or a new candidate) */
    }

    if (label == PV_LABEL_NORMAL) {
        if (st->candidate_label >= 0) pv_cancel_candidate(st, out);
        out->status = PV_CONF_NO_FAULT;
        pv_fill(st, out);
        return 0;
    }
    if (st->candidate_label != label) {
        if (st->candidate_label >= 0) {
            st->diag.label_flips++;
            pv_cancel_candidate(st, out);
        }
        st->candidate_label = label;
        st->consecutive = 1;
        st->gap = 0;
        st->diag.candidates_started++;
        if (out->event == PV_EVENT_NONE) out->event = PV_EVENT_CANDIDATE_STARTED;
        out->status = PV_CONF_ML_PREDICTED;
        pv_fill(st, out);
        return 0;
    }
    st->consecutive++;
    st->gap = 0;
    if (st->consecutive >= st->cfg.confirm_count) {
        st->confirmed_label = label;
        st->recovery = 0;
        st->candidate_label = -1;
        st->diag.confirmations++;
        out->event = PV_EVENT_CONFIRMED;
        out->status = PV_CONF_CONFIRMED_BY_CONFIGURED_RULES;
        out->fresh = 1;
        pv_fill(st, out);
        out->consecutive = st->consecutive;
        st->consecutive = 0;
        return 0;
    }
    out->status = PV_CONF_CONFIRMATION_PENDING;
    pv_fill(st, out);
    return 0;
}

const char *pv_confirm_status_name(pv_confirm_status_t s)
{
    switch (s) {
    case PV_CONF_NO_FAULT: return "NO_FAULT";
    case PV_CONF_ML_PREDICTED: return "ML_PREDICTED";
    case PV_CONF_CONFIRMATION_PENDING: return "CONFIRMATION_PENDING";
    case PV_CONF_CONFIRMED_BY_CONFIGURED_RULES: return "CONFIRMED_BY_CONFIGURED_RULES";
    case PV_CONF_INSUFFICIENT_EVIDENCE: return "INSUFFICIENT_EVIDENCE";
    }
    return "UNKNOWN";
}

const char *pv_evidence_name(pv_evidence_t e)
{
    switch (e) {
    case PV_EVID_USABLE: return "USABLE";
    case PV_EVID_SKIPPED_BY_QUALITY_GATE: return "SKIPPED_BY_QUALITY_GATE";
    case PV_EVID_LOW_CONFIDENCE: return "LOW_CONFIDENCE";
    case PV_EVID_ELECTRICAL_CONTRADICTION: return "ELECTRICAL_CONTRADICTION";
    case PV_EVID_STALE: return "STALE";
    }
    return "UNKNOWN";
}
