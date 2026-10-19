/* Fault confirmation layer (software only, NON-ACTUATING).
 *
 * Turns the stream of per-reading ML predictions into a diagnostic status.
 * A fault is reported as CONFIRMED_BY_CONFIGURED_RULES only after the same
 * fault label was predicted for confirm_count consecutive usable readings,
 * each with confidence >= min_confidence, without a time gap longer than
 * max_interval_ms and without more than max_gap_readings unusable readings
 * in between. A confirmed fault is cleared only after recovery_count
 * consecutive usable readings without that label (hysteresis). A confirmed
 * fault EXPIRES when it is no longer supported by fresh evidence: after more
 * than max_gap_readings unusable readings in a row, or after a time gap
 * longer than max_interval_ms; a new full sequence is then needed. Only an
 * output with fresh = 1 (this very reading was usable and showed the
 * confirmed label) may be used for a relay decision (pv_relay.c).
 *
 * This is a configurable software rule, NOT an independently validated
 * electrical fault confirmation. All defaults are PROVISIONAL and
 * conservative; the optional electrical consistency checks for OPEN_CIRCUIT
 * and SHORT_CIRCUIT are DISABLED by default because Member 1 has not
 * provided thresholds. Nothing here switches hardware. Portable C99. */
#ifndef PV_CONFIRM_H
#define PV_CONFIRM_H

#include <stdint.h>

#include "pv_api.h"

typedef enum {
    PV_CONF_NO_FAULT = 0,              /* usable reading, no fault pending or confirmed         */
    PV_CONF_ML_PREDICTED = 1,          /* first reading of a new fault label (ML only)          */
    PV_CONF_CONFIRMATION_PENDING = 2,  /* same fault label repeated, not yet confirm_count      */
    PV_CONF_CONFIRMED_BY_CONFIGURED_RULES = 3,
    PV_CONF_INSUFFICIENT_EVIDENCE = 4  /* reading unusable (skipped, low confidence, contradicted) */
} pv_confirm_status_t;

typedef enum {
    PV_EVID_USABLE = 0,
    PV_EVID_SKIPPED_BY_QUALITY_GATE = 1,   /* LOW_LIGHT, LUX_SATURATED, INVALID_INPUT */
    PV_EVID_LOW_CONFIDENCE = 2,
    PV_EVID_ELECTRICAL_CONTRADICTION = 3,  /* enabled check disagrees with OPEN/SHORT */
    PV_EVID_STALE = 4                      /* time since previous reading too long     */
} pv_evidence_t;

typedef enum {
    PV_ELEC_NOT_APPLICABLE = 0,   /* label is not OPEN_CIRCUIT / SHORT_CIRCUIT */
    PV_ELEC_NOT_CONFIGURED = 1,   /* threshold is NaN (default)                */
    PV_ELEC_CONSISTENT = 2,
    PV_ELEC_CONTRADICTED = 3
} pv_electrical_check_t;

typedef enum {
    PV_EVENT_NONE = 0, PV_EVENT_CANDIDATE_STARTED, PV_EVENT_CANDIDATE_CANCELLED,
    PV_EVENT_CONFIRMED, PV_EVENT_CLEARED, PV_EVENT_EXPIRED
} pv_confirm_event_t;

typedef struct {
    uint8_t confirm_count;      /* consecutive matching usable readings to confirm (>= 2)      */
    uint8_t recovery_count;     /* consecutive usable non-matching readings to clear (>= 1)    */
    uint8_t max_gap_readings;   /* unusable readings tolerated inside a pending sequence       */
    float min_confidence;       /* (0, 1]                                                     */
    int64_t max_interval_ms;    /* >0: longer gaps between timestamps reset a pending fault   */
    float open_max_current_ma;  /* NaN = disabled; OPEN_CIRCUIT needs current_ma <= this      */
    float short_max_voltage_v;  /* NaN = disabled; SHORT_CIRCUIT needs voltage_v <= this      */
} pv_confirm_config_t;

typedef struct {
    uint32_t readings, usable, skipped_by_quality_gate, low_confidence, stale, electrical_contradictions;
    uint32_t candidates_started, candidates_cancelled, label_flips, confirmations, clears, expirations;
} pv_confirm_diag_t;

typedef struct {
    pv_confirm_status_t status;
    pv_evidence_t evidence;              /* classification of this reading           */
    pv_electrical_check_t electrical;
    pv_confirm_event_t event;
    int candidate_label;                 /* -1 if none                                */
    int confirmed_label;                 /* -1 if none                                */
    int consecutive;                     /* matching usable readings of the candidate */
    int recovery_progress;               /* while a confirmed fault is clearing       */
    int expired_label;                   /* label of a confirmation that expired now  */
    int fresh;                           /* 1: this reading is usable evidence for the
                                            confirmed label (required by pv_relay)     */
    uint32_t reading_seq;                /* = diag.readings after this update          */
    float confidence;                    /* of this reading (0 if not classified)     */
} pv_confirm_output_t;

typedef struct {
    pv_confirm_config_t cfg;
    int candidate_label, consecutive, gap, confirmed_label, recovery;
    int unusable_since_evidence;         /* unusable readings since the last usable one */
    int64_t last_ts_ms;
    int has_last_ts;
    pv_confirm_diag_t diag;
} pv_confirm_t;

/* Conservative provisional defaults: confirm 5, recover 5, gap 1, confidence
 * 0.90, max interval 15000 ms (3 x the 5 s sampling period), electrical
 * checks disabled. None of these values is validated on hardware. */
void pv_confirm_default_config(pv_confirm_config_t *cfg);
/* Returns 0, or -1 if the configuration is invalid (NULL, confirm_count < 2,
 * recovery_count < 1, min_confidence outside (0, 1], negative thresholds). */
int pv_confirm_init(pv_confirm_t *st, const pv_confirm_config_t *cfg);
void pv_confirm_reset(pv_confirm_t *st);   /* clears state, keeps config and diagnostics */

/* One reading. pred: from pv_api_classify(); raw: the same eight inputs (for
 * the optional electrical checks; may be NULL); timestamp_ms < 0 = unknown. */
int pv_confirm_update(pv_confirm_t *st, const pv_prediction_t *pred, const float *raw,
                      int64_t timestamp_ms, pv_confirm_output_t *out);

const char *pv_confirm_status_name(pv_confirm_status_t s);
const char *pv_evidence_name(pv_evidence_t e);

#endif /* PV_CONFIRM_H */
