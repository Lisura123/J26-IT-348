/* Prediction message (JSON) for the group backend stream "prediction".
 *
 * Follows project-backend/schema.yaml (stream `prediction`, measurement
 * `fault_predictions`, mode open) and backend/validate.py on Member 3's
 * branch feature/mongodb-migration: one FLAT JSON object, snake_case field
 * names (^[a-z][a-z0-9_]*$), at most 40 fields, timestamp_utc as
 * "YYYY-MM-DDTHH:MM:SSZ", required record_id, site_id, timestamp_utc;
 * predicted_label one of the five labels. Fields without a value are
 * omitted (never sent as 0 or as NORMAL). MQTT topic: solar/{site_id}/prediction.
 * Field list and meaning: docs/MESSAGING.md (schema pv-msg-1). Portable C99. */
#ifndef PV_MESSAGE_H
#define PV_MESSAGE_H

#include <stddef.h>
#include <stdint.h>

#include "pv_api.h"
#include "pv_confirm.h"

#define PV_MSG_SCHEMA "pv-msg-1"
#define PV_MSG_MAX_ID 64

enum {
    PV_MSG_OK = 0,
    PV_MSG_ERR_NULL_ARG = -1,
    PV_MSG_ERR_BAD_ID = -2,          /* record/site/device id empty, too long or bad characters */
    PV_MSG_ERR_NO_TIMESTAMP = -3,    /* the backend requires timestamp_utc                      */
    PV_MSG_ERR_BUFFER = -4           /* output buffer too small                                 */
};

typedef struct {
    const char *record_id;
    const char *site_id;
    const char *device_id;           /* may be NULL (backend fills it from sites.csv)            */
    int64_t timestamp_unix_ms;
    int timestamp_valid;
    int has_sensor_mask;             /* 1: include sensor_invalid_mask                         */
    uint16_t sensor_invalid_mask;    /* pv_adapted_t.invalid_mask                              */
} pv_msg_meta_t;

/* Writes the JSON object into buf (NUL-terminated). conf may be NULL (no
 * confirmation fields). Returns PV_MSG_OK or a PV_MSG_ERR_* code. */
int pv_message_json(const pv_msg_meta_t *meta, const pv_prediction_t *pred, const pv_confirm_output_t *conf,
                    char *buf, size_t buf_len);
int pv_message_topic(const char *site_id, char *buf, size_t buf_len);
/* IDs: 1..64 characters from [A-Za-z0-9_.:-]. */
int pv_message_id_ok(const char *id);

#endif /* PV_MESSAGE_H */
