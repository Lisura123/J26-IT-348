/* Shadow-mode pipeline: one call per reading from Member 1's sampling loop.
 *
 *   pv_sensor_sample_t -> pv_sensor_adapt -> pv_api_classify (Float32 or INT8)
 *     -> pv_confirm_update (non-actuating) -> prediction JSON (pv-msg-1)
 *
 * Shadow mode only: the result is logged or sent to the backend. Nothing is
 * switched; there is no relay call in this path. Portable C99. */
#ifndef PV_PIPELINE_H
#define PV_PIPELINE_H

#include <stddef.h>

#include "pv_api.h"
#include "pv_confirm.h"
#include "pv_message.h"
#include "pv_sensor.h"

typedef struct {
    pv_backend_t backend;
    pv_confirm_t confirm;
    char site_id[PV_MSG_MAX_ID + 1];
    char device_id[PV_MSG_MAX_ID + 1];   /* empty: omitted from messages */
    unsigned long readings;
} pv_pipeline_t;

typedef struct {
    pv_adapted_t adapted;
    pv_prediction_t prediction;
    pv_confirm_output_t confirmation;
    int message_rc;                       /* PV_MSG_OK or PV_MSG_ERR_*     */
} pv_pipeline_output_t;

/* Returns 0, or -1 for a bad backend, configuration or id. Calls pv_api_init(). */
int pv_pipeline_init(pv_pipeline_t *pl, pv_backend_t backend, const pv_confirm_config_t *cfg,
                     const char *site_id, const char *device_id);
/* Returns 0 when the reading was processed (also when it was skipped by a
 * quality gate); message_rc tells whether json holds a message. Returns -1
 * on NULL arguments or an API error. */
int pv_pipeline_step(pv_pipeline_t *pl, const pv_sensor_sample_t *sample, const char *record_id,
                     pv_pipeline_output_t *out, char *json, size_t json_len);

#endif /* PV_PIPELINE_H */
