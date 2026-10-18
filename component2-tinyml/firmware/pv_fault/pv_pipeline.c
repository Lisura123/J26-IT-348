/* Shadow-mode pipeline (see pv_pipeline.h). */
#include <string.h>

#include "pv_pipeline.h"

int pv_pipeline_init(pv_pipeline_t *pl, pv_backend_t backend, const pv_confirm_config_t *cfg,
                     const char *site_id, const char *device_id)
{
    if (pl == NULL || cfg == NULL || !pv_message_id_ok(site_id)) return -1;
    if (device_id != NULL && device_id[0] != '\0' && !pv_message_id_ok(device_id)) return -1;
    if (backend != PV_BACKEND_FLOAT32 && backend != PV_BACKEND_INT8) return -1;
    memset(pl, 0, sizeof(*pl));
    if (pv_api_init() != PV_OK || pv_confirm_init(&pl->confirm, cfg) != 0) return -1;
    pl->backend = backend;
    strcpy(pl->site_id, site_id);
    if (device_id != NULL) strcpy(pl->device_id, device_id);
    return 0;
}

int pv_pipeline_step(pv_pipeline_t *pl, const pv_sensor_sample_t *sample, const char *record_id,
                     pv_pipeline_output_t *out, char *json, size_t json_len)
{
    pv_msg_meta_t meta;
    if (pl == NULL || sample == NULL || out == NULL) return -1;
    memset(out, 0, sizeof(*out));
    out->message_rc = PV_MSG_ERR_NULL_ARG;
    if (pv_sensor_adapt(sample, &out->adapted) != 0) return -1;
    if (pv_api_classify(pl->backend, out->adapted.raw, &out->prediction) != PV_OK) return -1;
    pv_confirm_update(&pl->confirm, &out->prediction, out->adapted.raw,
                      out->adapted.timestamp_valid ? out->adapted.timestamp_unix_ms : -1, &out->confirmation);
    pl->readings++;
    if (json != NULL && json_len > 0) {
        meta.record_id = record_id;
        meta.site_id = pl->site_id;
        meta.device_id = pl->device_id[0] ? pl->device_id : NULL;
        meta.timestamp_unix_ms = out->adapted.timestamp_unix_ms;
        meta.timestamp_valid = out->adapted.timestamp_valid;
        meta.has_sensor_mask = 1;
        meta.sensor_invalid_mask = out->adapted.invalid_mask;
        out->message_rc = pv_message_json(&meta, &out->prediction, &out->confirmation, json, json_len);
    }
    return 0;
}
