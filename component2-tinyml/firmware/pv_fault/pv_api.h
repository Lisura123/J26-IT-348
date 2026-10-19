/* Public inference API of the PV fault classifier (Component 2).
 *
 * Thin layer over pv_fault.h: model metadata, an initialisation self-check,
 * Float32 or INT8 selection per call, quality-gate status, label and
 * confidence output, INT8 clipping diagnostics and error codes.
 * Portable C99; no ESP-IDF headers. Classification only: nothing here
 * switches hardware. */
#ifndef PV_API_H
#define PV_API_H

#include <stdint.h>

#include "pv_fault.h"

#define PV_API_VERSION "pv-api-1"

typedef enum { PV_BACKEND_FLOAT32 = 0, PV_BACKEND_INT8 = 1 } pv_backend_t;

typedef enum {
    PV_OK = 0,
    PV_ERR_NOT_INITIALIZED = -1,   /* pv_api_init() not called or failed       */
    PV_ERR_NULL_ARG = -2,
    PV_ERR_BAD_BACKEND = -3,
    PV_ERR_MODEL_CHECK = -4        /* initialisation self-check failed          */
} pv_err_t;

/* Self-check failure bits (pv_api_last_check_failures()). */
#define PV_CHECK_LABELS        0x01u   /* label names or order differ from the contract */
#define PV_CHECK_SCALER        0x02u   /* scaler std not finite or <= 0                  */
#define PV_CHECK_F32_PROBES    0x04u   /* Float32 outputs differ from the reference      */
#define PV_CHECK_INT8_PROBES   0x08u   /* INT8 accumulators differ (bit-exact check)      */
#define PV_CHECK_METADATA      0x10u   /* version metadata differs from the check data   */

typedef struct {
    const char *api_version;
    const char *model_version;     /* contract model_version, e.g. c2-local-v1   */
    const char *feature_version;   /* pv8-v1                                     */
    const char *quant_version;     /* int8-v1                                    */
    const char *contract_sha256;
    const char *weights_sha256;
    int n_inputs, n_hidden, n_outputs, n_params;
} pv_model_info_t;

typedef struct {
    pv_backend_t backend;
    int input_status;              /* PV_STATUS_* quality gate result            */
    int label;                     /* index into pv_label_names, -1 if skipped   */
    const char *label_name;        /* NULL when not classified                   */
    float confidence;              /* max probability; 0 when not classified     */
    float prob[PV_N_OUT];          /* LABELS order; all 0 when not classified    */
    int input_clips;               /* INT8 only: inputs clamped to +-127         */
    int lut_clips;                 /* INT8 only: tanh lookup indices clamped     */
} pv_prediction_t;

/* Runs the self-check (labels, scaler, reference vectors for both backends).
 * Must succeed before pv_api_classify() is used. Idempotent. */
pv_err_t pv_api_init(void);
int pv_api_is_initialized(void);
unsigned pv_api_last_check_failures(void);
const pv_model_info_t *pv_api_model_info(void);

/* raw: the eight calibrated physical inputs in PV_RAW_* order (V, mA, mW,
 * degC, degC, degC, lx, V); NaN for a missing value. A reading that fails a
 * quality gate returns PV_OK with input_status != PV_STATUS_OK and no label:
 * a skipped reading is never a fault and never NORMAL. */
pv_err_t pv_api_classify(pv_backend_t backend, const float raw[PV_N_RAW], pv_prediction_t *out);

const char *pv_api_strerror(pv_err_t err);
const char *pv_backend_name(pv_backend_t backend);          /* "FLOAT32" / "INT8"     */
const char *pv_input_status_name(int status);               /* "OK", "LOW_LIGHT", ... */
const char *pv_prediction_status_name(int status);          /* "CLASSIFIED", "SKIPPED_..." */

#endif /* PV_API_H */
