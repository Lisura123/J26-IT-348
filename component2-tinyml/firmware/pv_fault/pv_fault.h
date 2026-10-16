/* PV fault classifier: Float32 preprocessing + INT8 (or Float32) MLP.
 * Portable C99. Constants come from generated headers (src/tinyml/codegen.py):
 *   generated/pv_feature_params.h  frozen FeatureConfig (pv8-v1)
 *   generated/pv_model_f32.h       Float32 weights (fallback / reference)
 *   generated/pv_model_int8.h      INT8 deployment derivative
 * Classification only: no relay or actuator code. */
#ifndef PV_FAULT_H
#define PV_FAULT_H

#include <stdint.h>

#define PV_N_RAW 8
#define PV_N_FEAT 8
#define PV_N_HID 16
#define PV_N_OUT 5

/* Raw input order = features.RAW_INPUTS */
enum {
    PV_RAW_VOLTAGE_V = 0, PV_RAW_CURRENT_MA, PV_RAW_POWER_MW, PV_RAW_PANEL_TEMP1_C,
    PV_RAW_PANEL_TEMP2_C, PV_RAW_AMBIENT_TEMP_C, PV_RAW_IRRADIANCE_LUX, PV_RAW_DUST_VO_V
};

/* Input status codes = features.STATUS_CODES */
enum {
    PV_STATUS_OK = 0, PV_STATUS_LOW_LIGHT = 1, PV_STATUS_LUX_SATURATED = 2, PV_STATUS_INVALID_INPUT = 3
};

typedef struct {
    int status;                /* PV_STATUS_*                                  */
    int label;                 /* index into pv_label_names, -1 when skipped   */
    float logits[PV_N_OUT];
    float prob[PV_N_OUT];
    float confidence;
    int32_t acc2[PV_N_OUT];    /* INT8 path only                               */
    int8_t qx[PV_N_FEAT];      /* INT8 path only                               */
    int input_clips;           /* INT8 inputs clamped to +-127                 */
    int lut_clips;             /* tanh lookup indices clamped to -128..127     */
} pv_result_t;

extern const char *const pv_label_names[PV_N_OUT];
extern const char *const pv_feature_names[PV_N_FEAT];

int pv_input_status(const float raw[PV_N_RAW]);
void pv_features(const float raw[PV_N_RAW], float feat[PV_N_FEAT]);
void pv_scale(const float feat[PV_N_FEAT], float z[PV_N_FEAT]);
void pv_softmax(const float logits[PV_N_OUT], float prob[PV_N_OUT]);
int pv_argmax_f32(const float v[PV_N_OUT]);
int pv_argmax_i32(const int32_t v[PV_N_OUT]);

void pv_mlp_f32(const float z[PV_N_FEAT], float logits[PV_N_OUT]);
/* INT8 core on already quantised inputs; hq_out/k_out may be NULL. */
void pv_mlp_int8_core(const int8_t qx[PV_N_FEAT], int32_t acc2[PV_N_OUT],
                      int8_t hq_out[PV_N_HID], int16_t k_out[PV_N_HID], int *lut_clips);
/* Quantise z and run the core. Returns -1 if z is not finite. */
int pv_mlp_int8(const float z[PV_N_FEAT], int8_t qx[PV_N_FEAT], int32_t acc2[PV_N_OUT],
                float logits[PV_N_OUT], int *input_clips, int *lut_clips);

void pv_classify_f32(const float raw[PV_N_RAW], pv_result_t *out);
void pv_classify_int8(const float raw[PV_N_RAW], pv_result_t *out);

#endif /* PV_FAULT_H */
