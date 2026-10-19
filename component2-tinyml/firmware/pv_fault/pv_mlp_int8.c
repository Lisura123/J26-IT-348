/* INT8 8-16-5 MLP core (deployment derivative of the float contract).
 * acc1 = qb1 + sum qx*qW1 (int32); k = sat(rshift_rnd(acc1*M1, SH1), -128, 127);
 * hq = tanh LUT[k + 128]; acc2 = qb2 + sum hq*qW2 (int32);
 * logits = (float)acc2 * OUT_SCALE. Bit-exact with quantize.int8_core(). */
#include <stddef.h>

#include "pv_fault.h"
#include "pv_model_int8.h"
#include "pv_quant.h"

typedef char pv_check_qw1[(sizeof(PV_QW1) == PV_N_FEAT * PV_N_HID) ? 1 : -1];
typedef char pv_check_qw2[(sizeof(PV_QW2) == PV_N_HID * PV_N_OUT) ? 1 : -1];
typedef char pv_check_lut[(sizeof(PV_TANH_LUT) == 256) ? 1 : -1];
typedef char pv_check_sh1[(PV_SH1 >= 1 && PV_SH1 <= 62) ? 1 : -1];

void pv_mlp_int8_core(const int8_t qx[PV_N_FEAT], int32_t acc2[PV_N_OUT],
                      int8_t hq_out[PV_N_HID], int16_t k_out[PV_N_HID], int *lut_clips)
{
    int8_t hq[PV_N_HID];
    int i, j, c;
    for (j = 0; j < PV_N_HID; j++) {
        int32_t acc = PV_QB1[j];
        int32_t k;
        for (i = 0; i < PV_N_FEAT; i++) acc += (int32_t)qx[i] * (int32_t)PV_QW1[i][j];
        k = pv_sat(pv_rshift_rnd((int64_t)acc * (int64_t)PV_M1, PV_SH1), PV_K_MIN, PV_K_MAX, lut_clips);
        hq[j] = PV_TANH_LUT[k - PV_K_MIN];
        if (hq_out != NULL) hq_out[j] = hq[j];
        if (k_out != NULL) k_out[j] = (int16_t)k;
    }
    for (c = 0; c < PV_N_OUT; c++) {
        int32_t acc = PV_QB2[c];
        for (j = 0; j < PV_N_HID; j++) acc += (int32_t)hq[j] * (int32_t)PV_QW2[j][c];
        acc2[c] = acc;
    }
}

int pv_mlp_int8(const float z[PV_N_FEAT], int8_t qx[PV_N_FEAT], int32_t acc2[PV_N_OUT],
                float logits[PV_N_OUT], int *input_clips, int *lut_clips)
{
    int i, c;
    for (i = 0; i < PV_N_FEAT; i++)
        if (pv_quantize_input(z[i], PV_INV_SX, &qx[i], input_clips) != 0) return -1;
    pv_mlp_int8_core(qx, acc2, NULL, NULL, lut_clips);
    for (c = 0; c < PV_N_OUT; c++) logits[c] = (float)acc2[c] * PV_OUT_SCALE;
    return 0;
}
