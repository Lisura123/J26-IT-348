/* Float32 8-16-5 tanh MLP: the deployment fallback and reference. */
#include <math.h>

#include "pv_fault.h"
#include "pv_model_f32.h"

typedef char pv_check_w1[(sizeof(PV_W1) == sizeof(float) * PV_N_FEAT * PV_N_HID) ? 1 : -1];
typedef char pv_check_w2[(sizeof(PV_W2) == sizeof(float) * PV_N_HID * PV_N_OUT) ? 1 : -1];

void pv_mlp_f32(const float z[PV_N_FEAT], float logits[PV_N_OUT])
{
    float h[PV_N_HID];
    int i, j, c;
    for (j = 0; j < PV_N_HID; j++) {
        float s = PV_B1[j];
        for (i = 0; i < PV_N_FEAT; i++) s += z[i] * PV_W1[i][j];
        h[j] = tanhf(s);
    }
    for (c = 0; c < PV_N_OUT; c++) {
        float s = PV_B2[c];
        for (j = 0; j < PV_N_HID; j++) s += h[j] * PV_W2[j][c];
        logits[c] = s;
    }
}
