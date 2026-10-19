/* Host test: portable C firmware vs Python golden vectors (golden_vectors.h).
 * Integer arithmetic and the INT8 core must match bit for bit; Float32
 * preprocessing and logits use small tolerances (logf/expf may differ by an ulp).
 * Prints one line per failure and "ALL PASSED <n>" on success. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "golden_vectors.h"
#include "pv_fault.h"
#include "pv_quant.h"

static int failures = 0, checks = 0;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static int close_f(float a, float b, float rel, float abs_tol)
{
    if (isnan(a) && isnan(b)) return 1;
    return fabsf(a - b) <= abs_tol + rel * fabsf(b);
}

static void test_arithmetic(void)
{
    int i;
    for (i = 0; i < GV_N_RSHIFT; i++)
        CHECK(pv_rshift_rnd(GV_RSHIFT[i].v, GV_RSHIFT[i].n) == GV_RSHIFT[i].expected,
              "rshift_rnd(%lld, %d)", (long long)GV_RSHIFT[i].v, GV_RSHIFT[i].n);
    for (i = 0; i < GV_N_QIN; i++) {
        int8_t q = 0;
        int clips = 0;
        int rc = pv_quantize_input(GV_QIN[i].z, GV_QIN[i].inv, &q, &clips);
        CHECK(rc == 0 && q == GV_QIN[i].expected && clips == GV_QIN[i].clipped,
              "quantize_input(%a, %a) = %d clips %d, expected %d clips %d",
              (double)GV_QIN[i].z, (double)GV_QIN[i].inv, q, clips, GV_QIN[i].expected, GV_QIN[i].clipped);
    }
    {
        int8_t q = 0;
        int clips = 0;
        CHECK(pv_quantize_input(NAN, 1.0f, &q, &clips) == -1, "NaN input must be rejected");
        CHECK(pv_quantize_input(INFINITY, 1.0f, &q, &clips) == -1, "inf input must be rejected");
    }
    for (i = 0; i < GV_N_ARGMAX; i++) {
        CHECK(pv_argmax_i32(GV_ARGMAX[i].acc) == GV_ARGMAX[i].expected, "argmax_i32 case %d", i);
        CHECK(pv_argmax_f32(GV_ARGMAX[i].f) == GV_ARGMAX[i].expected, "argmax_f32 case %d", i);
    }
    for (i = 0; i < GV_N_CORE; i++) {
        int32_t acc2[PV_N_OUT];
        int8_t hq[PV_N_HID];
        int16_t k[PV_N_HID];
        int lut = 0;
        pv_mlp_int8_core(GV_CORE[i].qx, acc2, hq, k, &lut);
        CHECK(memcmp(acc2, GV_CORE[i].acc2, sizeof(acc2)) == 0, "core %d acc2", i);
        CHECK(memcmp(hq, GV_CORE[i].hq, sizeof(hq)) == 0, "core %d hq", i);
        CHECK(memcmp(k, GV_CORE[i].k, sizeof(k)) == 0, "core %d k", i);
        CHECK(lut == GV_CORE[i].lut_clips, "core %d lut clips %d vs %d", i, lut, GV_CORE[i].lut_clips);
    }
}

static void test_z_cases(void)
{
    int i, c;
    for (i = 0; i < GV_N_Z; i++) {
        const gv_z_t *g = &GV_Z[i];
        int8_t qx[PV_N_FEAT];
        int32_t acc2[PV_N_OUT];
        float logits[PV_N_OUT], lf[PV_N_OUT];
        int in_clips = 0, lut = 0;
        CHECK(pv_mlp_int8(g->z, qx, acc2, logits, &in_clips, &lut) == 0, "%s: int8 rc", g->id);
        CHECK(memcmp(qx, g->qx, sizeof(qx)) == 0, "%s: qx differs", g->id);
        CHECK(memcmp(acc2, g->acc2, sizeof(acc2)) == 0, "%s: acc2 differs", g->id);
        CHECK(pv_argmax_i32(acc2) == g->label_int8, "%s: int8 label %d vs %d", g->id, pv_argmax_i32(acc2), g->label_int8);
        CHECK(in_clips == g->input_clips, "%s: input clips %d vs %d", g->id, in_clips, g->input_clips);
        pv_mlp_f32(g->z, lf);
        CHECK(pv_argmax_f32(lf) == g->label_f32, "%s: f32 label %d vs %d", g->id, pv_argmax_f32(lf), g->label_f32);
        for (c = 0; c < PV_N_OUT; c++)
            CHECK(close_f(lf[c], g->logits_f32[c], 1e-5f, 1e-4f), "%s: f32 logit %d %g vs %g",
                  g->id, c, (double)lf[c], (double)g->logits_f32[c]);
    }
}

static void test_raw_cases(void)
{
    int i, j;
    for (i = 0; i < GV_N_RAW; i++) {
        const gv_raw_t *g = &GV_RAW[i];
        pv_result_t r8, rf;
        int status = pv_input_status(g->raw);
        CHECK(status == g->status, "%s [%s]: status %d vs %d", g->id, g->origin, status, g->status);
        pv_classify_int8(g->raw, &r8);
        pv_classify_f32(g->raw, &rf);
        if (g->status != PV_STATUS_OK) {
            CHECK(r8.label == -1 && rf.label == -1, "%s: skipped reading must have no label", g->id);
            continue;
        }
        {
            float feat[PV_N_FEAT], z[PV_N_FEAT];
            pv_features(g->raw, feat);
            pv_scale(feat, z);
            for (j = 0; j < PV_N_FEAT; j++)
                CHECK(close_f(z[j], g->z[j], 1e-5f, 1e-5f), "%s: z[%d] %g vs %g", g->id, j, (double)z[j], (double)g->z[j]);
        }
        for (j = 0; j < PV_N_FEAT; j++)
            CHECK(r8.qx[j] - g->qx[j] <= 1 && g->qx[j] - r8.qx[j] <= 1, "%s: qx[%d] %d vs %d", g->id, j, r8.qx[j], g->qx[j]);
        if (g->exact_label) {
            CHECK(r8.label == g->label_int8, "%s: int8 label %d vs %d", g->id, r8.label, g->label_int8);
            CHECK(rf.label == g->label_f32, "%s: f32 label %d vs %d", g->id, rf.label, g->label_f32);
        }
        CHECK(r8.confidence > 0.0f && r8.confidence <= 1.0f, "%s: confidence", g->id);
    }
}

int main(void)
{
    test_arithmetic();
    test_z_cases();
    test_raw_cases();
    if (failures) {
        printf("FAILED %d of %d checks\n", failures, checks);
        return 1;
    }
    printf("ALL PASSED %d\n", checks);
    return 0;
}
