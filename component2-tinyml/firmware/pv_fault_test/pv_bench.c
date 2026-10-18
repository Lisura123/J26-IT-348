/* Inference benchmark core (see pv_bench.h). Portable C99. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "golden_vectors.h"
#include "pv_bench.h"

static int pv_cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

uint32_t pv_bench_percentile(const uint32_t *sorted, uint32_t n, double p)
{
    double rank;
    uint32_t idx;
    if (n == 0) return 0;
    rank = ceil(p / 100.0 * (double)n);
    idx = rank < 1.0 ? 0 : (uint32_t)rank - 1;
    return sorted[idx >= n ? n - 1 : idx];
}

static void pv_stats(uint32_t *v, uint32_t n, double tpu, pv_stage_stats_t *s)
{
    uint32_t i;
    double sum = 0.0;
    for (i = 0; i < n; i++) sum += v[i];
    qsort(v, n, sizeof(uint32_t), pv_cmp_u32);
    s->n = n;
    s->mean_us = sum / n / tpu;
    s->median_us = pv_bench_percentile(v, n, 50.0) / tpu;
    s->p95_us = pv_bench_percentile(v, n, 95.0) / tpu;
    s->p99_us = pv_bench_percentile(v, n, 99.0) / tpu;
    s->min_us = v[0] / tpu;
    s->max_us = v[n - 1] / tpu;
}

int pv_bench_run(pv_backend_t backend, uint32_t iterations, uint32_t warmup, pv_ticks_fn ticks,
                 double ticks_per_us, uint32_t *scratch, pv_bench_result_t *out)
{
    int ok_idx[GV_N_RAW];
    int n_ok = 0, i;
    uint32_t it, *t_pre, *t_nn, *t_post, *t_total;
    volatile uint32_t sink = 0;
    if (ticks == NULL || scratch == NULL || out == NULL || iterations == 0 || !(ticks_per_us > 0.0)) return -1;
    if (backend != PV_BACKEND_FLOAT32 && backend != PV_BACKEND_INT8) return -1;
    if (!pv_api_is_initialized()) return -1;
    for (i = 0; i < GV_N_RAW; i++) if (GV_RAW[i].status == PV_STATUS_OK) ok_idx[n_ok++] = i;
    if (n_ok == 0) return -1;
    memset(out, 0, sizeof(*out));
    t_pre = scratch; t_nn = scratch + iterations; t_post = scratch + 2 * iterations; t_total = scratch + 3 * iterations;

    for (it = 0; it < warmup + iterations; it++) {
        const float *raw = GV_RAW[ok_idx[it % (uint32_t)n_ok]].raw;
        float feat[PV_N_FEAT], z[PV_N_FEAT], logits[PV_N_OUT], prob[PV_N_OUT];
        int8_t qx[PV_N_FEAT];
        int32_t acc2[PV_N_OUT];
        int in_clips = 0, lut = 0, label, status;
        pv_prediction_t p;
        uint32_t a, b, c, d, e, f;
        a = ticks();
        status = pv_input_status(raw);
        pv_features(raw, feat);
        pv_scale(feat, z);
        b = ticks();
        if (backend == PV_BACKEND_FLOAT32) pv_mlp_f32(z, logits);
        else pv_mlp_int8(z, qx, acc2, logits, &in_clips, &lut);
        c = ticks();
        label = backend == PV_BACKEND_FLOAT32 ? pv_argmax_f32(logits) : pv_argmax_i32(acc2);
        pv_softmax(logits, prob);
        d = ticks();
        e = ticks();
        pv_api_classify(backend, raw, &p);
        f = ticks();
        sink += (uint32_t)(label + status) + (uint32_t)(p.label + 1) + (prob[0] > 2.0f);
        if (it >= warmup) {
            uint32_t k = it - warmup;
            t_pre[k] = b - a; t_nn[k] = c - b; t_post[k] = d - c; t_total[k] = f - e;
        }
    }
    out->backend = backend;
    out->iterations = iterations;
    out->warmup = warmup;
    out->inputs = (uint32_t)n_ok;
    out->label_checksum = sink;
    pv_stats(t_pre, iterations, ticks_per_us, &out->pre);
    pv_stats(t_nn, iterations, ticks_per_us, &out->nn);
    pv_stats(t_post, iterations, ticks_per_us, &out->post);
    pv_stats(t_total, iterations, ticks_per_us, &out->total);
    return 0;
}

static void pv_print_stage(const char *name, const pv_stage_stats_t *s)
{
    printf("\"%s\":{\"n\":%lu,\"mean_us\":%.4f,\"median_us\":%.4f,\"p95_us\":%.4f,\"p99_us\":%.4f,"
           "\"min_us\":%.4f,\"max_us\":%.4f}", name, (unsigned long)s->n, s->mean_us, s->median_us, s->p95_us,
           s->p99_us, s->min_us, s->max_us);
}

void pv_bench_print_json(const pv_bench_result_t *r, const char *platform, const char *extra_json)
{
    const pv_model_info_t *m = pv_api_model_info();
    printf("PVBENCH {\"schema\":\"pv-bench-1\",\"platform\":\"%s\",\"backend\":\"%s\",\"model_version\":\"%s\",",
           platform ? platform : "unknown", pv_backend_name(r->backend), m->model_version);
    if (r->backend == PV_BACKEND_INT8) printf("\"quant_version\":\"%s\",", m->quant_version);
    printf("\"iterations\":%lu,\"warmup\":%lu,\"inputs\":%lu,\"label_checksum\":%lu,", (unsigned long)r->iterations,
           (unsigned long)r->warmup, (unsigned long)r->inputs, (unsigned long)r->label_checksum);
    pv_print_stage("preprocessing", &r->pre); printf(",");
    pv_print_stage("neural", &r->nn); printf(",");
    pv_print_stage("post", &r->post); printf(",");
    pv_print_stage("total", &r->total);
    if (extra_json != NULL && extra_json[0] != '\0') printf(",%s", extra_json);
    printf("}\n");
}
