/* Inference benchmark core (portable). The platform supplies a tick counter
 * (ESP32: CPU cycle counter; host: clock_gettime) and its ticks per
 * microsecond. Each iteration times, on one status-OK golden reading:
 *   preprocessing  quality gate + pv8-v1 features + scaler
 *   neural         MLP (Float32) or input quantisation + INT8 core
 *   post           argmax + softmax
 *   total          one complete pv_api_classify() call (timed separately)
 * Results are printed as one JSON line per backend (schema pv-bench-1):
 *   PVBENCH {...}
 * Numbers are only meaningful for the platform named in the line. */
#ifndef PV_BENCH_H
#define PV_BENCH_H

#include <stdint.h>

#include "pv_api.h"

typedef uint32_t (*pv_ticks_fn)(void);

typedef struct {
    uint32_t n;
    double mean_us, median_us, p95_us, p99_us, min_us, max_us;
} pv_stage_stats_t;

typedef struct {
    pv_backend_t backend;
    uint32_t iterations, warmup, inputs;
    pv_stage_stats_t pre, nn, post, total;
    uint32_t label_checksum;     /* keeps results observable; equal across runs */
} pv_bench_result_t;

/* scratch: 4 * iterations uint32_t values. Returns 0, or -1 on bad arguments
 * or if the API is not initialised. */
int pv_bench_run(pv_backend_t backend, uint32_t iterations, uint32_t warmup, pv_ticks_fn ticks,
                 double ticks_per_us, uint32_t *scratch, pv_bench_result_t *out);

/* Prints the PVBENCH line. extra_json: "" or more fields without braces, e.g.
 * "\"heap_free_before\":1234" (platform memory figures). */
void pv_bench_print_json(const pv_bench_result_t *r, const char *platform, const char *extra_json);

/* Nearest-rank percentile of sorted values (p in 0..100). */
uint32_t pv_bench_percentile(const uint32_t *sorted, uint32_t n, double p);

#endif /* PV_BENCH_H */
