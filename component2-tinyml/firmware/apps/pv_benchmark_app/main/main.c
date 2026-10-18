/* ESP32 benchmark application (Component 2).
 *
 * 1. Runs the golden-vector self-test first: timing a wrong model is useless.
 * 2. Calibrates the CPU cycle counter against esp_timer (µs).
 * 3. Times Float32 and INT8 inference (pv_bench.c): preprocessing, neural
 *    network, post-processing and the full classify call separately, with
 *    mean, median, p95, p99 over CONFIG_PV_FAULT_BENCH_ITERATIONS runs.
 * 4. Reports heap and stack figures. Flash and static RAM come from the
 *    build: `idf.py size` / `idf.py size-components` (not measurable here).
 * Output: PVINFO, PVTEST and PVBENCH JSON lines, then PVBENCH_DONE.
 * No sensors, no network, no GPIO. Parse the log with
 * src/tinyml/benchmark.py; numbers count as ESP32 measurements only when this
 * application really ran on a board. */
#include <inttypes.h>
#include <stdio.h>

#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "pv_api.h"
#include "pv_bench.h"
#include "pv_selftest.h"
#include "sdkconfig.h"

#define N_ITER CONFIG_PV_FAULT_BENCH_ITERATIONS
#define N_WARM CONFIG_PV_FAULT_BENCH_WARMUP

static uint32_t scratch[4 * N_ITER];

static uint32_t cycles(void) { return (uint32_t)esp_cpu_get_cycle_count(); }

/* Cycles per microsecond, measured over about 100 ms of busy waiting. */
static double calibrate_cycles_per_us(void)
{
    int64_t t0 = esp_timer_get_time(), t1;
    uint32_t c0 = cycles(), c1;
    do { t1 = esp_timer_get_time(); } while (t1 - t0 < 100000);
    c1 = cycles();
    return (double)(uint32_t)(c1 - c0) / (double)(t1 - t0);
}

void app_main(void)
{
    pv_selftest_report_t rep;
    pv_bench_result_t r;
    const pv_model_info_t *m = pv_api_model_info();
    char extra[512];
    size_t heap_before, heap_after, heap_min;
    int64_t t_init0, t_init1;
    double cpu_us;
    int b;

    heap_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    t_init0 = esp_timer_get_time();
    if (pv_api_init() != PV_OK) {
        printf("PVBENCH_ERROR {\"stage\":\"init\",\"check_failures\":%u}\n", pv_api_last_check_failures());
        for (;;) vTaskDelay(pdMS_TO_TICKS(10000));
    }
    t_init1 = esp_timer_get_time();
    heap_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    printf("PVINFO {\"app\":\"pv_benchmark_app\",\"target\":\"%s\",\"idf_version\":\"%s\",\"cpu_freq_mhz_config\":%d,"
           "\"model_version\":\"%s\",\"quant_version\":\"%s\",\"weights_sha256\":\"%s\",\"iterations\":%d,"
           "\"warmup\":%d}\n",
           CONFIG_IDF_TARGET, esp_get_idf_version(), CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, m->model_version,
           m->quant_version, m->weights_sha256, N_ITER, N_WARM);

    if (pv_selftest_run(&rep, CONFIG_IDF_TARGET) != 0) {
        printf("PVBENCH_ERROR {\"stage\":\"selftest\",\"failures\":%d}\n", rep.failures);
        for (;;) vTaskDelay(pdMS_TO_TICKS(10000));
    }

    cpu_us = calibrate_cycles_per_us();
    for (b = 0; b < 2; b++) {
        pv_backend_t backend = b == 0 ? PV_BACKEND_FLOAT32 : PV_BACKEND_INT8;
        if (pv_bench_run(backend, N_ITER, N_WARM, cycles, cpu_us, scratch, &r) != 0) {
            printf("PVBENCH_ERROR {\"stage\":\"bench\",\"backend\":\"%s\"}\n", pv_backend_name(backend));
            continue;
        }
        heap_min = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);
        snprintf(extra, sizeof(extra),
                 "\"timer\":\"cpu_cycle_counter\",\"cycles_per_us_calibrated\":%.3f,\"init_us\":%" PRId64 ","
                 "\"heap_free_before_init\":%u,\"heap_free_after_init\":%u,\"heap_min_free\":%u,"
                 "\"stack_high_water_mark_bytes\":%u,\"flash_and_static_ram\":\"from idf.py size\"",
                 cpu_us, t_init1 - t_init0, (unsigned)heap_before, (unsigned)heap_after, (unsigned)heap_min,
                 (unsigned)uxTaskGetStackHighWaterMark(NULL));
        pv_bench_print_json(&r, CONFIG_IDF_TARGET, extra);
    }
    printf("PVBENCH_DONE\n");
    for (;;) vTaskDelay(pdMS_TO_TICKS(10000));
}
