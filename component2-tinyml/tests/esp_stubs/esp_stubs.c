/* Host stub implementations (tests only). */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/task.h"

void app_main(void);

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

const char *esp_get_idf_version(void) { return "host-stub"; }
int64_t esp_timer_get_time(void) { return (int64_t)(now_ns() / 1000u); }
uint32_t esp_cpu_get_cycle_count(void) { return (uint32_t)now_ns(); }
uint32_t esp_get_free_heap_size(void) { return 0; }
size_t heap_caps_get_free_size(unsigned caps) { (void)caps; return 0; }
size_t heap_caps_get_minimum_free_size(unsigned caps) { (void)caps; return 0; }
unsigned uxTaskGetStackHighWaterMark(void *task) { (void)task; return 0; }
void vTaskDelay(TickType_t ticks) { (void)ticks; fflush(stdout); exit(0); }

int main(void)
{
    app_main();
    return 0;
}
