/* ESP32 on-device test application (Component 2).
 *
 * Runs the golden-vector suite (pv_selftest.c) through the public API and
 * prints the result on the serial console. No sensors, no network, no GPIO.
 * Expected serial output: model information, any PVTEST_FAIL lines, one
 * "PVTEST {...}" JSON line and "PVTEST_DONE". A result counts as an ESP32
 * result only when this application was actually flashed and run on a board
 * and the console log is kept (docs/ESP32_INTEGRATION.md). */
#include <stdio.h>

#include "esp_idf_version.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "pv_api.h"
#include "pv_selftest.h"
#include "sdkconfig.h"

void app_main(void)
{
    pv_selftest_report_t rep;
    const pv_model_info_t *m = pv_api_model_info();
    int rc;
    printf("PVINFO {\"app\":\"pv_selftest_app\",\"target\":\"%s\",\"idf_version\":\"%s\",\"api_version\":\"%s\","
           "\"model_version\":\"%s\",\"feature_version\":\"%s\",\"quant_version\":\"%s\",\"contract_sha256\":\"%s\","
           "\"weights_sha256\":\"%s\"}\n",
           CONFIG_IDF_TARGET, esp_get_idf_version(), m->api_version, m->model_version, m->feature_version,
           m->quant_version, m->contract_sha256, m->weights_sha256);
    rc = pv_selftest_run(&rep, CONFIG_IDF_TARGET);
    printf("PVTEST_DONE %s\n", rc == 0 ? "PASS" : "FAIL");
    for (;;) vTaskDelay(pdMS_TO_TICKS(10000));   /* keep the console output; nothing else runs */
}
