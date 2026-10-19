#ifndef STUB_FREERTOS_TASK_H
#define STUB_FREERTOS_TASK_H
/* Host stub (tests only). vTaskDelay() ends the host process, so the
 * applications' idle loops terminate. */
#include "FreeRTOS.h"
void vTaskDelay(TickType_t ticks);
unsigned uxTaskGetStackHighWaterMark(void *task);
#endif
