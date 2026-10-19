#ifndef STUB_FREERTOS_FREERTOS_H
#define STUB_FREERTOS_FREERTOS_H
/* Host stub (tests only). */
#include <stdint.h>
typedef uint32_t TickType_t;
#define pdMS_TO_TICKS(ms) ((TickType_t)(ms))
#endif
