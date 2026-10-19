#ifndef STUB_ESP_HEAP_CAPS_H
#define STUB_ESP_HEAP_CAPS_H
/* Host stub (tests only). */
#include <stddef.h>
#define MALLOC_CAP_8BIT 4
size_t heap_caps_get_free_size(unsigned caps);
size_t heap_caps_get_minimum_free_size(unsigned caps);
#endif
