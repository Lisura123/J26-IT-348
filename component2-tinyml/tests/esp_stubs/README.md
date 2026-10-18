# Host stubs for ESP-IDF APIs (tests only)

Minimal stand-ins for the ESP-IDF functions used by `firmware/apps/*/main/main.c`,
so the two applications can be **compiled and executed on the host** by
`tests/test_esp_apps.py`. They check the application code, not the board:
`CONFIG_IDF_TARGET` is `"host-stub"`, the cycle counter is `clock_gettime`, and the
heap and stack figures are fixed placeholders. Output produced with these stubs is
never an ESP32 result. Not used by the real ESP-IDF build.
