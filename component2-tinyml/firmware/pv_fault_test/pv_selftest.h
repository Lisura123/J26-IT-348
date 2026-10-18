/* On-device test suite: the committed golden vectors (tests/host_c/
 * golden_vectors.h) run through the public API on the target. Independent
 * of sensors. Reports over stdout (the serial console on ESP32):
 *   PVTEST_FAIL <what>            one line per failure (first 20)
 *   PVTEST {...}                  one JSON summary line (schema pv-selftest-1)
 * A PASS line says only that this binary reproduced the Python reference on
 * the platform it ran on; the "platform" field records where that was. */
#ifndef PV_SELFTEST_H
#define PV_SELFTEST_H

typedef struct {
    int checks, failures;
    int f32_label_ok, f32_label_total;       /* expected-label comparisons, Float32      */
    int int8_exact_ok, int8_total;           /* bit-exact INT8 accumulators and labels   */
    int raw_label_ok, raw_label_total;       /* end-to-end raw readings, both backends   */
    int gate_ok, gate_total;                 /* quality-gate status comparisons          */
    float max_logit_abs_diff;                /* Float32 logits vs Python reference       */
    float max_z_abs_diff;                    /* Float32 preprocessing vs Python          */
    int qx_off_by_one;                       /* INT8 inputs 1 step from Python (allowed) */
} pv_selftest_report_t;

/* Returns 0 if every check passed. platform: free text, e.g. "host-gcc" or
 * the chip name reported by the ESP32 application. */
int pv_selftest_run(pv_selftest_report_t *rep, const char *platform);

#endif /* PV_SELFTEST_H */
