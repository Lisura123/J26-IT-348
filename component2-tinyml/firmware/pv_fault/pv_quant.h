/* Fixed integer arithmetic rules shared by the INT8 core (Phase 10).
 * Identical to src/tinyml/quantize.py: rnd(), rshift_rnd(), sat().
 * Portable C99; no platform headers. */
#ifndef PV_QUANT_H
#define PV_QUANT_H

#include <math.h>
#include <stdint.h>

#define PV_QMAX 127
#define PV_K_MIN (-128)
#define PV_K_MAX 127

/* v / 2^n rounded half away from zero, 1 <= n <= 62, |v| < 2^62.
 * Negative values are negated first, so no right shift of a negative number. */
static inline int64_t pv_rshift_rnd(int64_t v, int n)
{
    const int64_t half = (int64_t)1 << (n - 1);
    return v >= 0 ? (v + half) >> n : -((-v + half) >> n);
}

/* Clamp to [lo, hi]; increments *clips when the value was clamped. */
static inline int32_t pv_sat(int64_t v, int32_t lo, int32_t hi, int *clips)
{
    if (v < lo) { (*clips)++; return lo; }
    if (v > hi) { (*clips)++; return hi; }
    return (int32_t)v;
}

/* qx = sat(roundf(z * inv_sx), -127, 127); the product is a float32 multiply.
 * roundf() rounds half away from zero. Returns -1 for a non-finite input. */
static inline int pv_quantize_input(float z, float inv_sx, int8_t *q, int *clips)
{
    const float r = roundf(z * inv_sx);
    if (!isfinite(r)) return -1;
    if (r > (float)PV_QMAX) { (*clips)++; *q = (int8_t)PV_QMAX; return 0; }
    if (r < -(float)PV_QMAX) { (*clips)++; *q = (int8_t)(-PV_QMAX); return 0; }
    *q = (int8_t)r;
    return 0;
}

#endif /* PV_QUANT_H */
