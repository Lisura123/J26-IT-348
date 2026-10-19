/* Relay safety interface: ABSTRACT, MOCK-ONLY.
 *
 * PHYSICAL ISOLATION IS DISABLED. This module never touches GPIO and never
 * energises, isolates or shuts down hardware. pv_relay_init() accepts only a
 * driver that declares itself a mock (is_mock = 1); a physical driver is
 * refused with PV_RELAY_ERR_PHYSICAL_DISABLED, and defining
 * PV_RELAY_ENABLE_PHYSICAL stops the build. Enabling real actuation needs
 * hardware and electrical safety validation by Member 1 and a separate,
 * explicitly approved change.
 *
 * Software rules (for testing the decision path only):
 *  - an isolation request needs a CONFIRMED_BY_CONFIGURED_RULES fault of an
 *    eligible label (OPEN_CIRCUIT or SHORT_CIRCUIT; never DUST or shading),
 *    taken from the LATEST update of the confirmation state, whose reading
 *    was itself usable evidence for that fault (fresh = 1): a skipped,
 *    invalid, low-confidence or contradicting reading, an older output, or
 *    an expired confirmation is refused with STALE_EVIDENCE,
 *  - and a one-shot operator authorisation (pv_relay_authorize) that matches,
 *  - a repeated request in the same state is a DUPLICATE (driver not called),
 *  - reconnection is manual only (authorised), never automatic,
 *  - a driver failure enters LOCKED_OUT; only an authorised reset leaves it,
 *  - every request is logged in a ring buffer. Portable C99. */
#ifndef PV_RELAY_H
#define PV_RELAY_H

#include <stdint.h>

#include "pv_confirm.h"

#if defined(PV_RELAY_ENABLE_PHYSICAL)
#error "Physical relay actuation is disabled: it needs hardware and electrical safety validation first."
#endif

#define PV_RELAY_LOG_SIZE 16

typedef enum {
    PV_RELAY_STATE_CONNECTED = 0,          /* safe default: software takes no action    */
    PV_RELAY_STATE_ISOLATED_SIMULATED = 1, /* mock driver recorded an isolation          */
    PV_RELAY_STATE_LOCKED_OUT = 2          /* driver failed; state of hardware unknown   */
} pv_relay_state_t;

typedef enum { PV_RELAY_ACT_ISOLATE = 1, PV_RELAY_ACT_RECONNECT = 2, PV_RELAY_ACT_RESET = 3 } pv_relay_action_t;

typedef enum {
    PV_RELAY_OK = 0,
    PV_RELAY_ERR_NULL_ARG = -1,
    PV_RELAY_ERR_PHYSICAL_DISABLED = -2,
    PV_RELAY_ERR_NOT_INITIALIZED = -3,
    PV_RELAY_ERR_NOT_AUTHORIZED = -4,
    PV_RELAY_ERR_NOT_CONFIRMED = -5,
    PV_RELAY_ERR_LABEL_NOT_ELIGIBLE = -6,
    PV_RELAY_ERR_DUPLICATE = -7,
    PV_RELAY_ERR_LOCKED_OUT = -8,
    PV_RELAY_ERR_DRIVER = -9,
    PV_RELAY_ERR_BAD_STATE = -10,
    PV_RELAY_ERR_STALE_EVIDENCE = -11
} pv_relay_result_t;

typedef struct {
    int is_mock;                           /* must be 1                       */
    int (*isolate)(void *ctx);             /* 0 = success                     */
    int (*reconnect)(void *ctx);
    void *ctx;
} pv_relay_driver_t;

typedef struct {
    uint32_t seq;
    pv_relay_action_t action;
    pv_relay_result_t result;
    int label;                             /* fault label for isolate, else -1 */
    pv_relay_state_t state_after;
} pv_relay_log_entry_t;

typedef struct {
    int initialized;
    pv_relay_state_t state;
    pv_relay_driver_t driver;
    uint32_t armed_token;                  /* 0 = not armed                    */
    uint32_t seq;
    pv_relay_log_entry_t log[PV_RELAY_LOG_SIZE];
    uint32_t log_count;                    /* total entries ever written       */
} pv_relay_t;

pv_relay_result_t pv_relay_init(pv_relay_t *r, const pv_relay_driver_t *driver);
/* Arms the next single action with a non-zero operator token. */
pv_relay_result_t pv_relay_authorize(pv_relay_t *r, uint32_t token);
/* conf must be the output of the most recent pv_confirm_update(st, ...). */
pv_relay_result_t pv_relay_request_isolation(pv_relay_t *r, const pv_confirm_t *st, const pv_confirm_output_t *conf,
                                             uint32_t token);
pv_relay_result_t pv_relay_request_reconnect(pv_relay_t *r, uint32_t token);
pv_relay_result_t pv_relay_reset_lockout(pv_relay_t *r, uint32_t token);
/* Most recent first; returns the number copied (<= max and <= PV_RELAY_LOG_SIZE). */
unsigned pv_relay_log_read(const pv_relay_t *r, pv_relay_log_entry_t *dst, unsigned max);
const char *pv_relay_state_name(pv_relay_state_t s);
const char *pv_relay_result_name(pv_relay_result_t res);

#endif /* PV_RELAY_H */
