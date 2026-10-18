/* Mock-only relay safety interface (see pv_relay.h). No GPIO, no actuation. */
#include <stddef.h>
#include <string.h>

#include "pv_relay.h"

#define PV_LABEL_OPEN 3
#define PV_LABEL_SHORT 4

static pv_relay_result_t pv_log(pv_relay_t *r, pv_relay_action_t a, pv_relay_result_t res, int label)
{
    pv_relay_log_entry_t *e = &r->log[r->log_count % PV_RELAY_LOG_SIZE];
    e->seq = ++r->seq;
    e->action = a;
    e->result = res;
    e->label = label;
    e->state_after = r->state;
    r->log_count++;
    return res;
}

pv_relay_result_t pv_relay_init(pv_relay_t *r, const pv_relay_driver_t *driver)
{
    if (r == NULL || driver == NULL) return PV_RELAY_ERR_NULL_ARG;
    memset(r, 0, sizeof(*r));
    r->state = PV_RELAY_STATE_CONNECTED;
    if (driver->is_mock != 1 || driver->isolate == NULL || driver->reconnect == NULL)
        return PV_RELAY_ERR_PHYSICAL_DISABLED;       /* only a mock driver is accepted */
    r->driver = *driver;
    r->initialized = 1;
    return PV_RELAY_OK;
}

pv_relay_result_t pv_relay_authorize(pv_relay_t *r, uint32_t token)
{
    if (r == NULL) return PV_RELAY_ERR_NULL_ARG;
    if (!r->initialized) return PV_RELAY_ERR_NOT_INITIALIZED;
    if (token == 0) return PV_RELAY_ERR_NOT_AUTHORIZED;
    r->armed_token = token;
    return PV_RELAY_OK;
}

/* Consumes the authorisation whatever the outcome (one request per arm). */
static int pv_take_token(pv_relay_t *r, uint32_t token)
{
    int ok = r->armed_token != 0 && r->armed_token == token;
    r->armed_token = 0;
    return ok;
}

pv_relay_result_t pv_relay_request_isolation(pv_relay_t *r, const pv_confirm_t *st, const pv_confirm_output_t *conf,
                                             uint32_t token)
{
    int label;
    if (r == NULL || st == NULL || conf == NULL) return PV_RELAY_ERR_NULL_ARG;
    if (!r->initialized) return PV_RELAY_ERR_NOT_INITIALIZED;
    label = conf->confirmed_label;
    if (!pv_take_token(r, token)) return pv_log(r, PV_RELAY_ACT_ISOLATE, PV_RELAY_ERR_NOT_AUTHORIZED, label);
    if (r->state == PV_RELAY_STATE_LOCKED_OUT) return pv_log(r, PV_RELAY_ACT_ISOLATE, PV_RELAY_ERR_LOCKED_OUT, label);
    if (conf->status != PV_CONF_CONFIRMED_BY_CONFIGURED_RULES || label < 0)
        return pv_log(r, PV_RELAY_ACT_ISOLATE, PV_RELAY_ERR_NOT_CONFIRMED, label);
    if (label != PV_LABEL_OPEN && label != PV_LABEL_SHORT)
        return pv_log(r, PV_RELAY_ACT_ISOLATE, PV_RELAY_ERR_LABEL_NOT_ELIGIBLE, label);
    /* fresh evidence only: the latest update, a usable reading showing the confirmed label */
    if (conf->reading_seq != st->diag.readings || conf->fresh != 1 || conf->evidence != PV_EVID_USABLE ||
        st->confirmed_label != label)
        return pv_log(r, PV_RELAY_ACT_ISOLATE, PV_RELAY_ERR_STALE_EVIDENCE, label);
    if (r->state == PV_RELAY_STATE_ISOLATED_SIMULATED)
        return pv_log(r, PV_RELAY_ACT_ISOLATE, PV_RELAY_ERR_DUPLICATE, label);
    if (r->driver.isolate(r->driver.ctx) != 0) {
        r->state = PV_RELAY_STATE_LOCKED_OUT;
        return pv_log(r, PV_RELAY_ACT_ISOLATE, PV_RELAY_ERR_DRIVER, label);
    }
    r->state = PV_RELAY_STATE_ISOLATED_SIMULATED;
    return pv_log(r, PV_RELAY_ACT_ISOLATE, PV_RELAY_OK, label);
}

pv_relay_result_t pv_relay_request_reconnect(pv_relay_t *r, uint32_t token)
{
    if (r == NULL) return PV_RELAY_ERR_NULL_ARG;
    if (!r->initialized) return PV_RELAY_ERR_NOT_INITIALIZED;
    if (!pv_take_token(r, token)) return pv_log(r, PV_RELAY_ACT_RECONNECT, PV_RELAY_ERR_NOT_AUTHORIZED, -1);
    if (r->state == PV_RELAY_STATE_LOCKED_OUT) return pv_log(r, PV_RELAY_ACT_RECONNECT, PV_RELAY_ERR_LOCKED_OUT, -1);
    if (r->state == PV_RELAY_STATE_CONNECTED) return pv_log(r, PV_RELAY_ACT_RECONNECT, PV_RELAY_ERR_DUPLICATE, -1);
    if (r->driver.reconnect(r->driver.ctx) != 0) {
        r->state = PV_RELAY_STATE_LOCKED_OUT;
        return pv_log(r, PV_RELAY_ACT_RECONNECT, PV_RELAY_ERR_DRIVER, -1);
    }
    r->state = PV_RELAY_STATE_CONNECTED;
    return pv_log(r, PV_RELAY_ACT_RECONNECT, PV_RELAY_OK, -1);
}

/* Leaves LOCKED_OUT for CONNECTED in the software model only; the real
 * hardware state after a driver failure must be checked by a person. */
pv_relay_result_t pv_relay_reset_lockout(pv_relay_t *r, uint32_t token)
{
    if (r == NULL) return PV_RELAY_ERR_NULL_ARG;
    if (!r->initialized) return PV_RELAY_ERR_NOT_INITIALIZED;
    if (!pv_take_token(r, token)) return pv_log(r, PV_RELAY_ACT_RESET, PV_RELAY_ERR_NOT_AUTHORIZED, -1);
    if (r->state != PV_RELAY_STATE_LOCKED_OUT) return pv_log(r, PV_RELAY_ACT_RESET, PV_RELAY_ERR_BAD_STATE, -1);
    r->state = PV_RELAY_STATE_CONNECTED;
    return pv_log(r, PV_RELAY_ACT_RESET, PV_RELAY_OK, -1);
}

unsigned pv_relay_log_read(const pv_relay_t *r, pv_relay_log_entry_t *dst, unsigned max)
{
    unsigned n, i, avail;
    if (r == NULL || dst == NULL) return 0;
    avail = r->log_count < PV_RELAY_LOG_SIZE ? r->log_count : PV_RELAY_LOG_SIZE;
    n = max < avail ? max : avail;
    for (i = 0; i < n; i++) dst[i] = r->log[(r->log_count - 1 - i) % PV_RELAY_LOG_SIZE];
    return n;
}

const char *pv_relay_state_name(pv_relay_state_t s)
{
    switch (s) {
    case PV_RELAY_STATE_CONNECTED: return "CONNECTED";
    case PV_RELAY_STATE_ISOLATED_SIMULATED: return "ISOLATED_SIMULATED";
    case PV_RELAY_STATE_LOCKED_OUT: return "LOCKED_OUT";
    }
    return "UNKNOWN";
}

const char *pv_relay_result_name(pv_relay_result_t res)
{
    switch (res) {
    case PV_RELAY_OK: return "OK";
    case PV_RELAY_ERR_NULL_ARG: return "NULL_ARG";
    case PV_RELAY_ERR_PHYSICAL_DISABLED: return "PHYSICAL_DISABLED";
    case PV_RELAY_ERR_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case PV_RELAY_ERR_NOT_AUTHORIZED: return "NOT_AUTHORIZED";
    case PV_RELAY_ERR_NOT_CONFIRMED: return "NOT_CONFIRMED";
    case PV_RELAY_ERR_LABEL_NOT_ELIGIBLE: return "LABEL_NOT_ELIGIBLE";
    case PV_RELAY_ERR_DUPLICATE: return "DUPLICATE";
    case PV_RELAY_ERR_LOCKED_OUT: return "LOCKED_OUT";
    case PV_RELAY_ERR_DRIVER: return "DRIVER_ERROR";
    case PV_RELAY_ERR_BAD_STATE: return "BAD_STATE";
    case PV_RELAY_ERR_STALE_EVIDENCE: return "STALE_EVIDENCE";
    }
    return "UNKNOWN";
}
