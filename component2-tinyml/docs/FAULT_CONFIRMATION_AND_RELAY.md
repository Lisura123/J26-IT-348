# Fault Confirmation and Relay Safety Interface

**Status:** implemented in software (`firmware/pv_fault/pv_confirm.[ch]`, `pv_relay.[ch]`)
and tested on the host (`tests/host_c/test_runtime.c`). **Non-actuating.** No GPIO, relay,
MOSFET or shutdown code exists. Physical isolation is **disabled** until Member 1 completes
hardware and electrical safety validation and a separate change is explicitly approved.
Automatic isolation has **not** been demonstrated.

## 1. Why a separate layer

The ML output alone never triggers anything. A single reading can be wrong (DUST is the
weakest class, `OPERATION_RESULTS.md`), readings can be skipped by the quality gates, and the
dataset provenance is unresolved. The confirmation layer turns the per-reading predictions
into a diagnostic status with explicit evidence rules. **It is a configurable software rule,
not an independently validated electrical fault confirmation.**

## 2. States reported per reading

| Status | Meaning |
|---|---|
| `NO_FAULT` | usable reading, no fault pending or confirmed |
| `ML_PREDICTED` | first usable reading of a new fault label (ML only) |
| `CONFIRMATION_PENDING` | the same fault label repeated, fewer than `confirm_count` times |
| `CONFIRMED_BY_CONFIGURED_RULES` | the configured rule is met (see 3); stays until recovery |
| `INSUFFICIENT_EVIDENCE` | this reading cannot be used (quality gate, low confidence, contradiction) and no fault is confirmed |

Evidence per reading: `USABLE`, `SKIPPED_BY_QUALITY_GATE`, `LOW_CONFIDENCE`,
`ELECTRICAL_CONTRADICTION`, `STALE`. Events: candidate started / cancelled, confirmed, cleared.

## 3. Rules

1. **Quality-gate awareness / sensor validity:** a reading with status LOW_LIGHT,
   LUX_SATURATED or INVALID_INPUT is not evidence for or against a fault.
2. **Confidence:** a reading counts only if its confidence ≥ `min_confidence`.
3. **Repetition and temporal consistency:** a fault is confirmed after `confirm_count`
   consecutive usable readings with the same label. A different fault label restarts the
   sequence (label flip). More than `max_gap_readings` unusable readings, or a time gap
   longer than `max_interval_ms` between timestamps, cancels a pending sequence.
4. **Persistence and expiry:** up to `max_gap_readings` unusable readings in a row keep a
   confirmed fault (reported, but not *fresh*); one more, or a time gap longer than
   `max_interval_ms`, **expires** it (event `EXPIRED`), and confirming it again needs a
   complete new sequence. Old or invalid readings therefore never keep a fault confirmed.
5. **Recovery and hysteresis:** a confirmed fault clears only after `recovery_count`
   consecutive usable readings without that label; seeing the fault again resets the count.
   The reading that clears it is evaluated as new evidence (NORMAL → `NO_FAULT`, another
   fault label → `ML_PREDICTED` for that label).
6. **Freshness:** each output carries `fresh = 1` only when that very reading is usable
   evidence for the confirmed label, and `reading_seq` identifies the update.
7. **Optional electrical consistency (DISABLED by default):** for OPEN_CIRCUIT
   `current_ma ≤ open_max_current_ma`, for SHORT_CIRCUIT `voltage_v ≤ short_max_voltage_v`.
   A contradicting reading cancels a matching sequence. Thresholds are NaN (off) because
   Member 1 has not provided them; the contract's `rules` block also remains a placeholder.
8. **False-alarm diagnostics:** counters for readings, usable, skipped, low confidence,
   stale, electrical contradictions, candidates started and cancelled (bursts that never
   confirmed), label flips, confirmations, clears and expirations.

## 4. Provisional defaults (conservative, not validated)

| Parameter | Default | Note |
|---|---|---|
| `confirm_count` | 5 | 25 s at the assumed 5 s sampling; minimum 2 enforced |
| `recovery_count` | 5 | hysteresis |
| `max_gap_readings` | 1 | |
| `min_confidence` | 0.90 | |
| `max_interval_ms` | 15,000 | 3 × sampling period |
| `open_max_current_ma`, `short_max_voltage_v` | NaN (disabled) | need Member 1's electrical values |

None of these values has been checked against field data, the electrical design or any
time-to-isolate requirement. They are not tuned on OPERATION or any other data.

## 5. Relay safety interface (mock only)

| Property | Implementation |
|---|---|
| Safe-state representation | `CONNECTED` (software takes no action) is the initial state; `ISOLATED_SIMULATED` only with the mock driver; `LOCKED_OUT` after a driver failure (hardware state unknown) |
| Physical actuation disabled | `pv_relay_init` accepts only a driver with `is_mock = 1`; any other driver → `PHYSICAL_DISABLED`. Defining `PV_RELAY_ENABLE_PHYSICAL` stops compilation (`#error`), tested |
| Explicit authorisation | `pv_relay_authorize(token)` arms exactly one request; the token must match and is consumed whatever the outcome |
| Fault-confirmation dependency | isolation needs `CONFIRMED_BY_CONFIGURED_RULES` and an eligible label: OPEN_CIRCUIT or SHORT_CIRCUIT only; DUST and PARTIAL_SHADING are never eligible |
| Fresh evidence only | the request passes the confirmation state and its latest output; it is refused (`STALE_EVIDENCE`) unless that output is the latest update and `fresh = 1`. A skipped, invalid, low-confidence or disagreeing reading, an older output, a forged status or an expired confirmation cannot trigger isolation |
| Action logging | 16-entry ring buffer: sequence, action, result, label, state after |
| Duplicate prevention | a second isolation (or reconnection) in the same state returns `DUPLICATE` without calling the driver |
| Recovery | reconnection is manual and authorised only; never automatic |
| Failure handling | driver error → `LOCKED_OUT`; all requests refused until an authorised reset; a person must check the real hardware state |

The shadow pipeline (`pv_pipeline.c`) never calls the relay module. The prediction message
carries only the confirmation status; there is no relay field and no isolation
recommendation is sent.

## 6. Before any physical actuation (not started)

Member 1 must provide (`ARCHITECTURE.md` §7.5): relay/MOSFET type and DC rating, which
faults need isolation and the maximum time-to-isolate, existing protection (fuse, current
limit), and actuator ownership. Then: electrical safety review, bench test with the mock
replaced by a validated driver, supervisor approval, and a new explicitly approved change.
