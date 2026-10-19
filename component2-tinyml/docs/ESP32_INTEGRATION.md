# ESP32 Software Integration (Component 2)

**Status (2026-10-16):** software prepared and **tested on the host only**. Nothing has been
built with ESP-IDF, flashed or run on an ESP32. No flash, RAM, stack, heap or latency figure
exists for the board; every such figure is **NOT_EXECUTED**. The board model is unknown
(`ARCHITECTURE.md` §7.6.9), so no target, board or pin is assumed anywhere.

The frozen model is unchanged: `c2-local-v1`, pv8-v1, 8-16-5 tanh, 229 parameters,
`int8-v1`. Generated headers in `firmware/pv_fault/generated/` and the golden vectors in
`tests/host_c/golden_vectors.h` are not edited.

## 1. Layout

```
firmware/
  pv_fault/                 ESP-IDF component "pv_fault" (also a host CMake library)
    CMakeLists.txt  Kconfig
    pv_fault.[ch] pv_mlp_f32.c pv_mlp_int8.c pv_quant.h   frozen inference core (unchanged)
    generated/*.h                                        frozen generated model headers (unchanged)
    pv_api.[ch]        public inference API, init self-check, metadata, errors
    pv_sensor.[ch]     sensor-adapter contract (Section 4)
    pv_confirm.[ch]    non-actuating fault confirmation (FAULT_CONFIRMATION_AND_RELAY.md)
    pv_relay.[ch]      relay interface, MOCK ONLY, physical actuation disabled
    pv_message.[ch]    prediction message pv-msg-1 (MESSAGING.md)
    pv_pipeline.[ch]   shadow-mode pipeline: adapter -> classify -> confirm -> message
    selfcheck/pv_selfcheck.h   generated init self-check vectors (scripts/09_generate_selfcheck.py)
  pv_fault_test/            ESP-IDF component "pv_fault_test": on-device test and benchmark cores
    pv_selftest.[ch]  pv_bench.[ch]  (use tests/host_c/golden_vectors.h)
  apps/
    pv_selftest_app/        golden-vector test application (serial output)
    pv_benchmark_app/       Float32 vs INT8 benchmark application (serial output)
tests/esp_stubs/            host stand-ins for the few ESP-IDF calls (tests only)
```

## 2. Public API (`pv_api.h`)

| Function | Purpose |
|---|---|
| `pv_api_init()` | Runs the self-check once: label names and order, scaler (finite, std > 0), version metadata, 16 reference vectors through Float32 (logits within 1e-4 + 1e-5·\|x\|) and INT8 (output accumulators **bit-exact**). Returns `PV_ERR_MODEL_CHECK` with failure bits (`pv_api_last_check_failures()`) if anything differs. |
| `pv_api_classify(backend, raw[8], &pred)` | `PV_BACKEND_FLOAT32` or `PV_BACKEND_INT8` per call. Fills `input_status` (quality gate), `label` / `label_name` (none when skipped), `confidence`, `prob[5]`, INT8 `input_clips` / `lut_clips`. A skipped reading returns `PV_OK` with `label = -1`: it is neither a fault nor NORMAL. |
| `pv_api_model_info()` | api, model, feature and quantisation versions, contract and weights SHA-256, sizes. |
| `pv_api_strerror`, `pv_backend_name`, `pv_input_status_name`, `pv_prediction_status_name` | Strings for logs and messages. |

Errors: `PV_ERR_NOT_INITIALIZED`, `PV_ERR_NULL_ARG`, `PV_ERR_BAD_BACKEND`, `PV_ERR_MODEL_CHECK`.
Both backends are always linked; Kconfig `PV_FAULT_DEFAULT_BACKEND` (default Float32, the
reference) is for the application's own default. The Float32/INT8 choice is made from
board measurements (`ARCHITECTURE.md` §9.1); nothing is chosen yet.

Example (Member 1's sampling loop, shadow mode):

```c
#include "pv_pipeline.h"
static pv_pipeline_t pl;
pv_confirm_config_t cfg;
pv_confirm_default_config(&cfg);                       /* provisional, conservative */
pv_pipeline_init(&pl, PV_BACKEND_FLOAT32, &cfg, "SITE-A", "ESP32-A01");
/* every 5 s: fill pv_sensor_sample_t from the drivers (Section 4), then */
pv_pipeline_output_t out; char json[768];
pv_pipeline_step(&pl, &sample, record_id, &out, json, sizeof json);
if (out.message_rc == PV_MSG_OK) publish("solar/SITE-A/prediction", json);   /* Member 1's MQTT code */
```

## 3. Build and run (when a board is available)

Requirements: ESP-IDF **v5.x** (the apps use `esp_cpu_get_cycle_count()` and
`CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ`). Not yet tried with any ESP-IDF version.

```bash
cd component2-tinyml/firmware/apps/pv_selftest_app
idf.py set-target <chip>          # esp32 / esp32s3 / esp32c3 ... from Member 1's board
idf.py build
idf.py -p <PORT> flash monitor | tee selftest_console.log
cd ../pv_benchmark_app
idf.py set-target <chip> && idf.py build
idf.py size > size.txt && idf.py size-components > size_components.txt   # flash and static RAM
idf.py -p <PORT> flash monitor | tee benchmark_console.log
```

Expected console lines: `PVINFO {...}`, `PVTEST {...}` (`"result":"PASS"`), `PVTEST_DONE PASS`;
the benchmark adds two `PVBENCH {...}` lines (FLOAT32 and INT8) and `PVBENCH_DONE`.
Summarise a saved log with:

```python
from tinyml import benchmark
report = benchmark.summarize(open("benchmark_console.log").read(), size_report=open("size.txt").read())
```

`measurement_status` is `ESP32_MEASURED` only for an ESP-IDF target name, a passing
self-test and both backends; host logs are `HOST_ONLY`; no log is `NOT_EXECUTED`.

## 4. Sensor-adapter contract (`pv_sensor.h`)

Member 1's drivers deliver **calibrated physical values** with a state per channel
(`VALID`, `MISSING`, `NOT_INSTALLED`, `SENSOR_ERROR`, `SATURATED`, `UNCALIBRATED`), a UTC
timestamp in milliseconds with a "synchronised" flag, and a sequence number.
`pv_sensor_adapt()` produces the eight model inputs:

| Input | Unit | Source | Adapter rule |
|---|---|---|---|
| `voltage_v` | V | INA219 bus voltage | NaN unless VALID and finite |
| `current_ma` | mA | INA219 current | same |
| `power_mw` | mW | **computed** `voltage_v * current_ma` (same sample) | NaN if V or I unusable; never the INA219 power register |
| `panel_temp1_c` | °C | panel DS18B20 #1 | NaN unless VALID |
| `panel_temp2_c` | °C | panel DS18B20 #2: **not installed** on the stated hardware | NaN (`NOT_INSTALLED`); **never copied from probe #1** |
| `ambient_temp_c` | °C | air DS18B20 | NaN unless VALID |
| `irradiance_lux` | lx | TEMT6000 via ADC, **conversion unresolved** | NaN unless VALID (a saturated ADC is `SATURATED`, not a number) |
| `dust_vo_v` | V | GP2Y10 via ADC, **definition unresolved** | NaN unless VALID |

- Any NaN input makes the model's quality gate return **INVALID_INPUT**: the reading is
  skipped and reported, never treated as a fault. `invalid_mask` (bit *i* = input *i*) is
  sent in the message as `sensor_invalid_mask`.
- Range checks stay in the frozen quality gate (`pv_input_status`); the adapter adds none
  and changes none.
- Timestamps before 2020-01-01T00:00:00Z or without the synchronised flag are invalid;
  without a valid UTC time no backend message is built (`PV_MSG_ERR_NO_TIMESTAMP`),
  because the backend requires `timestamp_utc`.
- `pv_ds18b20_state(value, first_conversion)` marks −127 °C, non-finite values and the
  85.0 °C power-on value of the first conversion after power-up/reset as `SENSOR_ERROR`.

**With the stated hardware (one panel probe) every reading is INVALID_INPUT**, so the model
classifies nothing until a second panel DS18B20 is fitted (`ARCHITECTURE.md` §7.6.3).
This is intended: a copied temperature would silently disable shading detection.

### Conversions Member 1 must implement (not done by Component 2)

| Sensor | Conversion / requirement |
|---|---|
| INA219 | Shunt value, calibration register and current LSB giving ≈0.1 mA resolution or finer (OPEN_CIRCUIT median 0.105 mA); bus voltage in V; current sign (reverse flow is gated INVALID). |
| DS18B20 ×3 | Second panel probe on the panel surface; read -127 / CRC errors as `SENSOR_ERROR`; track the first conversion after reset (85.0 °C); 12-bit resolution and conversion time. |
| TEMT6000 | Load resistor, ADC1 pin and attenuation, ADC calibration (`esp_adc_cal`/`adc_cali`), voltage-to-lux equation checked against a reference lux meter, saturation point → `SATURATED`. Must cover the TRAINING range (4,878–100,000 lx). |
| GP2Y10 | LED pulse timing (≈0.32 ms per 10 ms, sample ≈0.28 ms after pulse start), supply and divider (undo in software), offset and averaging exactly as when `dust_vo_v` in the dataset was produced; mounting. |
| Clock | NTP or gateway time before messages are sent; `record_id` unique per reading. |
| Sampling | One reading every 5 s (the confirmation defaults assume it). |

Gate thresholds (LOW_LIGHT 2,400 lx, LUX_SATURATED 100,000 lx) remain placeholders and are
not changed without approval.

## 5. Hardware validation checklist (to run when the board exists)

| # | Step | Pass criterion | Status |
|---|---|---|---|
| H1 | Record board model, chip revision, flash/PSRAM, ESP-IDF version, compiler | recorded in the log (`PVINFO`) | NOT_EXECUTED |
| H2 | Build both apps with `-ffp-contract=off` (component CMake sets it) | builds without warnings | NOT_EXECUTED |
| H3 | Flash `pv_selftest_app` | `PVTEST ... "result":"PASS"`, 0 failures; INT8 bit-exact 105/105 | NOT_EXECUTED |
| H4 | `idf.py size`, `size-components` | flash and static RAM of `pv_fault` recorded | NOT_EXECUTED |
| H5 | Flash `pv_benchmark_app` | `PVBENCH` for FLOAT32 and INT8; `benchmark.summarize` → `ESP32_MEASURED` | NOT_EXECUTED |
| H6 | Heap and stack | heap before/after init, minimum free heap, stack high-water mark recorded | NOT_EXECUTED |
| H7 | Float32 vs INT8 decision | rule of `ARCHITECTURE.md` §9.1 applied to H4–H6 | NOT_EXECUTED |
| H8 | Sensor adapter with real drivers, sensors disconnected | every reading INVALID_INPUT, no label, messages still built | BLOCKED_BY_HARDWARE |
| H9 | Second panel DS18B20 fitted, TEMT6000 calibrated | readings inside TRAINING ranges; gates behave | BLOCKED_BY_HARDWARE |
| H10 | Shadow mode for an agreed period | messages accepted by the backend; no actuation | BLOCKED_BY_HARDWARE |
| H11 | Relay hardware and protection design | Member 1 electrical safety validation | BLOCKED_BY_HARDWARE (physical actuation stays disabled) |

## 6. What the host tests prove and what they do not

- The C modules compile with `-std=c99 -Wall -Wextra -Werror -pedantic` and run under
  AddressSanitizer and UndefinedBehaviorSanitizer (`tests/host_c/test_runtime.c`, 800+ checks).
- Both applications compile and run on the host against `tests/esp_stubs/`
  (`"target":"host-stub"`); the self-test suite passes there (2,483 checks).
- The component builds as a host CMake library.
- **Not proven:** that ESP-IDF builds the component (no ESP-IDF here), that results on
  Xtensa/RISC-V are identical (FPU, `tanhf`/`expf` libraries), any timing or memory
  figure, or any sensor behaviour.
