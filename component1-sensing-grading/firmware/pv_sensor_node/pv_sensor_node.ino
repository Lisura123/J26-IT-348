#include <Wire.h>
#include <Adafruit_INA219.h>
#include <Adafruit_ADS1X15.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#define SDA_PIN 21
#define SCL_PIN 22
#define ONE_WIRE_PIN 4
#define DUST_PIN 35
#define RELAY_PIN 26

#define READ_EVERY_MS 5000    // same as the dataset (one reading every 5 s)
#define DUST_WINDOW_MS 30000  // DSM501B needs about 30 s to measure
#define CALIBRATION_READINGS 20

#define WINDOW_SIZE 30        // entropy uses the last 30 readings (2.5 min)
#define MIN_READINGS 10       // start entropy after 10 readings
#define NUM_INDICATORS 5

// noise floor: a change must be bigger than 3 x noise to count as information
#define NOISE_SIGMAS 3.0
#define DUST_NOISE 1.0        // % (dust has no Kalman filter, so a fixed value)

// fixed ranges for the absolute score (first values, tune with real data)
#define TEMP_RISE_BAD 20.0    // C  (0 C = good, 20 C or more = bad)
#define DUST_BAD 30.0         // %  (0 % = good, 30 % or more = bad)

// TEMPORARY: panel (-) is not connected to GND yet, so the INA219 cannot
// measure the bus voltage. The load is a known 120 ohm resistor, so we
// calculate the voltage with Ohm's law: V = I x R.
// Set this to false when the GND wire (panel - to GND) is added.
#define USE_OHMS_LAW_VOLTAGE true
#define LOAD_OHMS 120.0

Adafruit_INA219 ina219;
Adafruit_ADS1115 ads;
OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature probes(&oneWire);

// probe addresses (checked by holding each probe)
DeviceAddress PANEL_PROBE = {0x28, 0x43, 0x8B, 0x6D, 0x00, 0x00, 0x00, 0x84};
DeviceAddress AIR_PROBE   = {0x28, 0x3B, 0xFF, 0x70, 0x00, 0x00, 0x00, 0x6A};

// =====================================================================
// Stage 2: Kalman filter
// =====================================================================

struct Kalman {
  float qRatio;    // Q = qRatio x R
  float q;
  float r;
  float x;         // best guess of the real value
  float p;         // how unsure we are about x
  bool started;

  // for estimating R from the first readings
  float lastReading;
  bool hasLast;
  int jumpCount;
  double jumpSum;
  double jumpSumSq;
};

Kalman kfVoltage = {0.3};
Kalman kfCurrent = {0.3};
Kalman kfSurface = {0.1};
Kalman kfAmbient = {0.1};

// save the jump between this reading and the last one
void addJump(Kalman &k, float z) {
  if (isnan(z)) return;
  if (k.hasLast) {
    double jump = z - k.lastReading;
    k.jumpSum += jump;
    k.jumpSumSq += jump * jump;
    k.jumpCount++;
  }
  k.lastReading = z;
  k.hasLast = true;
}

// same as estimate_noise() in Python: R = (std of jumps / sqrt(2))^2
void finishCalibration(Kalman &k) {
  float r = 0;
  if (k.jumpCount > 1) {
    double n = k.jumpCount;
    double variance = (k.jumpSumSq - k.jumpSum * k.jumpSum / n) / (n - 1);
    r = variance / 2.0;
  }
  if (r < 1e-6) r = 1e-6;   // never 0, or p / (p + r) becomes 0 / 0
  k.r = r;
  k.q = k.qRatio * r;
}

// same as kalman_1d() in Python, one reading at a time
float kalmanUpdate(Kalman &k, float z) {
  if (!k.started) {
    if (isnan(z)) return NAN;
    k.x = z;
    k.p = k.r;
    k.started = true;
    return k.x;
  }

  k.p = k.p + k.q;              // predict: we get a bit more unsure

  if (!isnan(z)) {              // update only if the reading is good
    float gain = k.p / (k.p + k.r);
    k.x = k.x + gain * (z - k.x);
    k.p = (1 - gain) * k.p;
  }
  return k.x;
}

// =====================================================================
// Stage 3: Entropy weighting
// =====================================================================

// the last 30 filtered readings (circular buffer)
float bufPower[WINDOW_SIZE];
float bufVoltage[WINDOW_SIZE];
float bufCurrent[WINDOW_SIZE];
float bufLux[WINDOW_SIZE];
float bufTempRise[WINDOW_SIZE];
float bufDust[WINDOW_SIZE];
int bufCount = 0;   // how many readings are in the buffer
int bufNext = 0;    // where the next reading goes

// indicator names and type (same order as INDICATORS in Python)
const char* IND_NAMES[NUM_INDICATORS] = {"power", "voltage", "current", "temp_rise", "dust"};
const bool IND_BENEFIT[NUM_INDICATORS] = {true, true, true, false, false};

float weights[NUM_INDICATORS];
float absScore[NUM_INDICATORS];   // newest reading on the fixed scale (1 = good)
float healthScore = NAN;

// best healthy reference seen since the ESP32 started
float bestRefPower = 0;
float bestRefCurrent = 0;
float bestRefVoltage = 0;

void addToWindow(float power, float voltage, float current, float lux,
                 float tempRise, float dust) {
  bufPower[bufNext] = power;
  bufVoltage[bufNext] = voltage;
  bufCurrent[bufNext] = current;
  bufLux[bufNext] = lux;
  bufTempRise[bufNext] = tempRise;
  bufDust[bufNext] = dust;

  bufNext = (bufNext + 1) % WINDOW_SIZE;
  if (bufCount < WINDOW_SIZE) bufCount++;
}

// 95th percentile with linear interpolation (same as pandas quantile)
float percentile95(float* values, int n) {
  float sorted[WINDOW_SIZE];
  for (int i = 0; i < n; i++) sorted[i] = values[i];

  // simple insertion sort (n is small)
  for (int i = 1; i < n; i++) {
    float key = sorted[i];
    int j = i - 1;
    while (j >= 0 && sorted[j] > key) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = key;
  }

  float pos = 0.95 * (n - 1);
  int low = (int)pos;
  float frac = pos - low;
  if (low + 1 >= n) return sorted[n - 1];
  return sorted[low] + frac * (sorted[low + 1] - sorted[low]);
}

// keep a value between 0 and 1
float clamp01(float v) {
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

// value / reference, kept between 0 and 1
float ratio(float value, float reference) {
  if (reference < 1e-9) return 0;
  return clamp01(value / reference);
}

// noise of a ratio: 3 x sqrt(R) / reference
float ratioNoise(float r, float reference) {
  if (reference < 1e-9) return 1e9;   // no signal: never informative
  return NOISE_SIGMAS * sqrt(r) / reference;
}

void runEntropy() {
  int n = bufCount;

  float ind[NUM_INDICATORS][WINDOW_SIZE];
  float powerPerIrr[WINDOW_SIZE];
  float currentPerIrr[WINDOW_SIZE];
  float voltages[WINDOW_SIZE];
  float currents[WINDOW_SIZE];

  // order: oldest reading first, newest reading last
  int start = (bufCount < WINDOW_SIZE) ? 0 : bufNext;
  int idx[WINDOW_SIZE];
  for (int i = 0; i < n; i++) idx[i] = (start + i) % WINDOW_SIZE;

  for (int i = 0; i < n; i++) {
    int b = idx[i];
    float irr = bufLux[b] / 120.0;    // W/m2 (data contract: lux / 120)
    if (irr < 1) irr = 1;
    powerPerIrr[i] = bufPower[b] / irr;
    currentPerIrr[i] = bufCurrent[b] / irr;
    voltages[i] = bufVoltage[b];
    currents[i] = bufCurrent[b];
  }

  // --- healthy reference ---
  // 95th percentile of the window, but never lower than the best one seen
  // so a panel that stays faulty does not become the new "healthy"
  float winRefPower = percentile95(powerPerIrr, n);
  float winRefCurrent = percentile95(currentPerIrr, n);
  float winRefVoltage = percentile95(voltages, n);
  if (winRefPower > bestRefPower) bestRefPower = winRefPower;
  if (winRefCurrent > bestRefCurrent) bestRefCurrent = winRefCurrent;
  if (winRefVoltage > bestRefVoltage) bestRefVoltage = winRefVoltage;

  // --- make_indicators() ---
  for (int i = 0; i < n; i++) {
    int b = idx[i];
    ind[0][i] = ratio(powerPerIrr[i], bestRefPower);
    ind[1][i] = ratio(voltages[i], bestRefVoltage);
    ind[2][i] = ratio(currentPerIrr[i], bestRefCurrent);
    ind[3][i] = bufTempRise[b];
    ind[4][i] = bufDust[b];
  }

  // --- absolute score of the newest reading (fixed ranges, 1 = good) ---
  int last = n - 1;
  absScore[0] = ind[0][last];   // ratios are already 0..1
  absScore[1] = ind[1][last];
  absScore[2] = ind[2][last];
  absScore[3] = clamp01((TEMP_RISE_BAD - ind[3][last]) / TEMP_RISE_BAD);
  absScore[4] = clamp01((DUST_BAD - ind[4][last]) / DUST_BAD);

  // --- noise floor for each indicator ---
  // a window range smaller than this is just sensor noise, not information
  float currentRef = percentile95(currents, n);
  float noiseFloor[NUM_INDICATORS];
  noiseFloor[1] = ratioNoise(kfVoltage.r, winRefVoltage);
  noiseFloor[2] = ratioNoise(kfCurrent.r, currentRef);
  noiseFloor[0] = noiseFloor[1] + noiseFloor[2];   // power = V x I
  noiseFloor[3] = NOISE_SIGMAS * sqrt(kfSurface.r + kfAmbient.r);
  noiseFloor[4] = DUST_NOISE;

  // --- normalize() for the weights (window min-max) ---
  for (int j = 0; j < NUM_INDICATORS; j++) {
    float lo = ind[j][0];
    float hi = ind[j][0];
    for (int i = 1; i < n; i++) {
      if (ind[j][i] < lo) lo = ind[j][i];
      if (ind[j][i] > hi) hi = ind[j][i];
    }

    for (int i = 0; i < n; i++) {
      if (hi - lo < noiseFloor[j]) {
        ind[j][i] = 1.0;   // only noise in the window = no information
      } else if (IND_BENEFIT[j]) {
        ind[j][i] = (ind[j][i] - lo) / (hi - lo);
      } else {
        ind[j][i] = (hi - ind[j][i]) / (hi - lo);
      }
    }
  }

  // --- entropy_weights() ---
  float info[NUM_INDICATORS];
  float infoSum = 0;

  for (int j = 0; j < NUM_INDICATORS; j++) {
    // step 1: share of each reading in the column
    double total = 0;
    for (int i = 0; i < n; i++) total += ind[j][i] + 1e-12;

    // step 2: entropy (0 = lots of information, 1 = no information)
    double e = 0;
    for (int i = 0; i < n; i++) {
      double p = (ind[j][i] + 1e-12) / total;
      e -= p * log(p);
    }
    e = e / log(n);

    // step 3: the information part
    info[j] = 1 - e;
    if (info[j] < 0) info[j] = 0;
    infoSum += info[j];
  }

  for (int j = 0; j < NUM_INDICATORS; j++) {
    if (infoSum < 1e-9) {
      weights[j] = 1.0 / NUM_INDICATORS;   // nothing changes: equal weights
    } else {
      weights[j] = info[j] / infoSum;
    }
  }

  // --- health score: adaptive weights x absolute indicator scores ---
  float score = 0;
  for (int j = 0; j < NUM_INDICATORS; j++) {
    score += weights[j] * absScore[j];
  }
  healthScore = score * 100.0;
}

// =====================================================================
// Dust sensor
// =====================================================================

volatile unsigned long lowStart = 0;
volatile unsigned long lowTotal = 0;
unsigned long dustWindowStart = 0;
float dustPercent = -1;   // -1 means "not ready yet" (first 30 s)

void IRAM_ATTR dustChanged() {
  if (digitalRead(DUST_PIN) == LOW) {
    lowStart = micros();
  } else {
    lowTotal += micros() - lowStart;
  }
}

// =====================================================================
// Helpers
// =====================================================================

// DS18B20 sends -127 when the probe is not found, and 85 before the first read
float cleanTemp(float t) {
  if (t == DEVICE_DISCONNECTED_C || t == 85.0) return NAN;
  return t;
}

void printPair(String name, float raw, float filtered, int decimals, String unit) {
  Serial.print(name);
  Serial.print(": ");
  Serial.print(raw, decimals);
  Serial.print(" -> ");
  Serial.print(filtered, decimals);
  Serial.print(unit);
  Serial.print(" | ");
}

int readingNumber = 0;
bool calibrated = false;
float lastTempRise = 0;
unsigned long lastRead = 0;

// =====================================================================
// setup and loop
// =====================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("PV sensor node - Member 1");

  // relay OFF at start (low trigger, so HIGH = OFF = panel connected)
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);

  Wire.begin(SDA_PIN, SCL_PIN);

  if (!ina219.begin()) {
    Serial.println("ERROR: INA219 not found");
  }
  if (!ads.begin(0x48)) {
    Serial.println("ERROR: ADS1115 not found");
  }
  ads.setGain(GAIN_ONE);   // range +/- 4.096 V

  probes.begin();
  probes.requestTemperatures();   // first read is often -127, so do one now
  Serial.print("DS18B20 probes found: ");
  Serial.println(probes.getDeviceCount());

  pinMode(DUST_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(DUST_PIN), dustChanged, CHANGE);
  dustWindowStart = millis();

  if (USE_OHMS_LAW_VOLTAGE) {
    Serial.println("NOTE: voltage is calculated as I x 120 ohm (GND wire not added yet)");
  }
  Serial.println("Type 1 = relay ON (panel cut), 0 = relay OFF");
  Serial.print("Calibrating Kalman filter for ");
  Serial.print(CALIBRATION_READINGS * READ_EVERY_MS / 1000);
  Serial.println(" s - keep the panel still");
  Serial.println();
}

void loop() {
  // relay control from Serial Monitor
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '1') {
      digitalWrite(RELAY_PIN, LOW);
      Serial.println(">> Relay ON - panel cut");
    } else if (c == '0') {
      digitalWrite(RELAY_PIN, HIGH);
      Serial.println(">> Relay OFF - panel connected");
    }
  }

  // dust: every 30 s, work out the low pulse percentage
  if (millis() - dustWindowStart >= DUST_WINDOW_MS) {
    noInterrupts();
    unsigned long low = lowTotal;
    lowTotal = 0;
    interrupts();
    dustPercent = (low / 1000.0) / DUST_WINDOW_MS * 100.0;
    dustWindowStart = millis();
  }

  if (millis() - lastRead < READ_EVERY_MS) return;
  lastRead = millis();
  readingNumber++;

  // ---- Stage 1: read the raw sensors ----
  float current = ina219.getCurrent_mA();
  float shuntV = ina219.getShuntVoltage_mV() / 1000.0;
  float voltage;

  if (USE_OHMS_LAW_VOLTAGE) {
    // V = I x R on the load, plus the small drop on the 0.1 ohm shunt
    voltage = (current / 1000.0) * LOAD_OHMS + shuntV;
  } else {
    // measured directly: bus voltage (VIN- to GND) + shunt voltage
    voltage = ina219.getBusVoltage_V() + shuntV;
  }

  float lightV = ads.computeVolts(ads.readADC_SingleEnded(0));
  float lux = lightV * 200.0;   // rough value with the 10k resistor on the board

  probes.requestTemperatures();
  float surface = cleanTemp(probes.getTempC(PANEL_PROBE));
  float ambient = cleanTemp(probes.getTempC(AIR_PROBE));

  // ---- Kalman calibration: first 20 readings ----
  if (!calibrated) {
    addJump(kfVoltage, voltage);
    addJump(kfCurrent, current);
    addJump(kfSurface, surface);
    addJump(kfAmbient, ambient);

    Serial.print("Calibrating ");
    Serial.print(readingNumber);
    Serial.print("/");
    Serial.print(CALIBRATION_READINGS);
    Serial.print(" | V: ");  Serial.print(voltage, 3);
    Serial.print(" V | I: "); Serial.print(current, 2);
    Serial.println(" mA");

    if (readingNumber >= CALIBRATION_READINGS) {
      finishCalibration(kfVoltage);
      finishCalibration(kfCurrent);
      finishCalibration(kfSurface);
      finishCalibration(kfAmbient);
      calibrated = true;

      Serial.println();
      Serial.println("Noise R found from my device:");
      Serial.print("  voltage R = "); Serial.println(kfVoltage.r, 8);
      Serial.print("  current R = "); Serial.println(kfCurrent.r, 6);
      Serial.print("  surface R = "); Serial.println(kfSurface.r, 6);
      Serial.print("  ambient R = "); Serial.println(kfAmbient.r, 6);
      Serial.println();
    }
    return;
  }

  // ---- Stage 2: Kalman filter ----
  float voltageKf = kalmanUpdate(kfVoltage, voltage);
  float currentKf = kalmanUpdate(kfCurrent, current);
  float surfaceKf = kalmanUpdate(kfSurface, surface);
  float ambientKf = kalmanUpdate(kfAmbient, ambient);
  float powerKf = voltageKf * currentKf;   // same as my pipeline

  printPair(USE_OHMS_LAW_VOLTAGE ? "V(calc)" : "V", voltage, voltageKf, 3, " V");
  printPair("I", current, currentKf, 2, " mA");
  Serial.print("P: ");        Serial.print(powerKf, 4);
  Serial.print(" mW | Light: "); Serial.print(lux, 0);
  Serial.print(" lux | ");
  printPair("Surface", surface, surfaceKf, 2, " C");
  printPair("Ambient", ambient, ambientKf, 2, " C");
  Serial.print("Dust: ");
  if (dustPercent < 0) {
    Serial.println("wait");
  } else {
    Serial.print(dustPercent, 2);
    Serial.println(" %");
  }

  // ---- Stage 3: entropy weighting ----
  float tempRise = surfaceKf - ambientKf;
  if (isnan(tempRise)) tempRise = lastTempRise;   // keep the last good value
  lastTempRise = tempRise;

  float dust = (dustPercent < 0) ? 0 : dustPercent;

  addToWindow(powerKf, voltageKf, currentKf, lux, tempRise, dust);

  if (bufCount < MIN_READINGS) {
    Serial.print("   Entropy: collecting readings ");
    Serial.print(bufCount);
    Serial.print("/");
    Serial.println(MIN_READINGS);
    Serial.println();
    return;
  }

  runEntropy();

  Serial.print("   Weights (window ");
  Serial.print(bufCount);
  Serial.print("): ");
  for (int j = 0; j < NUM_INDICATORS; j++) {
    Serial.print(IND_NAMES[j]);
    Serial.print(" ");
    Serial.print(weights[j], 3);
    if (j < NUM_INDICATORS - 1) Serial.print(", ");
  }
  Serial.println();

  Serial.print("   Indicators (1 = good): ");
  for (int j = 0; j < NUM_INDICATORS; j++) {
    Serial.print(IND_NAMES[j]);
    Serial.print(" ");
    Serial.print(absScore[j], 2);
    if (j < NUM_INDICATORS - 1) Serial.print(", ");
  }
  Serial.print(" | Score: ");
  Serial.print(healthScore, 1);
  Serial.println(" / 100");
  Serial.println();
}