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

Adafruit_INA219 ina219;
Adafruit_ADS1115 ads;
OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature probes(&oneWire);

// probe addresses (checked by holding each probe)
DeviceAddress PANEL_PROBE = {0x28, 0x43, 0x8B, 0x6D, 0x00, 0x00, 0x00, 0x84};
DeviceAddress AIR_PROBE   = {0x28, 0x3B, 0xFF, 0x70, 0x00, 0x00, 0x00, 0x6A};

// ---------- Kalman filter ----------

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

int readingNumber = 0;
bool calibrated = false;

// ---------- dust ----------

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

// ---------- helpers ----------

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

unsigned long lastRead = 0;

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

  // ---- read the sensors ----
  float voltage = ina219.getBusVoltage_V() + ina219.getShuntVoltage_mV() / 1000.0;
  float current = ina219.getCurrent_mA();

  float lightV = ads.computeVolts(ads.readADC_SingleEnded(0));
  float lux = lightV * 200.0;   // rough value with the 10k resistor on the board

  probes.requestTemperatures();
  float surface = cleanTemp(probes.getTempC(PANEL_PROBE));
  float ambient = cleanTemp(probes.getTempC(AIR_PROBE));

  // ---- calibration: first 20 readings ----
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
      Serial.println("Format: raw -> filtered");
      Serial.println();
    }
    return;
  }

  // ---- Kalman filter ----
  float voltageKf = kalmanUpdate(kfVoltage, voltage);
  float currentKf = kalmanUpdate(kfCurrent, current);
  float surfaceKf = kalmanUpdate(kfSurface, surface);
  float ambientKf = kalmanUpdate(kfAmbient, ambient);

  // power from the filtered values (same as my pipeline)
  float powerKf = voltageKf * currentKf;

  printPair("V", voltage, voltageKf, 3, " V");
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
}