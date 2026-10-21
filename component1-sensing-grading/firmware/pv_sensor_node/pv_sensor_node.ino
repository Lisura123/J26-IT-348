#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <time.h>
#include <math.h>
#include <Adafruit_INA219.h>
#include <Adafruit_ADS1X15.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "secrets.h"   // WIFI_SSID, WIFI_PASSWORD, BACKEND_URL, API_KEY, SITE_ID, DEVICE_ID

#define SDA_PIN 21
#define SCL_PIN 22
#define ONE_WIRE_PIN 4
#define DUST_PIN 35
#define RELAY_PIN 26

// SD card pins (my wiring)
#define SD_CS_PIN 5
#define SD_SCK_PIN 17
#define SD_MISO_PIN 16
#define SD_MOSI_PIN 23
#define SD_SPEED 1000000      // 1 MHz: slow but safe with jumper wires

#define READ_EVERY_MS 2000    // one reading every 2 s
#define DUST_WINDOW_MS 30000  // DSM501B needs about 30 s to measure
#define DUST_WARMUP_MS 60000  // and about 1 min to warm up
#define LOAD_OHMS 120.0       // panel(-) is isolated, so V = I x R
#define SHUNT_OHMS 0.1        // INA219 shunt (R100)
#define TEMP_MISSING -127.0   // same sentinel as the datasets
#define RELAY_SETTLE_MS 500   // power dips for a moment when the relay switches

#define BATCH_SIZE 15         // send 15 readings (30 s) in one request
#define JSON_LINE_MAX 460

// ---------------- Stage 1 settings (same as src/data_quality.py) ----------------
#define SIGN_WINDOW 31        // last 31 readings (~1 min) for the wire-reversed check
#define SPIKE_WINDOW 5        // same as rolling(5) in Python
#define V_PANEL_MAX 6.5       // V, a bit above Voc
#define SHORT_PATH_OHMS 0.6   // shunt 0.1 + wires and relay 0.5

// ---------------- Stage 2 settings (same as src/kalman_filter.py) ----------------
// R = noise variance, measured from my device (log_003.csv, estimate_noise())
// Q = 0.3 R for voltage and current, 0.1 R for temperature
#define R_VOLTAGE 0.000954897
#define R_CURRENT 0.134809
#define R_TEMP1   0.00032673
#define R_AMBIENT 0.000230931

// ---------------- Stage 3 settings (same as src/panel_model.py + notebook 06) ----------------
#define LUX_PER_WM2 120.0           // G (W/m2) = lux / 120
#define TEMP_RISE_PER_WM2 0.03125   // NOCT: (45 - 20) / 800
#define TEMP_RISE_BAD 20.0          // C above a healthy panel
#define DUST_BAD 30.0               // % (DSM501B)

// combined weights (50 % entropy + 50 % equal), from the site datasets (notebook 06)
#define W_POWER   0.237
#define W_VOLTAGE 0.175
#define W_CURRENT 0.166
#define W_TEMP    0.309
#define W_DUST    0.113

// expected current (mA) of a HEALTHY panel into my 120 ohm load
// made with expected_current() in panel_model.py (pvlib single-diode model)
#define N_LUX 19
#define N_TEMP 6
const float TABLE_LUX[N_LUX] = {0, 1000, 2500, 5000, 10000, 15000, 20000, 22500, 25000,
                                27500, 30000, 32500, 35000, 40000, 50000, 60000, 80000,
                                100000, 120000};
const float TABLE_TEMP[N_TEMP] = {15, 25, 35, 45, 55, 65};
const float TABLE_MA[N_TEMP][N_LUX] = {
  {0.000, 1.825, 4.549, 9.050, 17.914, 26.595, 35.057, 38.946, 41.859, 43.610, 44.695, 45.445, 46.007, 46.821, 47.858, 48.538, 49.444, 50.060, 50.527},  // 15 C
  {0.000, 1.834, 4.572, 9.096, 18.004, 26.727, 35.134, 38.641, 40.947, 42.336, 43.251, 43.915, 44.429, 45.195, 46.202, 46.877, 47.789, 48.414, 48.890},  // 25 C
  {0.000, 1.844, 4.595, 9.141, 18.093, 26.855, 35.057, 38.008, 39.820, 40.963, 41.758, 42.356, 42.832, 43.560, 44.542, 45.212, 46.129, 46.763, 47.248},  // 35 C
  {0.000, 1.853, 4.617, 9.187, 18.183, 26.972, 34.721, 37.094, 38.551, 39.521, 40.228, 40.776, 41.222, 41.916, 42.876, 43.543, 44.464, 45.108, 45.601},  // 45 C
  {0.000, 1.862, 4.640, 9.232, 18.272, 27.060, 34.080, 35.974, 37.184, 38.030, 38.669, 39.178, 39.598, 40.265, 41.206, 41.869, 42.796, 43.448, 43.950},  // 55 C
  {0.000, 1.871, 4.663, 9.277, 18.359, 27.076, 33.173, 34.715, 35.748, 36.502, 37.088, 37.564, 37.964, 38.607, 39.531, 40.191, 41.124, 41.784, 42.295},  // 65 C
};

// ---------------- Stage 4 settings (same as src/fuzzy_logic.py) ----------------
#define MIN_GRADING_LUX 500.0    // below this the panel gives < 1 mA: just sensor noise (same as Python)
#define HEALTHY_FROM 70.0
#define DEGRADED_FROM 40.0

Adafruit_INA219 ina219;
Adafruit_ADS1115 ads;
OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature probes(&oneWire);

// "P" sticker probe = panel, the other one = air
DeviceAddress PANEL_PROBE = {0x28, 0x43, 0x8B, 0x6D, 0x00, 0x00, 0x00, 0x84};
DeviceAddress AIR_PROBE   = {0x28, 0x3B, 0xFF, 0x70, 0x00, 0x00, 0x00, 0x6A};

// dust: add up how long the pin stays LOW
volatile unsigned long lowStart = 0;
volatile unsigned long lowTotal = 0;
unsigned long dustWindowStart = 0;
float dustPercent = -1;   // -1 = not ready yet

unsigned long lastRead = 0;
bool relayOn = false;
unsigned long relayChangedAt = 0;

// test mode: grade indoors too (my light sensor stops at ~650 lux)
// readings in test mode get the flag TEST_MODE, so they can be removed later
bool testMode = false;

// SD card
bool sdOk = false;
String logFile;
int bootNumber = 0;
unsigned long readingNumber = 0;

// time and backend
bool timeOk = false;
char batch[BATCH_SIZE][JSON_LINE_MAX];
int batchCount = 0;
unsigned long lastWifiTry = 0;

void IRAM_ATTR dustChanged() {
  if (digitalRead(DUST_PIN) == LOW) {
    lowStart = micros();
  } else {
    lowTotal += micros() - lowStart;
  }
}

// =====================================================================
// Stage 1: data quality
// =====================================================================

char qualityFlag[120];

void addFlag(const char* name) {
  // same as add_flag() in Python: join flags with "|"
  if (strcmp(qualityFlag, "OK") == 0) {
    strncpy(qualityFlag, name, sizeof(qualityFlag) - 1);
    qualityFlag[sizeof(qualityFlag) - 1] = '\0';
  } else {
    strncat(qualityFlag, "|", sizeof(qualityFlag) - strlen(qualityFlag) - 1);
    strncat(qualityFlag, name, sizeof(qualityFlag) - strlen(qualityFlag) - 1);
  }
}

// a small window that keeps the last N values
struct Window {
  float values[SIGN_WINDOW];
  int size;
  int count;
  int next;

  void begin(int n) { size = n; count = 0; next = 0; }

  void add(float v) {
    values[next] = v;
    next = (next + 1) % size;
    if (count < size) count++;
  }

  float median() {
    if (count == 0) return 0;
    float tmp[SIGN_WINDOW];
    for (int i = 0; i < count; i++) tmp[i] = values[i];
    // insertion sort (the window is small)
    for (int i = 1; i < count; i++) {
      float key = tmp[i];
      int j = i - 1;
      while (j >= 0 && tmp[j] > key) { tmp[j + 1] = tmp[j]; j--; }
      tmp[j + 1] = key;
    }
    if (count % 2 == 1) return tmp[count / 2];
    return (tmp[count / 2 - 1] + tmp[count / 2]) / 2.0;
  }

  // a spike is far from its neighbours (more than 50 %, and more than minJump)
  float fixSpike(float value, float minJump, const char* flagName) {
    add(value);
    float med = median();
    float jump = fabs(value - med);
    if (count >= 3 && jump > 0.5 * fabs(med) && jump > minJump) {
      addFlag(flagName);
      return med;
    }
    return value;
  }
};

Window signWindow;
Window spikeI, spikeLux;

// =====================================================================
// Stage 2: Kalman filter (same as kalman_1d() in Python)
// =====================================================================

struct Kalman {
  float q, r;
  float x = 0, p = 1;
  bool started = false;

  void begin(float r_, float ratio) { r = r_; q = ratio * r_; started = false; }

  // valid = false: the reading is missing, so we only predict
  float update(float z, bool valid) {
    if (!started) {
      if (!valid) return NAN;
      x = z;
      p = r;
      started = true;
      return x;
    }
    p = p + q;
    if (valid) {
      float k = p / (p + r);   // Kalman gain (0 to 1)
      x = x + k * (z - x);     // move the guess toward the reading
      p = (1 - k) * p;         // we are now more sure
    }
    return x;
  }
};

Kalman kfVoltage, kfCurrent, kfTemp1, kfAmbient;

// =====================================================================
// Stage 3: compare with a healthy panel (physics model) + weights
// =====================================================================

float clip01(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

int findIndex(const float* points, int n, float x) {
  // index i so that points[i] <= x < points[i+1]
  if (x <= points[0]) return 0;
  for (int i = 0; i < n - 1; i++) {
    if (x < points[i + 1]) return i;
  }
  return n - 2;
}

float expectedCurrent(float lux, float panelTemp) {
  // bilinear interpolation in the table (light and temperature)
  if (isnan(panelTemp)) panelTemp = 25.0;   // same as Python (nan -> 25 C)
  if (lux < 0) lux = 0;
  if (lux > TABLE_LUX[N_LUX - 1]) lux = TABLE_LUX[N_LUX - 1];
  if (panelTemp < TABLE_TEMP[0]) panelTemp = TABLE_TEMP[0];
  if (panelTemp > TABLE_TEMP[N_TEMP - 1]) panelTemp = TABLE_TEMP[N_TEMP - 1];

  int li = findIndex(TABLE_LUX, N_LUX, lux);
  int ti = findIndex(TABLE_TEMP, N_TEMP, panelTemp);
  float fl = (lux - TABLE_LUX[li]) / (TABLE_LUX[li + 1] - TABLE_LUX[li]);
  float ft = (panelTemp - TABLE_TEMP[ti]) / (TABLE_TEMP[ti + 1] - TABLE_TEMP[ti]);

  float a = TABLE_MA[ti][li] * (1 - fl) + TABLE_MA[ti][li + 1] * fl;
  float b = TABLE_MA[ti + 1][li] * (1 - fl) + TABLE_MA[ti + 1][li + 1] * fl;
  return a * (1 - ft) + b * ft;
}

// Stage 3 results
float expI, expV;
float powerRatio, voltageRatio, currentRatio, tempRise, dustLevel;
float scoreTemp, scoreDust;
float weightedScore;   // S, 0 - 100

void runStage3(float voltage, float current, float power, float lux,
               float surface, float ambient) {
  // a healthy panel at the same light and temperature
  expI = expectedCurrent(lux, surface);
  if (expI < 0.1) expI = 0.1;                       // same as np.clip(.., 0.1)
  expV = expI / 1000.0 * (LOAD_OHMS + SHUNT_OHMS);

  // 1 = as good as a healthy panel
  powerRatio   = clip01(power / (expI * expV));
  voltageRatio = clip01(voltage / expV);
  currentRatio = clip01(current / expI);

  // temp rise ABOVE what a healthy panel gets in this sunlight
  if (isnan(surface) || isnan(ambient)) tempRise = NAN;
  else tempRise = surface - ambient - lux / LUX_PER_WM2 * TEMP_RISE_PER_WM2;

  dustLevel = dustPercent >= 0 ? dustPercent : NAN;

  // scores: 1 = good, 0 = bad
  scoreTemp = isnan(tempRise) ? NAN : clip01((TEMP_RISE_BAD - tempRise) / TEMP_RISE_BAD);
  scoreDust = isnan(dustLevel) ? NAN : clip01((DUST_BAD - dustLevel) / DUST_BAD);

  // weighted score (a missing indicator is skipped, like pandas sum())
  weightedScore = W_POWER * powerRatio + W_VOLTAGE * voltageRatio + W_CURRENT * currentRatio;
  if (!isnan(scoreTemp)) weightedScore += W_TEMP * scoreTemp;
  if (!isnan(scoreDust)) weightedScore += W_DUST * scoreDust;
  weightedScore *= 100.0;
}

// =====================================================================
// Stage 4: fuzzy logic (same rules as src/fuzzy_logic.py)
// =====================================================================

float trap(float x, float a, float b, float c, float d) {
  // trapezoid: 0 before a, rises a->b, 1 from b to c, falls c->d
  if (isnan(x)) return 0;
  if (x >= b && x <= c) return 1;
  if (b > a && x >= a && x < b) return (x - a) / (b - a);
  if (d > c && x > c && x <= d) return (d - x) / (d - c);
  return 0;
}

float tri(float x, float a, float b, float c) { return trap(x, a, b, b, c); }

float fuzzyGrade(float S, float E, float T, float D) {
  // ---- fuzzify ----
  float sLow = trap(S, 0, 0, 30, 50), sMed = tri(S, 30, 55, 80), sHigh = trap(S, 65, 85, 100, 100);
  float eBad = trap(E, 0, 0, 0.3, 0.5), eOk = tri(E, 0.3, 0.6, 0.85), eGood = trap(E, 0.7, 0.9, 1, 1);
  float tHot = trap(T, 0, 0, 0.25, 0.5), tNormal = trap(T, 0.25, 0.5, 1, 1);
  float dDirty = trap(D, 0, 0, 0.4, 0.6), dClean = trap(D, 0.4, 0.6, 1, 1);

  // ---- rules (AND = min) ----
  float r1 = eBad;                                          // R1 -> FAULTY
  float r2 = tHot;                                          // R2 -> FAULTY
  float r3 = fminf(sLow, 1 - eGood);                        // R3 -> FAULTY
  float r4 = fminf(sMed, 1 - eBad);                         // R4 -> DEGRADED
  float r5 = fminf(dDirty, 1 - eBad);                       // R5 -> DEGRADED
  float r6 = eOk;                                           // R6 -> DEGRADED
  float r7 = fminf(fminf(sHigh, eGood), fminf(tNormal, dClean));   // R7 -> HEALTHY

  // ---- combine rules of the same output (OR = max) ----
  float faulty = fmaxf(r1, fmaxf(r2, r3));
  float degraded = fmaxf(r4, fmaxf(r5, r6));
  float healthy = r7;

  // ---- defuzzify (centroid over 0..100) ----
  float sumMu = 0, sumMuY = 0;
  for (int y = 0; y <= 100; y++) {
    float mu = fmaxf(fminf(faulty, trap(y, 0, 0, 20, 40)),
               fmaxf(fminf(degraded, tri(y, 30, 50, 70)),
                     fminf(healthy, trap(y, 60, 80, 100, 100))));
    sumMu += mu;
    sumMuY += mu * y;
  }
  return sumMu > 1e-6 ? sumMuY / sumMu : S;
}

// Stage 4 results
float healthScore;          // NAN = not graded
char healthStatus[12];
const char* notGradedReason = "";

void runStage4(float lux) {
  float minLux = testMode ? 0.0 : MIN_GRADING_LUX;

  if (lux < minLux) {
    healthScore = NAN;
    notGradedReason = "low light";
  } else if (isnan(scoreDust)) {
    healthScore = NAN;
    notGradedReason = "dust sensor warming up";
  } else {
    float E = fminf(voltageRatio, currentRatio);   // electrical part
    healthScore = fuzzyGrade(weightedScore, E, scoreTemp, scoreDust);
    notGradedReason = "";
  }

  if (isnan(healthScore))                strcpy(healthStatus, "NOT_GRADED");
  else if (healthScore >= HEALTHY_FROM)  strcpy(healthStatus, "HEALTHY");
  else if (healthScore >= DEGRADED_FROM) strcpy(healthStatus, "DEGRADED");
  else                                   strcpy(healthStatus, "FAULTY");
}

// =====================================================================
// WiFi and time
// =====================================================================

void startWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("WiFi: connecting");
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi OK, ESP32 IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi: not connected (will try again, readings still go to SD)");
  }
  lastWifiTry = millis();
}

void keepWiFi() {
  // try again every 30 s if the hotspot went away
  if (WiFi.status() != WL_CONNECTED && millis() - lastWifiTry > 30000) {
    Serial.println("WiFi: trying again...");
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    lastWifiTry = millis();
  }
}

void startTime() {
  configTime(0, 0, "pool.ntp.org", "time.google.com");   // UTC
  struct tm t;
  if (getLocalTime(&t, 10000)) {
    timeOk = true;
    Serial.println("Time OK (NTP)");
  } else {
    Serial.println("Time: no NTP yet (backend sending waits for it)");
  }
}

bool nowUtc(char* out, size_t size) {
  time_t now = time(nullptr);
  if (now < 1700000000) return false;   // clock not set yet
  timeOk = true;
  struct tm t;
  gmtime_r(&now, &t);
  strftime(out, size, "%Y-%m-%dT%H:%M:%SZ", &t);
  return true;
}

// =====================================================================
// SD card (RAW readings, so Python can run the same pipeline later)
// =====================================================================

void startSD() {
  SPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
  if (!SD.begin(SD_CS_PIN, SPI, SD_SPEED)) {
    Serial.println("ERROR: SD card not found (no backup on SD)");
    return;
  }

  for (int n = 1; n < 1000; n++) {
    char name[16];
    sprintf(name, "/log_%03d.csv", n);
    if (!SD.exists(name)) {
      logFile = name;
      bootNumber = n;
      break;
    }
  }

  File f = SD.open(logFile, FILE_WRITE);
  if (!f) {
    Serial.println("ERROR: cannot create the log file");
    return;
  }
  f.println("record_id,timestamp_utc,uptime_s,voltage_v,current_ma,power_mw,panel_temp1_c,"
            "ambient_temp_c,irradiance_lux,dust_lpo_pct,voltage_calculated,relay_on");
  f.close();

  sdOk = true;
  Serial.print("SD card OK, saving to ");
  Serial.println(logFile);
}

void saveToSD(const char* recordId, const char* timeText, float voltage, float current,
              float power, float lux, float surface, float ambient) {
  if (!sdOk) return;

  unsigned long sinceRelay = millis() - relayChangedAt;
  if (sinceRelay < RELAY_SETTLE_MS) delay(RELAY_SETTLE_MS - sinceRelay);

  char dust[12] = "";   // missing = empty, never 0
  if (dustPercent >= 0) snprintf(dust, sizeof(dust), "%.2f", dustPercent);

  char line[220];
  snprintf(line, sizeof(line), "%s,%s,%.1f,%.4f,%.2f,%.4f,%.4f,%.4f,%.1f,%s,1,%d",
           recordId, timeText, millis() / 1000.0,
           voltage, current, power, surface, ambient, lux, dust, relayOn ? 1 : 0);

  File f = SD.open(logFile, FILE_APPEND);
  if (!f) {
    Serial.println("ERROR: SD write failed");
    return;
  }
  f.println(line);
  f.close();
}

// =====================================================================
// backend (FILTERED readings + quality_flag + health_score + health_status)
// =====================================================================

void addToBatch(const char* recordId, const char* timeText, float voltage, float current,
                float power, float lux, float surface, float ambient) {
  // a missing value is LEFT OUT, never sent as 0
  char* p = batch[batchCount];
  int n = snprintf(p, JSON_LINE_MAX,
    "{\"timestamp_utc\":\"%s\",\"record_id\":\"%s\",\"site_id\":\"%s\",\"device_id\":\"%s\","
    "\"quality_flag\":\"%s\",\"health_status\":\"%s\",\"voltage_v\":%.4f,\"current_ma\":%.2f,"
    "\"power_mw\":%.4f,\"irradiance_lux\":%.1f",
    timeText, recordId, SITE_ID, DEVICE_ID, qualityFlag, healthStatus,
    voltage, current, power, lux);
  if (!isnan(surface))     n += snprintf(p + n, JSON_LINE_MAX - n, ",\"panel_temp1_c\":%.4f", surface);
  if (!isnan(ambient))     n += snprintf(p + n, JSON_LINE_MAX - n, ",\"ambient_temp_c\":%.4f", ambient);
  if (dustPercent >= 0)    n += snprintf(p + n, JSON_LINE_MAX - n, ",\"dust_lpo_pct\":%.2f", dustPercent);
  if (!isnan(healthScore)) n += snprintf(p + n, JSON_LINE_MAX - n, ",\"health_score\":%.2f", healthScore);
  snprintf(p + n, JSON_LINE_MAX - n, "}");
  batchCount++;
}

void sendBatch() {
  if (batchCount == 0) return;
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("   Backend: no WiFi, this batch is only on SD");
    batchCount = 0;
    return;
  }

  String body = "[";
  for (int i = 0; i < batchCount; i++) {
    if (i > 0) body += ",";
    body += batch[i];
  }
  body += "]";

  HTTPClient http;
  http.begin(String(BACKEND_URL) + "/api/v1/ingest/health");
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", API_KEY);
  http.setTimeout(10000);
  int code = http.POST(body);
  String reply = http.getString();
  http.end();

  Serial.print("   Backend: sent ");
  Serial.print(batchCount);
  Serial.print(" | HTTP ");
  Serial.print(code);
  Serial.print(" | ");
  Serial.println(reply);

  batchCount = 0;
}

// =====================================================================
// relay
// =====================================================================

void setRelay(bool on) {
  // low trigger: LOW = ON = panel cut, HIGH = OFF = panel connected
  digitalWrite(RELAY_PIN, on ? LOW : HIGH);
  relayOn = on;
  relayChangedAt = millis();
  Serial.println(on ? ">> Relay ON - panel cut" : ">> Relay OFF - panel connected");
}

// =====================================================================
// main
// =====================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);   // relay OFF = panel connected

  Wire.begin(SDA_PIN, SCL_PIN);
  if (!ina219.begin()) Serial.println("ERROR: INA219 not found");
  if (!ads.begin(0x48)) Serial.println("ERROR: ADS1115 not found");
  ads.setGain(GAIN_ONE);   // +/- 4.096 V

  probes.begin();
  probes.setWaitForConversion(false);
  probes.requestTemperatures();

  pinMode(DUST_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(DUST_PIN), dustChanged, CHANGE);

  // Stage 1 windows
  signWindow.begin(SIGN_WINDOW);
  spikeI.begin(SPIKE_WINDOW);
  spikeLux.begin(SPIKE_WINDOW);

  // Stage 2 filters
  kfVoltage.begin(R_VOLTAGE, 0.3);
  kfCurrent.begin(R_CURRENT, 0.3);
  kfTemp1.begin(R_TEMP1, 0.1);
  kfAmbient.begin(R_AMBIENT, 0.1);

  Serial.println();
  Serial.println("=============================================");
  Serial.println("  PV sensor node - Member 1 (full pipeline)");
  Serial.println("  Serial: 1 = relay ON, 0 = relay OFF, t = test mode");
  Serial.println("=============================================");
  startSD();
  startWiFi();
  startTime();
  Serial.print("Backend: ");
  Serial.println(BACKEND_URL);
  Serial.println();

  dustWindowStart = millis();
}

void loop() {
  keepWiFi();

  if (Serial.available()) {
    char c = Serial.read();
    if (c == '1') setRelay(true);
    else if (c == '0') setRelay(false);
    else if (c == 't') {
      testMode = !testMode;
      Serial.println(testMode ? ">> TEST MODE ON - grading also below 500 lux (flag TEST_MODE)"
                              : ">> TEST MODE OFF - normal grading");
    }
  }

  // dust: every 30 s (windows that start in the warm-up are not used)
  if (millis() - dustWindowStart >= DUST_WINDOW_MS) {
    noInterrupts();
    unsigned long low = lowTotal;
    lowTotal = 0;
    interrupts();
    if (dustWindowStart >= DUST_WARMUP_MS) {
      dustPercent = (low / 1000.0) / DUST_WINDOW_MS * 100.0;
    }
    dustWindowStart = millis();
  }

  if (millis() - lastRead < READ_EVERY_MS) return;
  lastRead = millis();
  readingNumber++;

  // ---------------- read the sensors (RAW) ----------------
  float rawCurrent = ina219.getCurrent_mA();
  if (rawCurrent < 0.5 && rawCurrent > -0.5) rawCurrent = 0.0;   // noise around 0
  float rawVoltage = rawCurrent / 1000.0 * (LOAD_OHMS + SHUNT_OHMS);
  float rawPower = rawVoltage * rawCurrent;

  float lightV = ads.computeVolts(ads.readADC_SingleEnded(0));
  float rawLux = lightV * 200.0;   // rough value with the 10k resistor on the board

  float rawSurface = probes.getTempC(PANEL_PROBE);
  float rawAmbient = probes.getTempC(AIR_PROBE);
  probes.requestTemperatures();
  if (rawSurface == DEVICE_DISCONNECTED_C) rawSurface = TEMP_MISSING;
  if (rawAmbient == DEVICE_DISCONNECTED_C) rawAmbient = TEMP_MISSING;

  char recordId[24];
  sprintf(recordId, "M1-DEV-%03d-%07lu", bootNumber, readingNumber);
  char timeText[24] = "";
  bool haveTime = nowUtc(timeText, sizeof(timeText));

  // SD gets the RAW reading
  saveToSD(recordId, timeText, rawVoltage, rawCurrent, rawPower, rawLux, rawSurface, rawAmbient);

  // ---------------- Stage 1: data quality ----------------
  strcpy(qualityFlag, "OK");

  // temperature: -127 is not a real value
  bool surfaceOk = rawSurface != TEMP_MISSING;
  bool ambientOk = rawAmbient != TEMP_MISSING;
  if (!surfaceOk) addFlag("TEMP1_SENTINEL_-127");
  if (!ambientOk) addFlag("AMBIENT_TEMP_SENTINEL_-127");

  // wire reversed: the median of the last minute is negative
  float current = rawCurrent;
  signWindow.add(rawCurrent);
  if (signWindow.median() < 0) {
    addFlag("CURRENT_SIGN_REVERSED");
    current = -current;
  }
  if (current < 0) current = 0.0;   // a small negative left is just noise

  // spikes (one reading far from its neighbours)
  // on my device V = I x R, so we check the current only and then
  // calculate the voltage from the fixed current (they always match)
  float before = current;
  current = spikeI.fixSpike(current, 3.0, "SPIKE_CURRENT");
  if (current != before) addFlag("SPIKE_VOLTAGE");   // the voltage jumped with it
  float voltage = current / 1000.0 * (LOAD_OHMS + SHUNT_OHMS);
  float lux = spikeLux.fixSpike(rawLux, 1000.0, "SPIKE_IRRADIANCE");

  // load shorted: a voltage above Voc is impossible
  if (voltage > V_PANEL_MAX) {
    addFlag("LOAD_SHORTED");
    voltage = current / 1000.0 * SHORT_PATH_OHMS;
  }

  // mark test readings, so the backend data can be cleaned later
  if (testMode) addFlag("TEST_MODE");

  // ---------------- Stage 2: Kalman filter ----------------
  voltage = kfVoltage.update(voltage, true);
  current = kfCurrent.update(current, true);
  float surface = kfTemp1.update(rawSurface, surfaceOk);     // missing -> predicted
  float ambient = kfAmbient.update(rawAmbient, ambientOk);
  float power = voltage * current;   // power again, from the filtered values

  // ---------------- Stage 3: indicators + weighted score ----------------
  runStage3(voltage, current, power, lux, surface, ambient);

  // ---------------- Stage 4: fuzzy grade ----------------
  runStage4(lux);

  // ---------------- send to the backend ----------------
  if (haveTime) {
    addToBatch(recordId, timeText, voltage, current, power, lux, surface, ambient);
    if (batchCount >= BATCH_SIZE) sendBatch();
  }

  // ---------------- Serial Monitor ----------------
  // line 1: the readings
  Serial.print(timeText[0] ? timeText : "no-time");
  Serial.print(" |V: ");         Serial.print(voltage, 3);
  Serial.print(" V | P: ");      Serial.print(power, 2);
  Serial.print(" mW | Light: "); Serial.print(lux, 0);
  Serial.print(" lux | Surface: ");
  if (isnan(surface)) Serial.print("--"); else Serial.print(surface, 2);
  Serial.print(" C | Ambient: ");
  if (isnan(ambient)) Serial.print("--"); else Serial.print(ambient, 2);
  Serial.print(" C | Dust: ");
  if (dustPercent < 0) Serial.print("warming up"); else { Serial.print(dustPercent, 2); Serial.print(" %"); }
  Serial.print(" | Relay: ");    Serial.print(relayOn ? "ON" : "OFF");
  Serial.println(" | ");

  // line 2: the pipeline (Stage 1 flag, Kalman, Stage 3, Stage 4)
  Serial.print("Flag: ");        Serial.print(qualityFlag);
  Serial.print(" | I raw: ");    Serial.print(rawCurrent, 1);
  Serial.print(" -> KF: ");      Serial.print(current, 2);
  Serial.print(" mA |Stage 3: ratios P/V/I ");
  Serial.print(powerRatio, 2);   Serial.print(" / ");
  Serial.print(voltageRatio, 2); Serial.print(" / ");
  Serial.print(currentRatio, 2);
  Serial.print(" |S = ");        Serial.print(weightedScore, 1);
  Serial.print("   Stage 4: ");
  if (isnan(healthScore)) {
    Serial.print("NOT_GRADED (");
    Serial.print(notGradedReason);
    Serial.println(")");
  } else {
    Serial.print("health ");
    Serial.print(healthScore, 1);
    Serial.print(" -> ");
    Serial.println(healthStatus);
  }
}