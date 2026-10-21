#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <time.h>
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
#define TEMP_MISSING -127.0   // same sentinel as the datasets
#define RELAY_SETTLE_MS 500   // power dips for a moment when the relay switches

#define BATCH_SIZE 15         // send 15 readings (30 s) in one request
#define JSON_LINE_MAX 360

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
unsigned long sentTotal = 0, acceptedTotal = 0;

void IRAM_ATTR dustChanged() {
  if (digitalRead(DUST_PIN) == LOW) {
    lowStart = micros();
  } else {
    lowTotal += micros() - lowStart;
  }
}

// ---------------- WiFi and time ----------------

void startWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(200);

  // 1) which networks can the ESP32 see? (only 2.4 GHz)
  Serial.println("WiFi scan:");
  int found = WiFi.scanNetworks();
  bool seen = false;
  for (int i = 0; i < found; i++) {
    Serial.print("   ");
    Serial.print(WiFi.SSID(i));
    Serial.print("  (");
    Serial.print(WiFi.RSSI(i));
    Serial.println(" dBm)");
    if (WiFi.SSID(i) == String(WIFI_SSID)) seen = true;
  }
  Serial.println(seen ? "   -> my hotspot is in the list"
                      : "   -> my hotspot is NOT in the list (Maximize Compatibility? name in secrets.h?)");

  // 2) connect
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("WiFi: connecting");
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  wl_status_t s = WiFi.status();
  if (s == WL_CONNECTED) {
    Serial.print("WiFi OK, ESP32 IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.print("WiFi: not connected, status = ");
    Serial.print(s);
    if (s == WL_NO_SSID_AVAIL) Serial.println(" (network not found)");
    else if (s == WL_CONNECT_FAILED) Serial.println(" (wrong password?)");
    else if (s == WL_DISCONNECTED) Serial.println(" (disconnected / still trying)");
    else Serial.println();
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
  configTime(0, 0, "pool.ntp.org", "time.google.com");   // UTC, no daylight saving
  struct tm t;
  if (getLocalTime(&t, 10000)) {
    timeOk = true;
    Serial.println("Time OK (NTP)");
  } else {
    Serial.println("Time: no NTP yet (backend sending waits for it)");
  }
}

bool nowUtc(char* out, size_t size) {
  // "2026-10-21T12:34:56Z", or false if the time is not known yet
  time_t now = time(nullptr);
  if (now < 1700000000) return false;   // clock not set yet
  timeOk = true;
  struct tm t;
  gmtime_r(&now, &t);
  strftime(out, size, "%Y-%m-%dT%H:%M:%SZ", &t);
  return true;
}

// ---------------- SD card ----------------

void startSD() {
  SPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
  if (!SD.begin(SD_CS_PIN, SPI, SD_SPEED)) {
    Serial.println("ERROR: SD card not found (no backup on SD)");
    return;
  }

  // a new file every time the ESP32 starts: /log_001.csv, /log_002.csv ...
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

  // the relay coil makes the power dip for a moment: wait before writing
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

// ---------------- backend ----------------

void addToBatch(const char* recordId, const char* timeText, float voltage, float current,
                float power, float lux, float surface, float ambient) {
  // one JSON record; a missing value is LEFT OUT, never sent as 0
  char* p = batch[batchCount];
  int n = snprintf(p, JSON_LINE_MAX,
    "{\"timestamp_utc\":\"%s\",\"record_id\":\"%s\",\"site_id\":\"%s\",\"device_id\":\"%s\","
    "\"quality_flag\":\"OK\",\"voltage_v\":%.4f,\"current_ma\":%.2f,\"power_mw\":%.4f,"
    "\"irradiance_lux\":%.1f",
    timeText, recordId, SITE_ID, DEVICE_ID, voltage, current, power, lux);
  if (surface != TEMP_MISSING) n += snprintf(p + n, JSON_LINE_MAX - n, ",\"panel_temp1_c\":%.4f", surface);
  if (ambient != TEMP_MISSING) n += snprintf(p + n, JSON_LINE_MAX - n, ",\"ambient_temp_c\":%.4f", ambient);
  if (dustPercent >= 0)        n += snprintf(p + n, JSON_LINE_MAX - n, ",\"dust_lpo_pct\":%.2f", dustPercent);
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

  sentTotal += batchCount;
  Serial.print("   Backend: sent ");
  Serial.print(batchCount);
  Serial.print(" | HTTP ");
  Serial.print(code);
  Serial.print(" | ");
  Serial.println(reply);
  if (code == 200 && reply.indexOf("\"rejected\":0") >= 0) acceptedTotal += batchCount;

  batchCount = 0;
}

// ---------------- relay ----------------

void setRelay(bool on) {
  // low trigger: LOW = ON = panel cut, HIGH = OFF = panel connected
  digitalWrite(RELAY_PIN, on ? LOW : HIGH);
  relayOn = on;
  relayChangedAt = millis();
  Serial.println(on ? ">> Relay ON - panel cut" : ">> Relay OFF - panel connected");
}

// ---------------- main ----------------

void setup() {
  Serial.begin(115200);
  delay(1000);

  // relay OFF at start (low trigger, so HIGH = OFF = panel connected)
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);

  Wire.begin(SDA_PIN, SCL_PIN);
  if (!ina219.begin()) Serial.println("ERROR: INA219 not found");
  if (!ads.begin(0x48)) Serial.println("ERROR: ADS1115 not found");
  ads.setGain(GAIN_ONE);   // +/- 4.096 V

  probes.begin();
  probes.setWaitForConversion(false);   // do not block the loop
  probes.requestTemperatures();

  pinMode(DUST_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(DUST_PIN), dustChanged, CHANGE);

  Serial.println();
  Serial.println("=============================================");
  Serial.println("  PV sensor node - Member 1");
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

  // relay control from Serial Monitor
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '1') setRelay(true);
    else if (c == '0') setRelay(false);
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

  // INA219: current is measured, voltage is calculated (V = I x R + shunt)
  float current = ina219.getCurrent_mA();
  if (current < 0.5 && current > -0.5) current = 0.0;   // noise around 0
  float voltage = current / 1000.0 * LOAD_OHMS + ina219.getShuntVoltage_mV() / 1000.0;
  if (current == 0.0) voltage = 0.0;
  float power = voltage * current;

  // TEMT6000 through ADS1115 A0
  float lightV = ads.computeVolts(ads.readADC_SingleEnded(0));
  float lux = lightV * 200.0;   // rough value with the 10k resistor on the board

  // DS18B20: read the last result, then start the next one
  float surface = probes.getTempC(PANEL_PROBE);
  float ambient = probes.getTempC(AIR_PROBE);
  probes.requestTemperatures();
  if (surface == DEVICE_DISCONNECTED_C) surface = TEMP_MISSING;
  if (ambient == DEVICE_DISCONNECTED_C) ambient = TEMP_MISSING;

  char recordId[24];
  sprintf(recordId, "M1-DEV-%03d-%07lu", bootNumber, readingNumber);
  char timeText[24] = "";
  bool haveTime = nowUtc(timeText, sizeof(timeText));

  saveToSD(recordId, timeText, voltage, current, power, lux, surface, ambient);

  // the backend needs a real UTC time, so only send when the clock is set
  if (haveTime) {
    addToBatch(recordId, timeText, voltage, current, power, lux, surface, ambient);
    if (batchCount >= BATCH_SIZE) sendBatch();
  }

  Serial.print(timeText[0] ? timeText : "no-time");
  Serial.print(" | V(calc): ");  Serial.print(voltage, 2);
  Serial.print(" V | I: ");      Serial.print(current, 1);
  Serial.print(" mA | P: ");     Serial.print(power, 1);
  Serial.print(" mW | Light: "); Serial.print(lux, 0);
  Serial.print(" lux | Surface: ");
  if (surface == TEMP_MISSING) Serial.print("--"); else Serial.print(surface, 2);
  Serial.print(" C | Ambient: ");
  if (ambient == TEMP_MISSING) Serial.print("--"); else Serial.print(ambient, 2);
  Serial.print(" C | Dust: ");
  if (dustPercent < 0) Serial.print("warming up"); else { Serial.print(dustPercent, 2); Serial.print(" %"); }
  Serial.print(" | Relay: ");
  Serial.print(relayOn ? "ON" : "OFF");
  Serial.print(" | WiFi: ");
  Serial.println(WiFi.status() == WL_CONNECTED ? "OK" : "--");
}