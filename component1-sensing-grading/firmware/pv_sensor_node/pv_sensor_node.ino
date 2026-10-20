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

#define READ_EVERY_MS 2000    // fast, so the demo reacts quickly
#define DUST_WINDOW_MS 30000  // DSM501B needs about 30 s to measure
#define DUST_WARMUP_MS 60000  // and about 1 min to warm up
#define LOAD_OHMS 120.0

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

void IRAM_ATTR dustChanged() {
  if (digitalRead(DUST_PIN) == LOW) {
    lowStart = micros();
  } else {
    lowTotal += micros() - lowStart;
  }
}

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
  dustWindowStart = millis();

  Serial.println();
  Serial.println("=============================================");
  Serial.println("  PV sensor node - Member 1 (live readings)");
  Serial.println("=============================================");
  Serial.print("  DS18B20 probes found: ");
  Serial.println(probes.getDeviceCount());
  Serial.println("  Voltage = current x 120 ohm (GND wire not added yet)");
  Serial.println("  Type 1 = relay ON (panel cut), 0 = relay OFF");
  Serial.println("  Dust needs about 90 s to start");
  Serial.println();
}

void loop() {
  // relay control from Serial Monitor
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '1') {
      digitalWrite(RELAY_PIN, LOW);
      relayOn = true;
      Serial.println(">> Relay ON - panel cut");
    } else if (c == '0') {
      digitalWrite(RELAY_PIN, HIGH);
      relayOn = false;
      Serial.println(">> Relay OFF - panel connected");
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

  // INA219: current is measured, voltage is calculated
  float current = ina219.getCurrent_mA();
  float voltage = current / 1000.0 * LOAD_OHMS + ina219.getShuntVoltage_mV() / 1000.0;
  float power = voltage * current;

  // TEMT6000 through ADS1115 A0
  float lightV = ads.computeVolts(ads.readADC_SingleEnded(0));
  float lux = lightV * 200.0;   // rough value with the 10k resistor on the board

  // DS18B20: read the last result, then start the next one
  float surface = probes.getTempC(PANEL_PROBE);
  float ambient = probes.getTempC(AIR_PROBE);
  probes.requestTemperatures();

  Serial.print("V(calc): ");   Serial.print(voltage, 2);
  Serial.print(" V | I: ");    Serial.print(current, 1);
  Serial.print(" mA | P: ");   Serial.print(power, 1);
  Serial.print(" mW | Light: "); Serial.print(lux, 0);
  Serial.print(" lux | Surface: ");
  if (surface == DEVICE_DISCONNECTED_C) Serial.print("--"); else Serial.print(surface, 2);
  Serial.print(" C | Ambient: ");
  if (ambient == DEVICE_DISCONNECTED_C) Serial.print("--"); else Serial.print(ambient, 2);
  Serial.print(" C | Dust: ");
  if (dustPercent < 0) Serial.print("warming up"); else { Serial.print(dustPercent, 2); Serial.print(" %"); }
  Serial.print(" | Relay: ");
  Serial.println(relayOn ? "ON" : "OFF");
}