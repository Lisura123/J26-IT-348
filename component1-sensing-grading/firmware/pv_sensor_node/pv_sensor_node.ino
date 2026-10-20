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

Adafruit_INA219 ina219;
Adafruit_ADS1115 ads;
OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature probes(&oneWire);

// probe addresses (checked by holding each probe)
DeviceAddress PANEL_PROBE = {0x28, 0x43, 0x8B, 0x6D, 0x00, 0x00, 0x00, 0x84};
DeviceAddress AIR_PROBE   = {0x28, 0x3B, 0xFF, 0x70, 0x00, 0x00, 0x00, 0x6A};

// dust: we add up how long the pin stays LOW
volatile unsigned long lowStart = 0;
volatile unsigned long lowTotal = 0;
unsigned long dustWindowStart = 0;
float dustPercent = -1;   // -1 means "not ready yet" (first 30 s)

unsigned long lastRead = 0;

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

  // read all the other sensors every 5 s
  if (millis() - lastRead >= READ_EVERY_MS) {
    lastRead = millis();

    // INA219: panel voltage and current
    float busV = ina219.getBusVoltage_V();
    float shuntMv = ina219.getShuntVoltage_mV();
    float voltage = busV + shuntMv / 1000.0;
    float current = ina219.getCurrent_mA();

    // power = V x I (same as my pipeline: power_mw = voltage_v * current_ma)
    // the INA219 power register steps in about 2 mW, so small values show as 0
    float power = voltage * current;

    // TEMT6000 through ADS1115 A0
    float lightV = ads.computeVolts(ads.readADC_SingleEnded(0));
    float lux = lightV * 200.0;   // rough value with the 10k resistor on the board

    // DS18B20: -127 means probe disconnected (my pipeline handles this)
    probes.requestTemperatures();
    float panelTemp = probes.getTempC(PANEL_PROBE);
    float airTemp = probes.getTempC(AIR_PROBE);

    Serial.print("V: ");        Serial.print(voltage, 3);
    Serial.print(" V | I: ");   Serial.print(current, 2);
    Serial.print(" mA | P: ");  Serial.print(power, 4);
    Serial.print(" mW | Light: "); Serial.print(lux, 0);
    Serial.print(" lux | Surface: "); Serial.print(panelTemp, 2);
    Serial.print(" C | Ambient: ");   Serial.print(airTemp, 2);
    Serial.print(" C | Dust: ");
    if (dustPercent < 0) {
      Serial.println("wait 30 s");
    } else {
      Serial.print(dustPercent, 2);
      Serial.println(" %");
    }
  }
}