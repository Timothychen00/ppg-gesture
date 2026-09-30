/*
  MAX30102 -> ESP32 raw PPG streamer
  ----------------------------------
  Streams raw Red / IR samples over USB serial as CSV lines:

      seq,red,ir

  `seq` counts samples; if the sensor FIFO ever overflows, `seq` jumps by the
  number of lost samples so the host can see (and fill) the gap. Lines that
  start with '#' are status messages; the Python tools in ../../host read them.
  All analysis and the gesture model run on the computer, in Python.

  Sensor setup: SpO2 mode (Red + IR), 400 Hz sampling with 2x on-chip
  averaging -> 200 Hz output, 411 us LED pulses (18-bit ADC).

  Wiring (ESP32 DevKit / WROOM-32):
      MAX30102  VIN -> 3V3         GND -> GND
                SDA -> GPIO21      SCL -> GPIO22
                INT, IRD, RD -> not connected
  Other boards use their default SDA / SCL pins; override them with
  build_flags in platformio.ini.

  Serial commands (newline terminated, e.g. from `pio device monitor`):
      led <red> <ir>   LED current 0-255, 0.2 mA per step   e.g. "led 60 60"
      range <0-3>      ADC full scale: 0=2048 nA 1=4096 2=8192 3=16384
      info             print the current settings
*/
#include <Arduino.h>
#include <Wire.h>

// ---------- user settings (can be overridden with build_flags) ----------
#ifndef I2C_SDA_PIN
#define I2C_SDA_PIN SDA  // the board's default I2C pins: GPIO21 / GPIO22 on esp32dev
#endif
#ifndef I2C_SCL_PIN
#define I2C_SCL_PIN SCL
#endif
#ifndef SERIAL_BAUD
#define SERIAL_BAUD 115200  // keep equal to monitor_speed in platformio.ini (200 Hz needs ~4.4 kB/s)
#endif
static uint8_t ledRed = 60;   // 60 * 0.2 mA = 12 mA
static uint8_t ledIr = 60;
static uint8_t adcRange = 2;  // 8192 nA full scale

// ---------- MAX30102 registers (datasheet table "Register Maps") ----------
static const uint8_t I2C_ADDR = 0x57;
static const uint8_t REG_INT_STATUS_1 = 0x00;
static const uint8_t REG_FIFO_WR_PTR = 0x04;  // followed by OVF_COUNTER (0x05)
static const uint8_t REG_OVF_COUNTER = 0x05;  // and FIFO_RD_PTR (0x06)
static const uint8_t REG_FIFO_RD_PTR = 0x06;
static const uint8_t REG_FIFO_DATA = 0x07;
static const uint8_t REG_FIFO_CONFIG = 0x08;
static const uint8_t REG_MODE_CONFIG = 0x09;
static const uint8_t REG_SPO2_CONFIG = 0x0A;
static const uint8_t REG_LED1_PA = 0x0C;  // red
static const uint8_t REG_LED2_PA = 0x0D;  // IR
static const uint8_t REG_PART_ID = 0xFF;
static const uint8_t PART_ID_MAX30102 = 0x15;
static const uint8_t PART_ID_MAX30100 = 0x11;

static const uint8_t SMP_AVE = 1;  // 2 samples averaged on chip
static const uint8_t SPO2_SR = 3;  // 400 Hz
static const uint8_t LED_PW = 3;   // 411 us, 18-bit
static const int OUTPUT_HZ = 200;  // 400 Hz / 2
static const uint8_t MAX_CHUNK = 16;  // samples per I2C burst (16 * 6 = 96 bytes < Wire buffer)

static bool sensorOk = false;
static uint32_t seq = 0;
static uint32_t lastRetryMs = 0;
static String command;

static bool writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(I2C_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

static bool readRegs(uint8_t reg, uint8_t *buf, uint8_t len) {
  Wire.beginTransmission(I2C_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(I2C_ADDR, len) != len) return false;
  for (uint8_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}

static uint8_t spo2Config() {
  return (adcRange << 5) | (SPO2_SR << 2) | LED_PW;
}

static void printInfo() {
  Serial.printf("# max30102 fs=%d led_red=%u led_ir=%u range=%u ok=%d\n",
                OUTPUT_HZ, ledRed, ledIr, adcRange, sensorOk ? 1 : 0);
}

static void scanBus() {
  Serial.print("# i2c scan:");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf(" 0x%02X", addr);
      found++;
    }
  }
  Serial.println(found ? "" : " (nothing found)");
}

static bool setupSensor() {
  uint8_t id = 0;
  if (!readRegs(REG_PART_ID, &id, 1)) return false;
  if (id != PART_ID_MAX30102) {
    Serial.printf("# ERROR part id 0x%02X, expected 0x15 (MAX30102)%s\n", id,
                  id == PART_ID_MAX30100 ? " - this is a MAX30100" : "");
    return false;
  }

  writeReg(REG_MODE_CONFIG, 0x40);  // soft reset, the bit clears itself when done
  uint8_t mode = 0x40;
  uint32_t t0 = millis();
  while ((mode & 0x40) && millis() - t0 < 100) {
    delay(1);
    readRegs(REG_MODE_CONFIG, &mode, 1);
  }
  uint8_t status;
  readRegs(REG_INT_STATUS_1, &status, 1);  // clears the power-ready flag

  return writeReg(REG_FIFO_CONFIG, (SMP_AVE << 5) | 0x0F)  // rollover off: overflow shows up in OVF_COUNTER
         && writeReg(REG_SPO2_CONFIG, spo2Config())
         && writeReg(REG_LED1_PA, ledRed)
         && writeReg(REG_LED2_PA, ledIr)
         && writeReg(REG_FIFO_WR_PTR, 0)
         && writeReg(REG_OVF_COUNTER, 0)
         && writeReg(REG_FIFO_RD_PTR, 0)
         && writeReg(REG_MODE_CONFIG, 0x03);  // SpO2 mode: Red + IR
}

static void connectSensor() {
  sensorOk = setupSensor();
  if (sensorOk) {
    printInfo();
  } else {
    Serial.println("# ERROR MAX30102 not found at 0x57 - check wiring and the pull-up jumper (see README)");
    scanBus();
  }
  lastRetryMs = millis();
}

static void readFifo() {
  uint8_t ptr[3];  // FIFO_WR_PTR, OVF_COUNTER, FIFO_RD_PTR
  if (!readRegs(REG_FIFO_WR_PTR, ptr, 3)) {
    sensorOk = false;
    Serial.println("# ERROR i2c read failed, retrying");
    return;
  }
  uint8_t lost = ptr[1] & 0x1F;
  uint8_t n = (ptr[0] - ptr[2]) & 0x1F;
  if (n == 0 && lost) n = 32;  // write pointer wrapped onto the read pointer: FIFO full

  uint8_t buf[MAX_CHUNK * 6];
  while (n > 0) {
    uint8_t chunk = n > MAX_CHUNK ? MAX_CHUNK : n;
    if (!readRegs(REG_FIFO_DATA, buf, chunk * 6)) {
      sensorOk = false;
      Serial.println("# ERROR i2c read failed, retrying");
      return;
    }
    for (uint8_t i = 0; i < chunk; i++) {
      const uint8_t *p = buf + i * 6;
      uint32_t red = (((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2]) & 0x3FFFF;
      uint32_t ir = (((uint32_t)p[3] << 16) | ((uint32_t)p[4] << 8) | p[5]) & 0x3FFFF;
      Serial.printf("%lu,%lu,%lu\n", (unsigned long)seq++, (unsigned long)red, (unsigned long)ir);
    }
    n -= chunk;
  }
  // With rollover off, the lost samples come after the ones still in the FIFO.
  seq += lost;
}

static void handleCommand(String line) {
  line.trim();
  int a, b;
  if (sscanf(line.c_str(), "led %d %d", &a, &b) == 2) {
    ledRed = constrain(a, 0, 255);
    ledIr = constrain(b, 0, 255);
    writeReg(REG_LED1_PA, ledRed);
    writeReg(REG_LED2_PA, ledIr);
  } else if (sscanf(line.c_str(), "range %d", &a) == 1) {
    adcRange = constrain(a, 0, 3);
    writeReg(REG_SPO2_CONFIG, spo2Config());
  } else if (line != "info") {
    Serial.printf("# unknown command: %s\n", line.c_str());
  }
  printInfo();
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, 400000);
  Serial.println("# MAX30102 streamer v1, columns: seq,red,ir");
  Serial.printf("# i2c sda=%d scl=%d\n", (int)I2C_SDA_PIN, (int)I2C_SCL_PIN);
  connectSensor();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (command.length()) handleCommand(command);
      command = "";
    } else if (command.length() < 64) {
      command += c;
    }
  }

  if (!sensorOk) {
    if (millis() - lastRetryMs > 2000) connectSensor();
    return;
  }
  readFifo();
  delay(2);  // 200 Hz fills the 32-sample FIFO in 160 ms; polling every ~2 ms is plenty
}
