# ppg-gesture

Real-time heart-rate monitoring and wrist gesture recognition from **PPG** (photoplethysmography), using an **ESP32** and a **MAX30102** pulse-oximeter breakout board.

The ESP32 only reads the sensor and streams raw red / infrared samples (200 Hz) over USB. Everything else runs on your computer in Python: the live heart-rate dashboard, gesture recording, model training and real-time prediction.

![Heart-rate dashboard](docs/images/monitor.png)

> The screenshots in this README were taken with the built-in simulator (`--sim`), so you can try every tool without any hardware. The GUI text and the command-line help are in Traditional Chinese; this README explains what each screen shows. 中文說明請看 [README.zh-TW.md](README.zh-TW.md)。

## What it does

**1. Live heart-rate dashboard** (`host/monitor.py`)
Filtered PPG with detected beats, raw red / IR counts, PPG spectrum, heart-rate trend, Poincaré plot, HRV (SDNN, RMSSD, pNN50, SD1 / SD2, LF / HF), an uncalibrated SpO₂ estimate, perfusion index and a signal-quality hint.

**2. Gesture recognition from the wrist** (`host/collect.py` → `host/train.py` → `host/predict.py`)
Hand and finger movements squeeze the arteries and move the tendons under a wrist-worn PPG sensor, which changes the reflected light. This project implements the pipeline of Zhao et al., *PPG-based Finger-level Gesture Recognition Leveraging Wearables* (IEEE INFOCOM 2018) on a single MAX30102 (its red and IR channels): rest calibration, energy-based gesture detection, segmentation, hand-crafted features and a gradient-boosting classifier. The envelope + SVM / 1-NN baseline of a second paper is included for comparison.

| Record gestures | Recognize them live |
|---|---|
| ![collect.py](docs/images/collect.png) | ![predict.py](docs/images/predict.png) |

## How it works

```
MAX30102 --I2C--> ESP32 (PlatformIO firmware) --USB serial, 200 Hz "seq,red,ir"--> Python on your computer
                                                                                    |- monitor.py   heart rate / HRV / SpO2
                                                                                    |- collect.py   record labelled gestures
                                                                                    |- train.py     segment, features, models
                                                                                    '- predict.py   live recognition
```

The gesture pipeline (`host/ppgkit/gesture.py`):

1. **Rest calibration.** While you keep still (10 s in `collect.py`, 8 s in `predict.py`) the short-time energy of the high-passed signal is measured and turned into a detection threshold.
2. **Detection.** The signal is converted to relative change (raw / DC − 1), high-passed at 2 Hz, and its short-time energy (0.25 s window) is compared with the threshold. A gesture starts when the energy stays above it.
3. **Segmentation.** 1 s before and 3 s after the onset are kept; the energy burst (with hysteresis, short gaps bridged) defines the gesture window, which is then low-passed at 5 Hz and resampled to 100 Hz.
4. **Features** (per channel, plus red/IR): mean / peak-to-peak / RMS / variance, FFT statistics below 5 Hz, Haar wavelet (level 3), Wigner–Ville, 9 AR coefficients, red–IR cross-correlation, and 2D-DTW distance to a few representative segments of each gesture.
5. **Classifier.** One binary gradient-boosting classifier per gesture; the one with the highest score wins. `train.py` also trains a random forest, the SVM / 1-NN baselines and two models of our own (logistic regression and 3-NN on a fixed 3.3 s waveform window anchored at the trigger, `wlr` / `wknn`), and keeps the best by time-blocked cross-validation. Triggers that match no cue (for example the release of a held gesture) are trained as a `none` class, so live prediction can ignore them.

Differences from the paper:

| | Zhao et al. | This project |
|---|---|---|
| Sensors | two green-LED PPG sensors, 100 Hz | one MAX30102: red + IR channels, 200 Hz (features at 100 Hz) |
| Detection threshold | mean + 3σ of the energy at rest | median + 10 × MAD by default, kept between the energy of a 0.6 % and a 1.4 % level step: a very still calibration would otherwise make the threshold trigger on noise and the pulse, and one twitch during calibration inflates σ and hides most gestures (`--threshold paper` restores the paper's rule) |
| Gesture end | DTW against the user's pulse template | end of the energy burst (`--mode energy` uses the paper's fixed 1.2 s window) |
| Feature selection | elastic net | not implemented |
| Evaluation | 20 × Monte-Carlo cross-validation, 50 % for training | headline number is time-blocked cross-validation (each recording is cut into 4 stretches that take turns as the test set); the paper's random-split number is printed as an optimistic reference (`--rounds 20` for the paper's setting), plus cross-session accuracy |

## Hardware

- An ESP32 dev board (developed on an `esp32dev`-type DevKit)
- A **MAX30102** breakout (the common purple 7-pin board: VIN, SCL, SDA, INT, IRD, RD, GND)
- 4 jumper wires and a USB cable
- For wrist use: a strap (velcro, elastic band or an old watch band), and black tape or foam to shield the sensor from ambient light

### Wiring

![Wiring diagram](docs/images/wiring.svg)

| MAX30102 | ESP32 |
|---|---|
| VIN | 3V3 |
| GND | GND |
| SDA | GPIO21 |
| SCL | GPIO22 |
| INT, IRD, RD | not connected |

Power the module from **3V3**. Pin order differs between boards, so go by the silkscreen. Other ESP32 variants use their own default I²C pins, or set them with `-D I2C_SDA_PIN=..` / `-D I2C_SCL_PIN=..` in `build_flags`.

Some cheap MAX30102 boards pull SDA / SCL / INT up to 1.8 V instead of 3.3 V, which can make the sensor invisible to a 3.3 V ESP32. If the firmware reports that it cannot find the sensor, see [Troubleshooting](#troubleshooting).

## Quick start

Developed and tested on macOS. The code is plain Python + pyserial + Qt, so it should also run on Linux and Windows, but that is untested.

### 1. Get the code

```bash
git clone https://github.com/Timothychen00/ppg-gesture.git
cd ppg-gesture
```

### 2. Flash the firmware

Install [PlatformIO](https://platformio.org/) (VS Code extension or CLI), then:

```bash
cd ppg
pio run -t upload
pio device monitor
```

(The PlatformIO project lives in the `ppg/` folder; if you use VS Code, open that folder, not the repository root.) The firmware needs no libraries beyond Arduino's `Wire`. The serial monitor runs at 115200 baud and should show:

```
# MAX30102 streamer v1, columns: seq,red,ir
# i2c sda=21 scl=22
# max30102 fs=200 led_red=60 led_ir=60 range=2 ok=1
0,98213,112874
1,98220,112869
```

Commands you can type in the monitor: `led <red> <ir>` (LED current 0–255, 0.2 mA per step), `range <0-3>` (ADC full scale 2048 / 4096 / 8192 / 16384 nA) and `info`.

**Only one program can use the serial port at a time.** Close the serial monitor before running the Python tools, and close them before uploading again.

### 3. Set up Python

[uv](https://docs.astral.sh/uv/) is the easiest way; it installs the locked dependency versions (and the Python version the project was developed with) automatically on first run:

```bash
cd ../host
uv run monitor.py --sim      # try it without hardware first
```

Without uv:

```bash
cd host
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python monitor.py --sim
```

Below, commands are written as `uv run <script>`; use `python <script>` if you installed with pip.

### 4. Heart-rate dashboard

```bash
uv run monitor.py                 # ESP32 on an auto-detected serial port
uv run monitor.py --led 90        # stronger LEDs, for a weak signal (wrist)
uv run monitor.py --save rec.npz  # write the raw recording when the window closes
```

| Panel | Shows |
|---|---|
| PPG (top) | 0.5–4 Hz band-passed, inverted signal; red dots are detected beats (sub-sample peak position) |
| Raw signal | red / IR ADC counts, to check saturation and skin contact |
| PPG spectrum | Welch spectrum; peak frequency × 60 is the heart rate |
| Heart-rate trend | instantaneous rate (60 / RR) of every beat in the last 5 minutes |
| Poincaré plot | RRₙ against RRₙ₊₁; SD1 is short-term, SD2 long-term variability |
| HRV spectrum | Lomb–Scargle spectrum of the RR series with LF (0.04–0.15 Hz) and HF (0.15–0.4 Hz) bands; appears after 60 s of data |
| Side panel | heart rate (median of the last 8 beats), spectral estimate, SpO₂, perfusion index, SDNN, RMSSD, pNN50, SD1 / SD2, LF / HF, signal quality, IR DC level, number of interpolated samples |

Aim for an **IR DC level of 30–70 % of full scale** (shown in the side panel): raise `--led` if it is low, lower `--led` or raise `--range` if the signal saturates. RR intervals outside 0.33–2 s, or more than 25 % away from the local median, are ignored.

### 5. Gesture recognition

Wear the sensor first (see [Mounting the sensor](#mounting-the-sensor)), then:

```bash
# 1) Record: do each gesture when the screen says "go", then release right away
uv run collect.py --subject me --reps 20 --led 100

# 2) Train on one or more recordings
uv run train.py data/me_*.npz

# 3) Recognize live (keep still for the first 8 s: calibration)
uv run predict.py --led 100
```

**Recording.** Before the calibration, a checklist shows whether the sensor is ready: it has skin contact (IR level 20–85 % of full scale, 30–70 % is ideal), a pulse is visible, and your hand is still (motion below 1 %). Fix any red cross first; pressing Space with a failing check asks for a second press. Then comes a 10 s rest calibration: forearm flat on the table, hand relaxed, no talking, and the sensor not pressed against the table. If you move, the calibration restarts automatically, because a threshold measured while moving hides the gestures later. Then each trial is *ready → go → rest*: do the gesture **once and release it** (do not hold it: the release would be detected as a second gesture). The screen tells you immediately whether a movement was detected. On the plot, a green line marks the cue, a dashed orange line marks the end of the "go" window, and the shaded area is the movement the live detector found; all of these are saved with the recording. Keys: `Space` start / pause, `Backspace` delete the last trial (it is queued again at the end), `Esc` save and quit. The recording is saved to `host/data/<subject>_<timestamp>.npz` after every trial.

**Strap tightness.** A bar above the plot shows how snugly the sensor sits: the IR level in % of full scale, which rises with pressure and falls with a gap. The green band is the target: `collect.py` uses this session's own calibration level, or that of an earlier recording with `--match data/earlier.npz`; `predict.py` uses the level of the recordings the model was trained on. Adjust the strap until the marker is inside the band before you calibrate, so every wearing is held at the same level. The numbers only compare between runs with the same `--led` and `--range`. The IR level is only a proxy for pressure: it is measured at rest, not while you gesture.

Gesture sets: `--preset basic` (default: `fist`, `open`, `pinch`, `wrist_up`, `wrist_down`), `--preset asl` (American Sign Language numbers 1–9, as in the paper), or your own with `--gestures fist,open,pinch`. Try a few large movements first (for example `--gestures fist,wrist_up,wrist_down --reps 10`, which takes a few minutes) before recording a large set.

**Training.** `train.py` prints, for each recording, how many cues were matched to a detected movement, then a time-blocked cross-validation of every model (neighbouring trials never end up on both sides; the random split of paper A is printed once for reference and is typically 10–20 points higher), the confusion matrix (also saved as a PNG), the accuracy when a whole recording is held out (only when you pass more than one recording; this is the honest number, see below) and the most important features. It saves the best model to `host/models/gesture_model.joblib`.

**Prediction.** `predict.py` shows the recognized gesture with its confidence, the probability of every gesture and a history. A result appears about 3 s after the gesture starts, because the whole movement including the release is needed. Press `C` to recalibrate after moving your arm or re-wearing the sensor; `--min-conf 0.6` shows "?" for results below 60 % confidence. Movements the model classifies as `none` (for example the release of a held gesture) are ignored and only noted in the history. The status line in the side panel says what the detector is doing (calibrating, waiting with the current threshold, movement detected, collecting), and every detected movement is also printed in the terminal with its peak energy and the top three class probabilities, which is the first thing to look at when nothing seems to happen. The calibration restarts by itself while the hand moves. With `--sim` or `--replay` the window also compares each result with the true gesture.

### 6. Try it without hardware

```bash
uv run collect.py --sim --reps 10
uv run train.py data/me_*.npz
uv run predict.py --sim
```

The simulator invents its own gesture shapes, so it only shows that the pipeline works end to end. Its accuracy says nothing about real wrist signals.

### 7. Use the model from your own code

`predict.py` is only a window around this loop:

```python
import time
import joblib
from ppgkit.gesture import LivePredictor, display_name
from ppgkit.sources import SerialSource

bundle = joblib.load("models/gesture_model.joblib")
predictor = LivePredictor(bundle)   # the first 8 s are calibration: keep still
src = SerialSource().start()        # finds the ESP32's serial port
try:
    while True:
        red, ir = src.read()
        for r in predictor.feed(red, ir):
            print(display_name(r["label"]), f"{r['conf']:.0%}")
        time.sleep(0.03)
finally:
    src.stop()
```

Each result `r` also has `r["proba"]` (in the order of `predictor.classes`) and `r["start"]` / `r["end"]` (sample indices of the gesture). `joblib` files are Python pickles: only load models you trained yourself or trust.

## Mounting the sensor

- **Heart rate:** the fingertip gives the strongest signal. Other places, roughly from strong to weak: the base of a finger, the earlobe (clip), the forehead (headband), the wrist.
- **Gestures:** wrist only. Put the sensor on the **inside of the wrist**, about one finger-width from the wrist crease, window against the skin, and strap it on.
- **Tightness:** a loose sensor lets light in and slides; a very tight one squeezes the pulse away. Adjust while watching the perfusion index in `monitor.py`: the tightness that maximizes it is a good one.
- **Ambient light:** cover the sensor with black tape or foam. Do not put clear tape over the window; the reflection disturbs the signal.
- **Cables:** tape them to the forearm so that arm movements do not pull on the sensor.
- **Repeatability matters most.** Gesture waveforms depend strongly on exactly where and how tightly the sensor sits. Fix the sensor to a watch band or wrist strap (with a foam ring around the window), mark its position on your skin, and note which hole of the strap you use.

## Status and limitations

This is an experimental project.

- **Hardware and streaming** work: 200 Hz red / IR samples over USB, with lost samples detected and interpolated.
- **Heart rate / HRV** were developed and tested mostly on simulated PPG. Check them against your own reference before relying on them (below). The **SpO₂ value is not calibrated** for your board; treat it as a trend.
- **Gesture recognition, preliminary results** (one person, one device, wrist sensor, 20 trials per gesture per recording):
  - Within one wearing, three large movements (fist, wrist up, wrist down; 10 trials each) were recognized about 90–95 % of the time.
  - With five gestures (fist, open hand, pinch, wrist up, wrist down) about 55–65 %. Wrist down is recognized reliably (about 85–95 %); fist, open hand and pinch are often confused with each other.
  - **Across wearings it did not generalize.** A model trained on one recording and tested on a recording after taking the sensor off and putting it back on scored near chance (15–18 % for five gestures). After re-wearing, the response to the same gesture can even change sign.
  - So, for now: **record, train and use within the same wearing**, or fix the sensor rigidly and record several wearings into one model. Prefer gestures whose movements are very different from each other.
- The paper uses two separate sensors on the wrist. A single MAX30102 gives only two wavelengths at one spot, which may carry less information about finger movements.

## Checking that the numbers are right

- **Count your pulse.** Note the beat counter in the side panel, count your pulse for 60 s (neck or wrist, stopwatch), and compare with how much the counter advanced. A difference of 0–1 is good. If it is bigger, look at the plot for beats without a red dot or dots doubled on one beat.
- **Compare with a pulse oximeter** (finger clip) on another finger while sitting still: expect agreement within 2–3 bpm.
- **Internal consistency:** the heart rate and the spectral estimate agree within 3 bpm, every peak has exactly one red dot, the Poincaré points cluster near the diagonal.
- **Physiology:** standing up should raise the heart rate by roughly 10–20 bpm; slow breathing (5 s in, 5 s out) makes the trend oscillate and puts a peak near 0.1 Hz in the HRV spectrum.
- **Gestures:** trust the cross-recording accuracy from `train.py`, or record a fresh recording after re-wearing the sensor, do not train on it, and replay it with `uv run predict.py --replay data/that_file.npz`. The line at the bottom of the side panel then shows how many gestures were performed, how many were detected, how many were right and how many false triggers there were; right / performed is your real accuracy.

## Command reference

Options shared by `monitor.py`, `collect.py` and `predict.py`:

| Option | Meaning |
|---|---|
| `--port PORT` | serial port (default: auto-detect, e.g. `/dev/cu.usbserial-0001`, `/dev/ttyUSB0`, `COM3`) |
| `--baud N` | baud rate, default 115200 (must match the firmware) |
| `--led 0-255` | LED current for red and IR (0.2 mA per step), re-sent on every connection because the ESP32 resets when the port opens |
| `--range 0-3` | ADC range: 0 = 2048, 1 = 4096, 2 = 8192, 3 = 16384 nA |
| `--swap` | swap the red and IR columns |
| `--sim` | built-in simulator instead of hardware |
| `--replay FILE.npz` | replay a recording in real time |
| `--speed X` | speed factor for `--sim` / `--replay` |

Tool-specific options:

| Tool | Option | Meaning |
|---|---|---|
| `monitor.py` | `--save FILE.npz` | save the raw recording on exit |
| `collect.py` | `--subject NAME` | name used in the file name |
| | `--preset basic\|asl`, `--gestures a,b,c` | gesture set |
| | `--reps N` | trials per gesture (paper: 40) |
| | `--out DIR` | output folder (default `data`) |
| `train.py` | `--model auto\|gbt\|rf\|svm\|knn\|wlr\|wknn` | which classifier to save (`auto`: best cross-validation) |
| | `--compare wlr,rf,...` | which models to compare (default `wlr,wknn,rf,svm,knn,gbt`) |
| | `--gestures a,b,c` | train only these gestures, e.g. `wrist_down,wrist_up,fist` |
| | `--blocks N`, `--no-none` | stretches per recording for blocked cross-validation (4); do not train the `none` class |
| | `--rounds N`, `--train-frac F`, `--trees N` | cross-validation rounds (10), training fraction (0.5), boosting rounds (300; the paper used 2000) |
| | `--threshold robust\|paper`, `--k K` | detection threshold rule and multiplier. Many missed gestures: lower `--k` (e.g. 5). Many false triggers: raise it (e.g. 20) |
| | `--mode threshold\|energy`, `--hp HZ` | segmentation mode, detection high-pass cut-off (default 2 Hz) |
| | `--out FILE` | model file (default `models/gesture_model.joblib`) |
| `predict.py` | `--model FILE` | model to load |
| | `--min-conf F` | show "?" below this confidence (default 0.5) |

## Project layout

```
ppg/                      PlatformIO project (ESP32 firmware)
  platformio.ini
  src/main.cpp            reads the MAX30102 FIFO, streams "seq,red,ir" lines
host/                     Python tools (uv project)
  monitor.py              heart-rate dashboard
  collect.py              gesture recording
  train.py                training and evaluation
  predict.py              live gesture recognition
  ppgkit/
    sources.py            serial / replay sources, recording file I/O
    dsp.py                filtering, beat detection, HRV, SpO2
    fitbar.py             live strap-tightness gauge shown by collect.py / predict.py
    gesture.py            detection, segmentation, features, models
    sim.py                synthetic PPG for testing without hardware
  data/  models/          your recordings and models (created on first use, not in git)
docs/images/              figures used in this README
```

### Serial protocol and recording format

The firmware prints one line per sample, `seq,red,ir` (raw 18-bit counts at 200 Hz). `seq` counts samples; a jump means samples were lost, and the Python side fills the gap by interpolation. Lines starting with `#` are status messages.

Recordings are NumPy `.npz` files with `red`, `ir` (float32), `fs` (Hz), `event_label` (name of every gesture), `calib` (start and end sample of the rest period), `meta` (JSON string) and the following markers, all sample indices with −1 meaning "not known":

| Array | Meaning |
|---|---|
| `event_index` | the cue: "go" appeared |
| `event_go_end` | the "go" window closed (cue + 2.5 s); the gesture should be finished and released by now |
| `event_end` | end of the trial, rest included; the next trial starts here |
| `event_move_start`, `event_move_end` | start and end of the movement the live detector found after the cue (−1 if it found none) |

Load a recording with `ppgkit.sources.load_session(path)`; its `events` are the cues and `spans` holds the other markers, one dict per event (`None` for recordings made before the end markers existed, which only have the cues). `train.py` still finds the segments itself from the raw signal, so you can change `--k` and friends without recording again; the stored markers are for your own analysis and plots.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `# ERROR MAX30102 not found` and `# i2c scan:` finds nothing | Check the wiring, SDA / SCL not swapped, module on 3V3. If it still fails, see the pull-up note below |
| `part id 0x11` | The module is a MAX30100; this firmware supports only the MAX30102 |
| No serial port found | Pass `--port` (`pio device list` lists ports). Boards with a CH340 or CP210x chip may need a driver |
| `打不開 ...` (cannot open the port) | Another program has the port: close the PlatformIO serial monitor |
| Upload fails with "port busy" | A Python tool still has the port open: close it |
| Upload hangs at `Connecting....` | Hold the board's BOOT button until writing starts |
| Flat line, or signal at the maximum | Adjust `--led` / `--range`; check skin contact and shading |
| "Interpolated samples" keeps growing | Try another USB cable or plug straight into the computer |
| Red and IR seem swapped | Type `led 60 0` in the serial monitor (turns IR off); the column that drops to about 0 is IR. If it is the other one, use `--swap` |
| Many missed or extra gestures in `train.py` | Tune `--k` (see the command reference), keep still during calibration, make gestures larger |

**Pull-ups on 1.8 V.** Some MAX30102 breakouts connect the 4.7 kΩ pull-up resistors of SDA / SCL / INT to the 1.8 V regulator instead of 3.3 V. Look for a 3-pad solder jumper (1.8 V / 3.3 V) on the back and bridge it to 3.3 V; otherwise cut the trace to 1.8 V and wire the resistors to 3.3 V. Adding external pull-ups to 3.3 V on top of the 1.8 V ones is not a fix. See [this write-up](https://reedpaper.wordpress.com/2018/08/22/pulse-oximeter-max30100-max30102-how-to-fix-wrong-board/).

## References

- T. Zhao, J. Liu, Y. Wang, H. Liu, Y. Chen, "PPG-based Finger-level Gesture Recognition Leveraging Wearables," *IEEE INFOCOM 2018*. https://ieeexplore.ieee.org/document/8486006
- "Principal component analysis of photoplethysmography signals for improved gesture recognition" (wrist PPG with a MAX30105; source of the envelope + SVM / 1-NN baseline). https://pmc.ncbi.nlm.nih.gov/articles/PMC9669422/
- MAX30102 datasheet, Analog Devices / Maxim Integrated.

## Disclaimer

This is a hobby / research project, not a medical device. Do not use it to diagnose, monitor or treat any condition.
