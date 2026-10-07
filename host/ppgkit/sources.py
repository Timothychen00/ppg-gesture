"""Where samples come from: the ESP32 over USB serial, the built-in simulator
(see sim.py) or a recorded session played back.

All sources share one small interface:

    src = open_source(args).start()
    red, ir = src.read()   # raw counts that arrived since the previous call
    src.total              # samples delivered so far = index of the next sample
    src.stop()

Sample indices are continuous: when the firmware's `seq` counter shows lost
samples, the gap is filled by linear interpolation and counted in `dropped`.
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

FS = 200.0  # firmware output rate


class GrowBuffer:
    """Append-only (channels, n) float array with amortised growth."""

    def __init__(self, channels=2, capacity=1 << 15):
        self._a = np.zeros((channels, capacity))
        self.n = 0

    def extend(self, *cols):
        k = len(cols[0])
        if self.n + k > self._a.shape[1]:
            grown = np.zeros((self._a.shape[0], max(2 * self._a.shape[1], self.n + k)))
            grown[:, :self.n] = self._a[:, :self.n]
            self._a = grown
        for c, col in enumerate(cols):
            self._a[c, self.n:self.n + k] = col
        self.n += k

    @property
    def data(self):
        return self._a[:, :self.n]

    def __len__(self):
        return self.n


class Source:
    fs = FS

    def __init__(self):
        self._lock = threading.RLock()
        self._red: list[float] = []
        self._ir: list[float] = []
        self._thread = None
        self._running = False
        self.total = 0
        self.dropped = 0
        self.last_rx = time.time()  # when samples last arrived, for stall warnings
        self.info: dict[str, str] = {}
        self.status = ""

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def read(self):
        with self._lock:
            red = np.asarray(self._red, dtype=float)
            ir = np.asarray(self._ir, dtype=float)
            self._red.clear()
            self._ir.clear()
        return red, ir

    def send(self, command: str):
        """Send a firmware command such as "led 60 60" (ignored without hardware)."""

    def _push(self, red, ir):
        with self._lock:
            self._red.extend(red)
            self._ir.extend(ir)
            self.total += len(red)
            self.last_rx = time.time()

    def stalled(self, seconds=1.0):
        """True when a live source has delivered nothing for `seconds` (unplugged, board reset)."""
        return self._running and time.time() - self.last_rx > seconds

    def _run(self):
        raise NotImplementedError


# ---------------------------------------------------------------- serial ----

# USB-serial chips found on ESP32 boards: CP210x, CH34x, FTDI, Espressif native USB
_KNOWN_VIDS = {0x10C4, 0x1A86, 0x0403, 0x303A}
_PORT_HINTS = ("usbserial", "wchusbserial", "SLAB_USBtoUART", "usbmodem", "ttyUSB", "ttyACM")


def find_port() -> str:
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    for p in ports:
        if p.vid in _KNOWN_VIDS:
            return p.device
    for p in ports:
        if any(h in p.device for h in _PORT_HINTS):
            return p.device
    names = ", ".join(p.device for p in ports) or "（沒有）"
    raise SystemExit(f"找不到 ESP32 的序列埠，請用 --port 指定。目前看到的埠：{names}")


class SerialSource(Source):
    """Reads `seq,red,ir` lines from the ESP32 firmware in ../ppg (PlatformIO project)."""

    def __init__(self, port=None, baud=115200, swap=False):
        super().__init__()
        import serial

        self.port = port or find_port()
        try:
            self.ser = serial.Serial(self.port, baud, timeout=0.05)
        except (serial.SerialException, OSError) as e:
            raise SystemExit(f"打不開 {self.port}：{e}\n（PlatformIO 的 Serial Monitor 有開著嗎？要先關掉）")
        self.swap = swap
        self.status = f"已連線 {self.port}，等待資料…"
        self._last_seq = None
        self._last = None
        self._queued: list[str] = []
        self._commands: list[str] = []

    def queue(self, command: str):
        """Send `command` once data is flowing (opening the port resets most ESP32 boards).
        It is sent again whenever the board restarts, since the firmware forgets its settings."""
        self._queued.append(command)
        self._commands.append(command)

    def send(self, command: str):
        self.ser.write((command.strip() + "\n").encode())

    def stop(self):
        super().stop()
        self.ser.close()

    def _run(self):
        pending = b""
        while self._running:
            try:
                chunk = self.ser.read(max(1, self.ser.in_waiting))
            except Exception as e:  # unplugged
                self.status = f"序列埠錯誤：{e}"
                return
            if not chunk:
                continue
            *lines, pending = (pending + chunk).split(b"\n")
            reds, irs = [], []
            for raw in lines:
                line = raw.decode("ascii", "ignore").strip()
                if line.startswith("#"):
                    self._on_status(line[1:].strip())
                elif line:
                    self._on_sample(line, reds, irs)
            if reds:
                self._push(reds, irs)
                while self._queued:
                    self.send(self._queued.pop(0))

    def _on_status(self, text):
        self.status = text
        if text.startswith("MAX30102 streamer"):
            self._last_seq = None  # board (re)started, seq begins again at 0
            self._queued = list(self._commands)
        for token in text.split():
            key, eq, value = token.partition("=")
            if eq:
                self.info[key] = value
        if "fs" in self.info:
            self.fs = float(self.info["fs"])

    def _on_sample(self, line, reds, irs):
        parts = line.split(",")
        if len(parts) != 3:
            return
        try:
            seq, red, ir = (int(p) for p in parts)
        except ValueError:
            return  # boot noise
        if not (0 <= red < 1 << 18 and 0 <= ir < 1 << 18):
            return  # corrupted line (the sensor is 18-bit)
        if self.swap:
            red, ir = ir, red
        if self._last_seq is not None:
            gap = seq - self._last_seq - 1
            if gap < 0:  # counter went backwards without a banner: treat as a restart
                self._last_seq = None
            elif 0 < gap <= 10 * self.fs:
                w = np.arange(1, gap + 1) / (gap + 1)
                reds.extend(self._last[0] + (red - self._last[0]) * w)
                irs.extend(self._last[1] + (ir - self._last[1]) * w)
                self.dropped += gap
        self._last_seq, self._last = seq, (red, ir)
        reds.append(red)
        irs.append(ir)


# ---------------------------------------------------------------- replay ----

class ReplaySource(Source):
    """Plays a recorded session (.npz from collect.py or monitor.py --save) in real time."""

    def __init__(self, path, speed=1.0):
        super().__init__()
        s = load_session(path)
        self.fs = s["fs"]
        self._data = (s["red"], s["ir"])
        self.truth = s["events"]
        self.calib = s["calib"]
        self.speed = speed
        self.status = f"重播 {os.path.basename(path)}"

    def _run(self):
        red, ir = self._data
        start, pos = time.monotonic(), 0
        while self._running and pos < len(red):
            due = min(len(red), int((time.monotonic() - start) * self.fs * self.speed))
            if due > pos:
                self._push(red[pos:due], ir[pos:due])
                pos = due
            time.sleep(0.01)
        self.status = "重播結束"


# ------------------------------------------------------------ session io ----

# Where each recorded trial ends (sample indices, -1 = not known). `event_index` is the cue.
#   go_end      the "go" window closed: the gesture should be finished and released
#   end         the trial is over, rest included (the next trial's "ready" starts here)
#   move_start  start and end of the movement the live detector found after the cue,
#   move_end    -1 when it found none
SPAN_KEYS = ("go_end", "end", "move_start", "move_end")


def save_session(path, red, ir, fs, events=(), calib=None, meta=None, spans=None):
    """Save raw data plus (sample_index, label) events; written atomically.

    `spans` is optional: one dict per event with any of SPAN_KEYS (see above).
    """
    events = list(events)
    extra = {}
    if spans is not None:
        spans = list(spans)
        if len(spans) != len(events):
            raise ValueError("spans must have one entry per event")
        for key in SPAN_KEYS:
            extra["event_" + key] = np.array([int(s.get(key, -1)) for s in spans], dtype=np.int64)
    tmp = path[:-4] + ".tmp.npz" if path.endswith(".npz") else path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        red=np.asarray(red, np.float32),
        ir=np.asarray(ir, np.float32),
        fs=np.float64(fs),
        event_index=np.array([e[0] for e in events], dtype=np.int64),
        event_label=np.array([e[1] for e in events], dtype=str),
        calib=np.array(calib if calib else (-1, -1), dtype=np.int64),
        meta=np.array(json.dumps(meta or {}, ensure_ascii=False)),
        **extra,
    )
    os.replace(tmp, path)


def load_session(path):
    """Returns a dict with red, ir, fs, events [(cue index, label)], calib, meta and
    spans: one dict per event with the SPAN_KEYS that were recorded, or None for
    recordings made before end markers existed."""
    with np.load(path) as z:
        calib = tuple(int(v) for v in z["calib"])
        n = len(z["event_index"])
        keys = [k for k in SPAN_KEYS if "event_" + k in z.files]
        return {
            "red": z["red"].astype(float),
            "ir": z["ir"].astype(float),
            "fs": float(z["fs"]),
            "events": [(int(i), str(l)) for i, l in zip(z["event_index"], z["event_label"])],
            "spans": [{k: int(z["event_" + k][i]) for k in keys} for i in range(n)] if keys else None,
            "calib": calib if calib[0] >= 0 else None,
            "meta": json.loads(str(z["meta"])),
        }


# ------------------------------------------------------------------- cli ----

def add_source_args(parser):
    g = parser.add_argument_group("資料來源")
    g.add_argument("--port", help="序列埠，例如 /dev/cu.usbserial-0001（預設自動尋找）")
    g.add_argument("--baud", type=int, default=115200)
    g.add_argument("--led", type=int, metavar="0-255", help="LED 電流，紅光與紅外光一起設定（每格 0.2 mA）")
    g.add_argument("--range", dest="adc_range", type=int, choices=range(4),
                   help="ADC 量程：0=2048 1=4096 2=8192 3=16384 nA")
    g.add_argument("--swap", action="store_true", help="紅光／紅外光欄位對調")
    g.add_argument("--sim", action="store_true", help="不接硬體，用內建模擬訊號")
    g.add_argument("--replay", metavar="FILE.npz", help="重播錄好的資料")
    g.add_argument("--speed", type=float, default=1.0, help="模擬／重播速度倍率")


def open_source(args, sim_gestures=None):
    if args.replay:
        return ReplaySource(args.replay, speed=args.speed)
    if args.sim:
        from .sim import SimSource

        return SimSource(gestures=sim_gestures, speed=args.speed)
    src = SerialSource(args.port, args.baud, swap=args.swap)
    if args.led is not None:
        src.queue(f"led {args.led} {args.led}")
    if args.adc_range is not None:
        src.queue(f"range {args.adc_range}")
    return src
