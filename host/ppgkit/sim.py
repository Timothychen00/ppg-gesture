"""Synthetic wrist PPG so every tool can be tried (and tested) without hardware.

The model: raw = DC * (1 + relative change) + ADC noise, where the relative
change holds heartbeats (with respiratory sinus arrhythmia), breathing, slow
drift and gesture artefacts. Gesture shapes are invented but follow what the
papers describe: tendon/muscle movement compresses the tissue under the
sensor, giving short bumps that are several times larger than a pulse.
"""
from __future__ import annotations

import time
import zlib

import numpy as np
from scipy.signal import lfilter

from .sources import Source

# Each gesture = a few Gaussian bumps: (centre s, width s, red %, IR %).
TEMPLATES = {
    "fist": [(0.25, 0.10, -3.0, -2.2), (0.60, 0.15, 1.2, 0.8)],
    "open": [(0.20, 0.08, 1.8, 2.4), (0.45, 0.12, -0.8, -1.0)],
    "pinch": [(0.15, 0.06, -1.2, -0.6), (0.32, 0.06, 1.0, 0.5)],
    "index": [(0.20, 0.08, -0.9, -1.4), (0.45, 0.10, 0.5, 0.9)],
    "wrist_up": [(0.30, 0.16, 2.5, 1.2), (0.70, 0.12, -0.6, -0.3)],
    "wrist_down": [(0.30, 0.16, -1.5, -2.8), (0.70, 0.12, 0.4, 0.7)],
}


def template(label):
    if label in TEMPLATES:
        return TEMPLATES[label]
    rng = np.random.default_rng(zlib.crc32(label.encode()))  # stable per label
    centres = np.sort(rng.uniform(0.1, 0.7, rng.integers(2, 4)))
    return [(c, rng.uniform(0.05, 0.14), *rng.uniform(-3, 3, 2)) for c in centres]


def pulse_shape(d):
    """Blood-volume pulse, d = seconds since the beat: fast rise, slow fall, dicrotic wave."""
    peak = np.where(d < 0.12, np.exp(-((d - 0.12) / 0.045) ** 2), np.exp(-((d - 0.12) / 0.13) ** 2))
    return peak + 0.25 * np.exp(-((d - 0.38) / 0.06) ** 2)


class SimSource(Source):
    def __init__(self, fs=200.0, hr=72.0, seed=None, gestures=None, every=(4.0, 7.0), speed=1.0):
        super().__init__()
        self.fs = fs
        self.hr = hr
        self.speed = speed
        self.rng = np.random.default_rng(seed)
        self.dc = np.array([90000.0, 110000.0])  # red, IR counts
        self.ac = np.array([0.0033, 0.006])  # pulse depth / DC; red/IR ratio 0.55 -> SpO2 ~98 %
        self.noise = 12.0  # counts
        self.auto = list(gestures or [])  # perform random gestures from this list by itself
        self.every = every
        self.truth = []  # (sample index, label) of every gesture performed
        self.info = {"fs": f"{fs:g}", "source": "sim"}
        self.status = "模擬訊號（--sim）"
        self._n = 0
        self._beats = [0.2]
        self._events = []  # (t0, bumps, level step)
        self._next_auto = 12.0  # leave the first seconds quiet for calibration
        self._drift_zi = np.zeros((2, 1))

    def trigger(self, label, delay=None):
        """Perform `label` after a human-like reaction delay (called at a cue)."""
        with self._lock:
            if delay is None:
                delay = self.rng.uniform(0.25, 0.6)
            self._schedule(label, self._n / self.fs + delay)

    def _schedule(self, label, t0):
        r = self.rng
        speed, strength = r.uniform(0.8, 1.25), r.uniform(0.6, 1.4)
        bumps = [(c * speed + r.normal(0, 0.03), w * speed * r.uniform(0.85, 1.15),
                  red * strength * r.uniform(0.85, 1.15), ir * strength * r.uniform(0.85, 1.15))
                 for c, w, red, ir in template(label)]
        self._events.append((t0, bumps, r.normal(0, 0.25, 2)))
        self.truth.append((int(round(t0 * self.fs)), label))

    def generate(self, n):
        """Return the next n samples as (red, ir) raw counts."""
        t = (self._n + np.arange(n)) / self.fs

        while self._beats[-1] < t[-1] + 1.0:
            tb = self._beats[-1]
            rr = 60.0 / self.hr * (1 + 0.05 * np.sin(2 * np.pi * 0.25 * tb) + 0.02 * self.rng.standard_normal())
            self._beats.append(tb + rr)
        self._beats = [b for b in self._beats if b > t[0] - 1.2]
        pulse = np.zeros(n)
        for tb in self._beats:
            d = t - tb
            m = (d > -0.1) & (d < 1.0)
            pulse[m] += pulse_shape(d[m])

        while self.auto and self._next_auto < t[-1]:
            self._schedule(self.rng.choice(self.auto), self._next_auto)
            self._next_auto += self.rng.uniform(*self.every)

        gesture = np.zeros((2, n))
        active = np.zeros(n)
        running = []
        for t0, bumps, step in self._events:
            d = t - t0
            for c, w, red, ir in bumps:
                b = np.exp(-((d - c) / w) ** 2)
                gesture[0] += red * b
                gesture[1] += ir * b
                active += b
            dd = np.maximum(d, 0)
            gesture += step[:, None] * (1 - np.exp(-dd / 0.15)) * np.exp(-dd / 1.5)
            if d[0] < 6.0:
                running.append((t0, bumps, step))
        self._events = running

        drift, self._drift_zi = lfilter([1.0], [1.0, -0.9995], self.rng.normal(0, 4e-6, (2, n)),
                                        axis=1, zi=self._drift_zi)
        breath = 0.0015 * np.sin(2 * np.pi * 0.25 * t)
        squeeze = 1 - 0.6 * np.clip(active, 0, 1)  # pressure flattens the pulse during a gesture
        rel = -self.ac[:, None] * pulse * squeeze + breath + drift + gesture / 100.0
        x = self.dc[:, None] * (1 + rel) + self.noise * self.rng.standard_normal((2, n))
        self._n += n
        return x[0], x[1]

    def _run(self):
        start, produced = time.monotonic(), 0
        while self._running:
            due = int((time.monotonic() - start) * self.fs * self.speed)
            if due > produced:
                with self._lock:
                    red, ir = self.generate(due - produced)
                    self._push(red, ir)
                produced = due
            time.sleep(0.01)
