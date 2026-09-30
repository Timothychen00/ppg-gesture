"""Heart-rate analysis on raw PPG: filtering, beat detection, HRV and an SpO2 estimate."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, lombscargle, sosfiltfilt, welch

FULL_SCALE = 2 ** 18 - 1  # 18-bit ADC


def bandpass(fs, lo=0.5, hi=4.0, order=3):
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def _vertex(y, k):
    """Sub-sample offset of the peak at k (parabola through 3 points)."""
    a, b, c = y[k - 1], y[k], y[k + 1]
    den = a - 2 * b + c
    return 0.0 if den == 0 else 0.5 * (a - c) / den


class BeatTracker:
    """Finds heartbeats in a sliding window and keeps a de-duplicated list of beat times.

    Peaks within 0.4 s of the newest sample are left for a later call because the
    zero-phase filter has not settled there yet.
    """

    def __init__(self, fs, band=(0.5, 4.0)):
        self.fs = fs
        self.sos = bandpass(fs, *band)
        self.beats: list[float] = []

    def filter(self, x):
        """Band-pass and invert, so the systolic peak points up (more blood = less light back)."""
        return -sosfiltfilt(self.sos, x - x.mean())

    def update(self, x, t_end):
        """x: newest raw samples of one channel; the sample after the last one is at t_end (s)."""
        y = self.filter(x)
        amp = np.percentile(y, 95) - np.percentile(y, 5)
        peaks, _ = find_peaks(y, distance=int(0.33 * self.fs), prominence=0.3 * amp)
        t0 = t_end - len(x) / self.fs
        for k in peaks:
            if k == 0 or k == len(y) - 1:
                continue
            tk = t0 + (k + _vertex(y, k)) / self.fs
            if tk > t_end - 0.4 or (self.beats and tk < self.beats[-1] + 0.3):
                continue
            self.beats.append(tk)
        return y


def clean_rr(rr):
    """Mask of plausible RR intervals (s): 0.33-2 s and within 25 % of the local median."""
    ok = (rr > 0.33) & (rr < 2.0)
    if len(rr) >= 5:
        med = median_filter(rr, 5, mode="nearest")
        ok &= np.abs(rr - med) < 0.25 * med
    return ok


def rr_series(beats):
    """Returns beat times (end of each interval), RR intervals (s) and the clean mask."""
    b = np.asarray(beats, float)
    rr = np.diff(b)
    return b[1:], rr, clean_rr(rr)


def hrv_time(rr, ok):
    """Time-domain HRV. Successive differences only use pairs of clean intervals."""
    good = rr[ok]
    if len(good) < 3:
        return {}
    ms = rr * 1000.0
    pairs = ok[1:] & ok[:-1]
    d = np.diff(ms)[pairs]
    sdnn = ms[ok].std(ddof=1)
    out = {"hr": 60.0 / good.mean(), "mean_rr": ms[ok].mean(), "sdnn": sdnn}
    if len(d) >= 2:
        sd1 = np.sqrt(0.5) * d.std(ddof=1)
        out.update(rmssd=np.sqrt(np.mean(d ** 2)), pnn50=100.0 * np.mean(np.abs(d) > 50),
                   sd1=sd1, sd2=np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0.0)))
    return out


def hrv_spectrum(t, rr, ok, min_span=60.0):
    """Lomb-Scargle spectrum of the (unevenly sampled) RR series, LF and HF power.

    Short-term HRV standards ask for 5 minutes; below min_span seconds this returns None.
    """
    t, rr = t[ok], rr[ok]
    if len(rr) < 20 or t[-1] - t[0] < min_span:
        return None
    f = np.linspace(0.02, 0.5, 240)
    p = lombscargle(t, rr - rr.mean(), 2 * np.pi * f)
    lf = p[(f >= 0.04) & (f < 0.15)].sum()
    hf = p[(f >= 0.15) & (f < 0.40)].sum()
    return f, p, lf, hf


def spo2_estimate(red, ir, fs):
    """Ratio-of-ratios SpO2 with the curve from Maxim's reference design.

    Not calibrated for this board, so treat it as a trend indicator, not a medical value.
    """
    sos = bandpass(fs)
    ac_r = sosfiltfilt(sos, red - red.mean()).std()
    ac_i = sosfiltfilt(sos, ir - ir.mean()).std()
    dc_r, dc_i = red.mean(), ir.mean()
    if min(ac_r, ac_i, dc_r, dc_i) <= 0:
        return None, None
    r = (ac_r / dc_r) / (ac_i / dc_i)
    return float(np.clip(-45.060 * r * r + 30.354 * r + 94.845, 0, 100)), float(r)


def perfusion_index(x, fs):
    """Pulse amplitude as % of the DC level."""
    y = sosfiltfilt(bandpass(fs), x - x.mean())
    return 100.0 * (np.percentile(y, 95) - np.percentile(y, 5)) / max(x.mean(), 1.0)


def spectrum(y, fs, fmax=5.0):
    f, p = welch(y, fs, nperseg=min(len(y), int(8 * fs)), nfft=4096)
    keep = f <= fmax
    return f[keep], p[keep]


def signal_quality(dc_ir, pi):
    if dc_ir < 20000:
        return "沒貼到皮膚？IR 太低"
    if dc_ir > 0.9 * FULL_SCALE:
        return "飽和：調低 --led 或加大 --range"
    if pi < 0.05:
        return "脈搏很弱：貼緊、遮光"
    return "良好"
