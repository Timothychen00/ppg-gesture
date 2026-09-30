"""即時心率儀表板：ESP32 + MAX30102 的 PPG、心跳、心率趨勢、HRV 與 SpO2 估計。

    python monitor.py                     # 自動找 ESP32 的序列埠
    python monitor.py --led 80            # 手腕訊號弱時加大 LED 電流
    python monitor.py --save rec.npz      # 關視窗時存下原始資料
    python monitor.py --sim               # 沒有硬體：模擬訊號
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ppgkit import dsp
from ppgkit.sources import GrowBuffer, add_source_args, open_source, save_session

WINDOW = 10.0  # s of PPG on screen
BEAT_WINDOW = 6.0  # s searched for new beats every frame
TREND = 300.0  # s of heart-rate history
CHANNELS = ("Red", "IR")


class Monitor(QtWidgets.QWidget):
    def __init__(self, src, save=None):
        super().__init__()
        self.src, self.save = src, save
        self.fs = src.fs
        self.buf = GrowBuffer(2)
        self.tracker = dsp.BeatTracker(self.fs)
        self.channel = 1  # beats are found on the channel with the stronger pulse
        self.last_slow = 0.0
        self.setWindowTitle("PPG 即時心率 — ESP32 + MAX30102")
        self.resize(1440, 880)

        glw = pg.GraphicsLayoutWidget()
        self.stats = QtWidgets.QLabel()
        self.stats.setFixedWidth(290)
        self.stats.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.stats.setStyleSheet("QLabel { background: #151515; color: #ddd; padding: 14px; font-size: 14px; }")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(glw, 1)
        layout.addWidget(self.stats)

        self.p_ppg = glw.addPlot(row=0, col=0, colspan=3, title="PPG（0.5–4 Hz 帶通、反相；紅點＝偵測到的心跳）")
        self.c_ppg = self.p_ppg.plot(pen=pg.mkPen("#4fc3f7", width=2))
        self.s_beats = pg.ScatterPlotItem(size=10, brush="#ff5252", pen=None)
        self.p_ppg.addItem(self.s_beats)
        self.p_ppg.setLabel("bottom", "時間", "s")

        self.p_raw = glw.addPlot(row=1, col=0, colspan=2, title="原始訊號（ADC counts）")
        self.p_raw.addLegend(offset=(8, 8))
        self.c_red = self.p_raw.plot(pen=pg.mkPen("#ef5350", width=1.5), name="Red 660 nm")
        self.c_ir = self.p_raw.plot(pen=pg.mkPen("#ce93d8", width=1.5), name="IR 880 nm")

        self.p_psd = glw.addPlot(row=1, col=2, title="PPG 頻譜（Welch）")
        self.c_psd = self.p_psd.plot(pen="#ffca28", fillLevel=0, brush=(255, 202, 40, 50))
        self.l_psd = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ff5252", style=QtCore.Qt.PenStyle.DashLine))
        self.p_psd.addItem(self.l_psd)
        self.p_psd.setLabel("bottom", "頻率", "Hz")

        self.p_hr = glw.addPlot(row=2, col=0, title="心率趨勢（每一拍）")
        self.c_hr = self.p_hr.plot(pen=pg.mkPen("#66bb6a55"), symbol="o", symbolSize=4,
                                   symbolBrush="#66bb6a", symbolPen=None)
        self.p_hr.setLabel("left", "bpm")
        self.p_hr.setLabel("bottom", "時間", "s")

        self.p_poi = glw.addPlot(row=2, col=1, title="Poincaré 圖（RRₙ → RRₙ₊₁）")
        self.s_poi = pg.ScatterPlotItem(size=5, brush=(79, 195, 247, 160), pen=None)
        self.p_poi.addItem(self.s_poi)
        self.c_diag = self.p_poi.plot(pen=pg.mkPen("#666", style=QtCore.Qt.PenStyle.DotLine))
        self.p_poi.setLabel("left", "RRₙ₊₁", "ms")
        self.p_poi.setLabel("bottom", "RRₙ", "ms")

        self.p_hrv = glw.addPlot(row=2, col=2, title="HRV 頻譜（Lomb-Scargle，需 ≥ 60 s）")
        for lo, hi, color in ((0.04, 0.15, (255, 167, 38, 40)), (0.15, 0.4, (79, 195, 247, 40))):
            self.p_hrv.addItem(pg.LinearRegionItem((lo, hi), movable=False, brush=color, pen=pg.mkPen(None)))
        self.c_hrv = self.p_hrv.plot(pen="#e0e0e0")
        self.p_hrv.setLabel("bottom", "頻率（LF 橘、HF 藍）", "Hz")
        self.p_hrv.getAxis("bottom").enableAutoSIPrefix(False)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    def tick(self):
        red, ir = self.src.read()
        if len(red):
            self.buf.extend(red, ir)
        n = len(self.buf)
        if n < 3 * self.fs:
            self.stats.setText(f"<b>等待資料…</b><br><br>{self.src.status}")
            return
        data = self.buf.data
        a = max(0, n - int(WINDOW * self.fs))
        t = np.arange(a, n) / self.fs
        self.c_red.setData(t, data[0, a:])
        self.c_ir.setData(t, data[1, a:])

        now = time.monotonic()
        slow = now - self.last_slow > 0.5
        if slow:
            self.last_slow = now
            self._pick_channel(data[:, max(0, n - int(5 * self.fs)):])
        b = max(0, n - int(BEAT_WINDOW * self.fs))
        self.tracker.update(data[self.channel, b:], n / self.fs)
        y = self.tracker.filter(data[self.channel, a:])
        self.c_ppg.setData(t, y)
        beats = np.asarray(self.tracker.beats)
        beats = beats[beats >= t[0]]
        self.s_beats.setData(beats, np.interp(beats, t, y))
        if slow:
            self._analysis(data, y, n)

    def _pick_channel(self, recent):
        pi = [dsp.perfusion_index(x, self.fs) for x in recent]
        other = 1 - self.channel
        if pi[other] > 1.3 * pi[self.channel]:
            self.channel = other

    def _analysis(self, data, y, n):
        fs = self.fs
        f, p = dsp.spectrum(y, fs)
        self.c_psd.setData(f, p)
        f_peak = f[np.argmax(p)] if len(p) else 0.0
        self.l_psd.setValue(f_peak)

        t_b, rr, ok = dsp.rr_series(self.tracker.beats)
        keep = t_b > n / fs - TREND
        t_b, rr, ok = t_b[keep], rr[keep], ok[keep]
        self.c_hr.setData(t_b[ok], 60.0 / rr[ok])
        pairs = ok[1:] & ok[:-1]
        ms = rr * 1000
        self.s_poi.setData(ms[:-1][pairs], ms[1:][pairs])
        if pairs.any():
            lo, hi = ms[ok].min() - 50, ms[ok].max() + 50
            self.c_diag.setData([lo, hi], [lo, hi])
        hrv = dsp.hrv_time(rr, ok)
        spec = dsp.hrv_spectrum(t_b, rr, ok)
        if spec:
            self.c_hrv.setData(spec[0], spec[1])

        recent = data[:, max(0, n - int(5 * fs)):]
        spo2, _ = dsp.spo2_estimate(recent[0], recent[1], fs)
        pi = dsp.perfusion_index(recent[self.channel], fs)
        quality = dsp.signal_quality(recent[1].mean(), pi)
        hr_now = 60.0 / np.median(rr[ok][-8:]) if ok.sum() >= 3 else None
        self.stats.setText(self._stats_html(hr_now, f_peak, hrv, spec, spo2, pi, quality, recent[1].mean(), n))

    def _stats_html(self, hr_now, f_peak, hrv, spec, spo2, pi, quality, dc_ir, n):
        def row(name, value):
            return f"<tr><td style='color:#999;padding-right:12px'>{name}</td><td align=right>{value}</td></tr>"

        hr_text = f"{hr_now:.0f}" if hr_now else "--"
        rows = [
            row("頻譜估心率", f"{f_peak * 60:.0f} bpm"),
            row("SpO₂（未校正）", f"{spo2:.0f} %" if spo2 and quality == "良好" else "--"),
            row("灌注指數 PI", f"{pi:.2f} %"),
            row("平均心率", f"{hrv['hr']:.1f} bpm" if "hr" in hrv else "--"),
            row("SDNN", f"{hrv['sdnn']:.0f} ms" if "sdnn" in hrv else "--"),
            row("RMSSD", f"{hrv['rmssd']:.0f} ms" if "rmssd" in hrv else "--"),
            row("pNN50", f"{hrv['pnn50']:.0f} %" if "pnn50" in hrv else "--"),
            row("SD1 / SD2", f"{hrv['sd1']:.0f} / {hrv['sd2']:.0f} ms" if "sd1" in hrv else "--"),
            row("LF / HF", f"{spec[2] / spec[3]:.2f}" if spec and spec[3] > 0 else "資料不足"),
            row("心跳數", str(len(self.tracker.beats))),
        ]
        color = "#66bb6a" if quality == "良好" else "#ffa726"
        return (
            "<div style='color:#999'>心率</div>"
            f"<div style='font-size:64px;font-weight:bold;color:#ff5252'>{hr_text}"
            "<span style='font-size:18px;color:#999'> bpm</span></div>"
            f"<table width=100%>{''.join(rows)}</table><br>"
            f"<div>訊號：<span style='color:{color}'>{quality}</span></div>"
            f"<div style='color:#999'>IR DC {dc_ir / 1000:.0f}k（滿格 {100 * dc_ir / dsp.FULL_SCALE:.0f}%）"
            f"<br>心跳用 {CHANNELS[self.channel]} 通道｜補值 {self.src.dropped} 點"
            f"<br>已錄 {n / self.fs:.0f} s</div><br>"
            f"<div style='color:#666;font-size:12px'>{self.src.status}</div>"
        )

    def closeEvent(self, event):
        self.timer.stop()
        if self.save and len(self.buf):
            red, ir = self.buf.data
            save_session(self.save, red, ir, self.fs, meta={"beats": self.tracker.beats, "tool": "monitor"})
            print(f"已存檔 {self.save}（{len(self.buf) / self.fs:.0f} s）")
        self.src.stop()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(parser)
    parser.add_argument("--save", metavar="FILE.npz", help="關閉視窗時把原始資料存成 .npz")
    args = parser.parse_args()

    app = pg.mkQApp("PPG monitor")
    src = open_source(args).start()
    win = Monitor(src, save=args.save)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
