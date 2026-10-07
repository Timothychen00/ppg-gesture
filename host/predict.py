"""即時手勢辨識：用 train.py 訓練的模型，辨識手腕 PPG 上的手勢。

    python predict.py                           # ESP32 + models/gesture_model.joblib
    python predict.py --sim                     # 模擬訊號，會自己隨機做手勢（畫面會對答案）
    python predict.py --replay data/xxx.npz     # 拿錄好的資料重跑一次

開始時先放鬆不動 8 秒做校正。按鍵：C＝重新校正　Esc＝離開
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ppgkit.fitbar import FitBar
from ppgkit import dsp
from ppgkit.gesture import LivePredictor, display_name, threshold_percent
from ppgkit.sources import add_source_args, open_source

VIEW = 8.0  # s of signal on screen
CALIB_MOTION = 1.5  # % motion during the rest calibration that restarts it (same rule as collect.py)
Shortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut


class PredictWindow(QtWidgets.QWidget):
    def __init__(self, src, bundle, min_conf=0.5):
        super().__init__()
        self.src, self.min_conf = src, min_conf
        self.fs = src.fs
        self.lp = LivePredictor(bundle, calib_range=getattr(src, "calib", None))
        self.truth = getattr(src, "truth", None)  # simulator / replay know the real gestures
        self.matched = self.correct = self.false = 0
        self.restarts = 0  # calibrations restarted because the hand moved
        self.last_event = ""  # what the detector did most recently, shown in the status panel
        self.watch_calib = getattr(src, "calib", None) is None  # a replay brings its own rest period
        self.marks = []  # plot items of recognised segments

        self.setWindowTitle(f"PPG 手勢辨識 — 模型 {bundle.get('kind', '')}")
        self.resize(1300, 760)
        self.fit = FitBar()
        self.fit.set_target(bundle.get("fit"), "訓練資料的鬆緊")
        glw = pg.GraphicsLayoutWidget()
        self.p_sig = glw.addPlot(row=0, col=0, title="相對於基準的變化（%）")
        self.p_sig.addLegend(offset=(8, 8))
        self.c_red = self.p_sig.plot(pen=pg.mkPen("#ef5350", width=1.5), name="Red")
        self.c_ir = self.p_sig.plot(pen=pg.mkPen("#ce93d8", width=1.5), name="IR")
        self.p_e = glw.addPlot(row=1, col=0, title="動作能量 ÷ 門檻（log₁₀）")
        self.c_e = self.p_e.plot(pen="#ffca28")
        self.p_e.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#ff5252", style=QtCore.Qt.PenStyle.DashLine)))
        self.p_e.setYRange(-2, 2)
        self.p_e.setXLink(self.p_sig)
        self.p_e.setLabel("bottom", "時間", "s")
        glw.ci.layout.setRowStretchFactor(0, 3)
        glw.ci.layout.setRowStretchFactor(1, 2)

        side = QtWidgets.QWidget()
        side.setFixedWidth(340)
        col = QtWidgets.QVBoxLayout(side)
        self.big = QtWidgets.QLabel("—")
        self.big.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.big.setWordWrap(True)
        self.big.setStyleSheet("font-size: 44px; font-weight: bold; color: #66bb6a;")
        self.conf = QtWidgets.QLabel()
        self.conf.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.conf.setStyleSheet("font-size: 16px; color: #bbb;")
        self.lamp = QtWidgets.QLabel("—")
        self.lamp.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lamp.setWordWrap(True)
        self.lamp.setStyleSheet("font-size: 15px; color: #888;")
        col.addWidget(self.big)
        col.addWidget(self.conf)
        col.addWidget(self.lamp)
        self.bars = {}
        grid = QtWidgets.QGridLayout()
        for i, label in enumerate(self.lp.classes):
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("%p%")
            grid.addWidget(QtWidgets.QLabel(display_name(label)), i, 0)
            grid.addWidget(bar, i, 1)
            self.bars[label] = bar
        col.addLayout(grid)
        col.addWidget(QtWidgets.QLabel("紀錄"))
        self.history = QtWidgets.QListWidget()
        col.addWidget(self.history, 1)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #999;")
        col.addWidget(self.status)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.fit)
        left.addWidget(glw, 1)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addLayout(left, 1)
        layout.addWidget(side)
        Shortcut(QtGui.QKeySequence("C"), self, activated=self.lp.recalibrate)
        Shortcut(QtGui.QKeySequence("Esc"), self, activated=self.close)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    def tick(self):
        red, ir = self.src.read()
        if len(red):
            for result in self.lp.feed(red, ir):
                self.show_result(result)
        self.watch_calibration()
        self.update_lamp()
        quiet = self.lp.cal is None or float(self.lp.score.data[0, -int(3 * self.fs):].max(initial=0)) < 1.0
        if quiet:  # a gesture shifts the level for a moment, so only measure the fit at rest
            self.fit.update_from(*self.lp.buf.data, self.fs, time.monotonic())
        self.redraw()

    def watch_calibration(self):
        """Restart the calibration while the hand moves: a threshold measured on a moving hand
        is too high and hides the gestures. (Not for replays, which bring their own rest period.)"""
        lp = self.lp
        if lp.cal is not None or not self.watch_calib:
            return
        n = len(lp.buf)
        if n - lp.cal_start < int(1.5 * self.fs):
            return
        red, ir = (x[lp.cal_start:] for x in lp.buf.data)
        if max(dsp.motion_percent(red, self.fs), dsp.motion_percent(ir, self.fs)) > CALIB_MOTION:
            self.restarts += 1
            lp.recalibrate()

    def update_lamp(self):
        """One line that says what the detector is doing right now, so a silent screen can be diagnosed."""
        lp = self.lp
        if lp.cal is None:
            self.lamp.setText(f"校正中（重新計時 {self.restarts} 次）" if self.restarts else "校正中：手放鬆不動")
            return
        n = len(lp.buf)
        recent = float(lp.score.data[0, -int(0.5 * self.fs):].max(initial=0.0))
        if recent > 1.0:
            text, color = f"偵測到動作（{recent:.0f}× 門檻），等 {lp.cfg.wait:.0f} 秒收完整個動作再辨識…", "#ffca28"
        elif lp.pending:
            left = (lp.pending[0] + int(lp.cfg.wait * self.fs) - n) / self.fs
            text, color = f"收集動作中，還要 {max(left, 0):.1f} 秒…", "#ffca28"
        else:
            text, color = (f"待命：門檻 ≈ {threshold_percent(lp.cal, lp.cfg):.1f}% 的位準變化，"
                           f"做大於這個幅度的動作才會偵測", "#888")
        self.lamp.setStyleSheet(f"font-size: 15px; color: {color};")
        self.lamp.setText(text + (f"<br>{self.last_event}" if self.last_event else ""))

    def show_result(self, r):
        peak = float(self.lp.score.data[0, max(0, r["onset"] - 5):r["onset"] + int(1.5 * self.fs)].max(initial=0.0))
        top = "  ".join(f"{l} {p:.0%}" for p, l in sorted(zip(r["proba"], self.lp.classes), reverse=True)[:3])
        print(f"[{time.strftime('%H:%M:%S')}] 動作 {r['onset'] / self.fs:.1f}s（最大 {peak:.0f}× 門檻）→ "
              f"{r['label']}{'（忽略）' if r['reject'] else ''}  {top}", flush=True)
        real = []
        if self.truth is not None:
            real = [label for idx, label in self.truth if -2.0 * self.fs < idx - r["onset"] < 0.5 * self.fs]
        if r["reject"] and not real:  # model says "not a gesture" (e.g. releasing the hand): ignore it
            best = max((p, l) for p, l in zip(r["proba"], self.lp.classes) if l != "none")
            self.history.insertItem(0, f"{time.strftime('%H:%M:%S')}  （無手勢，忽略；最像 {display_name(best[1])} {best[0]:.0%}）")
            self.last_event = "上一次動作被判成「無手勢」（放開手勢或雜動作），沒有輸出"
            for label, p in zip(self.lp.classes, r["proba"]):
                self.bars[label].setValue(int(round(100 * p)))
            return
        self.last_event = ""
        sure = r["conf"] >= self.min_conf and not r["reject"]
        name = display_name(r["label"])
        self.big.setText(name if sure else "？")
        self.big.setStyleSheet(f"font-size: 44px; font-weight: bold; color: {'#66bb6a' if sure else '#888'};")
        text = f"信心 {r['conf']:.0%}" + ("" if sure else f"（最像：{name}）")
        for label, p in zip(self.lp.classes, r["proba"]):
            self.bars[label].setValue(int(round(100 * p)))

        entry = f"{time.strftime('%H:%M:%S')}  {name}  {r['conf']:.0%}"
        if self.truth is not None:
            if real:
                self.matched += 1
                self.correct += real[0] == r["label"]
                mark = "✓" if real[0] == r["label"] else f"✗ 實際 {display_name(real[0])}"
                entry += f"  {mark}"
                text += f"｜{mark}"
            else:
                self.false += 1
                entry += "  ✗ 沒有做手勢"
        self.conf.setText(text)
        self.history.insertItem(0, entry)

        region = pg.LinearRegionItem((r["start"] / self.fs, r["end"] / self.fs), movable=False,
                                     brush=(102, 187, 106, 50), pen=pg.mkPen(None))
        tag = pg.TextItem(name if sure else "？", color="#66bb6a", anchor=(0, 0))
        tag.setPos(r["start"] / self.fs, float(np.max(r["seg"])))
        self.p_sig.addItem(region)
        self.p_sig.addItem(tag)
        self.marks.append((r["end"] / self.fs, region, tag))

    def redraw(self):
        lp = self.lp
        n = len(lp.buf)
        if lp.cal is None:
            self.status.setText(f"校正中：請放鬆不動… {lp.calib_progress:.0%}<br>{self.src.status}")
        else:
            score = ""
            if self.truth is not None:
                wait = (lp.cfg.wait + 0.5) * self.fs
                seen = sum(1 for idx, _ in self.truth if idx + wait <= n)
                score = (f"｜做了 {seen} 次手勢：抓到 {self.matched}、答對 {self.correct}、"
                         f"誤觸發 {self.false}")
            self.status.setText(f"偵測中（C 重新校正）{score}<br>補值 {self.src.dropped} 點｜{self.src.status}")
        if n < 2:
            return
        a = max(0, n - int(VIEW * self.fs))
        data = lp.buf.data[:, a:]
        rel = (data / np.median(data, axis=1, keepdims=True) - 1.0) * 100.0
        t = np.arange(a, n) / self.fs
        self.c_red.setData(t, rel[0])
        self.c_ir.setData(t, rel[1])
        self.c_e.setData(t, np.log10(np.maximum(lp.score.data[0, a:n], 1e-2)))
        while self.marks and self.marks[0][0] < t[0]:
            _, region, tag = self.marks.pop(0)
            self.p_sig.removeItem(region)
            self.p_sig.removeItem(tag)

    def closeEvent(self, event):
        self.timer.stop()
        self.src.stop()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(parser)
    parser.add_argument("--model", default="models/gesture_model.joblib", help="train.py 存的模型")
    parser.add_argument("--min-conf", type=float, default=0.5, help="信心低於這個值就顯示「？」")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    app = pg.mkQApp("PPG predict")
    src = open_source(args, sim_gestures=bundle["labels"]).start()
    win = PredictWindow(src, bundle, args.min_conf)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
