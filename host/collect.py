"""手勢資料收集：照畫面提示做動作，自動錄下原始 PPG 和每次提示的時間點。

    python collect.py --subject tim                 # 5 個基本手勢，各 20 次
    python collect.py --preset asl --reps 40        # 論文 A 的 ASL 數字 1-9，各 40 次
    python collect.py --gestures fist,open,pinch    # 自訂手勢
    python collect.py --sim                         # 沒有硬體，先試流程

按鍵：空白鍵＝開始／暫停　Backspace＝刪掉上一筆（排到最後重做）　Esc＝存檔結束
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from collections import Counter

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ppgkit.gesture import PRESETS, Detector, SegConfig, Trigger, calibrate, display_name
from ppgkit.sources import GrowBuffer, add_source_args, open_source, save_session

CALIB, READY, GO, REST = 10.0, 1.5, 2.5, (2.0, 3.0)  # phase lengths in seconds
VIEW = 6.0  # s of signal on screen
Shortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut


class Collector(QtWidgets.QWidget):
    def __init__(self, src, labels, reps, path, meta):
        super().__init__()
        self.src, self.labels, self.reps, self.path, self.meta = src, labels, reps, path, meta
        self.fs = src.fs
        self.cfg = SegConfig(fs=self.fs)
        rng = np.random.default_rng()
        self.order = [label for _ in range(reps) for label in rng.permutation(labels)]
        self.pos = 0  # next entry of self.order
        self.buf = GrowBuffer(2)
        self.score = GrowBuffer(1)
        self.events = []  # (sample index of the cue, label)
        self.current = None  # event of the trial in progress
        self.onsets = []  # live detector triggers, only used for feedback
        self.calib = None
        self.calib_start = 0
        self.det = self.trig = None
        self.phase, self.phase_end, self.paused = "wait", 0.0, False
        self.label = None
        self.note = ""
        self.cue_lines = {}

        self.setWindowTitle("PPG 手勢資料收集")
        self.resize(1100, 820)
        self.prompt = QtWidgets.QLabel()
        self.prompt.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sub = QtWidgets.QLabel()
        self.sub.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sub.setWordWrap(True)
        self.sub.setStyleSheet("font-size: 17px; color: #bbb;")
        self.progress = QtWidgets.QLabel()
        self.progress.setStyleSheet("font-size: 14px; color: #999; padding: 6px;")
        self.progress.setWordWrap(True)

        glw = pg.GraphicsLayoutWidget()
        self.p_sig = glw.addPlot(row=0, col=0, title="相對於基準的變化（%）")
        self.p_sig.addLegend(offset=(8, 8))
        self.c_red = self.p_sig.plot(pen=pg.mkPen("#ef5350", width=1.5), name="Red")
        self.c_ir = self.p_sig.plot(pen=pg.mkPen("#ce93d8", width=1.5), name="IR")
        self.p_e = glw.addPlot(row=1, col=0, title="動作能量 ÷ 門檻（log₁₀，超過紅線＝偵測到動作）")
        self.c_e = self.p_e.plot(pen="#ffca28")
        self.p_e.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#ff5252", style=QtCore.Qt.PenStyle.DashLine)))
        self.p_e.setYRange(-2, 2)
        self.p_e.setXLink(self.p_sig)
        self.p_e.setLabel("bottom", "時間", "s")
        glw.ci.layout.setRowStretchFactor(0, 3)
        glw.ci.layout.setRowStretchFactor(1, 2)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.prompt)
        layout.addWidget(self.sub)
        layout.addWidget(glw, 1)
        layout.addWidget(self.progress)

        Shortcut(QtGui.QKeySequence("Space"), self, activated=self.on_space)
        Shortcut(QtGui.QKeySequence("Backspace"), self, activated=self.on_undo)
        Shortcut(QtGui.QKeySequence("Esc"), self, activated=self.close)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    # ---------------------------------------------------------- flow ----

    def set_phase(self, phase, seconds=0.0):
        self.phase, self.phase_end = phase, time.monotonic() + seconds

    def on_space(self):
        if self.phase == "wait":
            self.calib_start = len(self.buf)
            self.set_phase("calib", CALIB)
        elif self.phase == "paused":
            self.paused = False
            self.next_trial()
        elif self.phase in ("ready", "go", "rest"):
            self.paused = not self.paused

    def on_undo(self):
        if not self.events or self.phase in ("wait", "calib", "done"):
            return
        event = self.events.pop()
        self.order.append(event[1])
        if event == self.current:
            self.current = None
        self.note = f"已刪除上一筆「{display_name(event[1])}」，排到最後重做"
        self.save()

    def next_trial(self):
        if self.pos >= len(self.order):
            self.set_phase("done")
            self.save()
        elif self.paused:
            self.set_phase("paused")
        else:
            self.label = self.order[self.pos]
            self.set_phase("ready", READY)

    def advance(self):
        if self.phase == "calib":
            c0, c1 = self.calib_start, len(self.buf)
            red, ir = self.buf.data
            cal = calibrate(red[c0:c1], ir[c0:c1], self.cfg)
            self.calib = (c0, c1)
            self.det, self.trig = Detector(self.cfg, cal), Trigger(self.cfg)
            self.next_trial()
        elif self.phase == "ready":
            self.current = (self.src.total, self.label)  # cue time = every sample received so far
            self.events.append(self.current)
            self.pos += 1
            if hasattr(self.src, "trigger"):  # simulator performs the gesture itself
                self.src.trigger(self.label)
            self.note = ""
            self.set_phase("go", GO)
        elif self.phase == "go":
            self.set_phase("rest", np.random.uniform(*REST))
        elif self.phase == "rest":
            self.save()
            self.next_trial()

    def detected(self, event):
        cue = event[0]
        return any(cue - 0.3 * self.fs <= o <= cue + 3.0 * self.fs for o in self.onsets)

    def tick(self):
        red, ir = self.src.read()
        if len(red):
            start = len(self.buf)
            self.buf.extend(red, ir)
            if self.det is None:
                self.score.extend(np.zeros(len(red)))
            else:
                e = self.det.process(red, ir).max(axis=0)
                self.score.extend(e)
                self.onsets += self.trig.feed(e, start)
        if self.phase in ("calib", "ready", "go", "rest") and time.monotonic() >= self.phase_end:
            self.advance()
        self.redraw()

    # ------------------------------------------------------- display ----

    def redraw(self):
        remaining = max(0.0, self.phase_end - time.monotonic())
        name = display_name(self.label) if self.label else ""
        texts = {
            "wait": ("按空白鍵開始", "#fff",
                     "把 MAX30102 綁在手腕內側（掌側），手自然放桌上。先確認下面的波形有在動、沒有變成一條直線。"),
            "calib": ("放鬆不動", "#4fc3f7", f"校正中 {remaining:.0f} s：量測靜止時的雜訊，當作偵測門檻"),
            "ready": (f"準備：{name}", "#888", "看到「做！」就做一次、馬上放開，回到放鬆姿勢（不要一直維持）"),
            "go": (f"做！{name}", "#66bb6a", "做一次就放開"),
            "rest": ("放鬆", "#888", ""),
            "paused": ("暫停中", "#ffa726", "按空白鍵繼續"),
            "done": ("完成！", "#66bb6a", f"已存到 {self.path}　下一步：python train.py {self.path}"),
        }
        title, color, sub = texts[self.phase]
        self.prompt.setText(title)
        self.prompt.setStyleSheet(f"font-size: 60px; font-weight: bold; color: {color}; padding: 10px;")
        if self.current and self.phase in ("go", "rest"):
            if self.detected(self.current):
                sub = "✓ 有偵測到動作"
            elif self.phase == "rest":
                sub = "✗ 沒偵測到動作：做明顯一點，或把綁帶綁緊（Backspace 可刪掉重做）"
        self.sub.setText(self.note or sub)

        done = Counter(label for _, label in self.events)
        counts = "　".join(f"{display_name(l)} {done[l]}/{self.reps}" for l in self.labels)
        paused = "（這次做完會暫停）" if self.paused and self.phase in ("ready", "go", "rest") else ""
        self.progress.setText(f"第 {self.pos}/{len(self.order)} 次{paused}　｜　{counts}　｜　"
                              f"補值 {self.src.dropped} 點　｜　空白鍵 開始/暫停　Backspace 刪上一筆　Esc 存檔結束")

        n = len(self.buf)
        if n < 2:
            return
        a = max(0, n - int(VIEW * self.fs))
        data = self.buf.data[:, a:]
        rel = (data / np.median(data, axis=1, keepdims=True) - 1.0) * 100.0
        t = np.arange(a, n) / self.fs
        self.c_red.setData(t, rel[0])
        self.c_ir.setData(t, rel[1])
        self.c_e.setData(t, np.log10(np.maximum(self.score.data[0, a:n], 1e-2)))
        self._cue_lines(t[0])

    def _cue_lines(self, t0):
        visible = {idx: label for idx, label in self.events if idx / self.fs >= t0}
        for idx in list(self.cue_lines):
            if idx not in visible:
                self.p_sig.removeItem(self.cue_lines.pop(idx))
        for idx, label in visible.items():
            if idx not in self.cue_lines:
                line = pg.InfiniteLine(idx / self.fs, angle=90, pen=pg.mkPen("#66bb6a", width=1),
                                       label=display_name(label), labelOpts={"position": 0.92, "color": "#66bb6a"})
                self.p_sig.addItem(line)
                self.cue_lines[idx] = line

    def save(self):
        if self.calib is None:
            return
        red, ir = self.buf.data
        meta = dict(self.meta, labels=self.labels, names={l: display_name(l) for l in self.labels},
                    saved=dt.datetime.now().isoformat(timespec="seconds"), source=self.src.info)
        save_session(self.path, red, ir, self.fs, self.events, self.calib, meta)

    def closeEvent(self, event):
        self.timer.stop()
        self.save()
        if self.calib is not None:
            print(f"已存檔 {self.path}：{len(self.events)} 次手勢")
        self.src.stop()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(parser)
    parser.add_argument("--subject", default="me", help="受試者名稱（會放進檔名）")
    parser.add_argument("--preset", choices=PRESETS, default="basic", help="basic＝5 個基本手勢；asl＝論文 A 的 ASL 1-9")
    parser.add_argument("--gestures", help="自訂手勢，逗號分隔，例如 fist,open,pinch（優先於 --preset）")
    parser.add_argument("--reps", type=int, default=20, help="每個手勢做幾次（論文 A：40 次）")
    parser.add_argument("--out", default="data", help="存檔資料夾")
    args = parser.parse_args()

    labels = [g.strip() for g in args.gestures.split(",") if g.strip()] if args.gestures else PRESETS[args.preset]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.subject}_{dt.datetime.now():%Y%m%d-%H%M%S}.npz")

    app = pg.mkQApp("PPG collect")
    src = open_source(args).start()
    win = Collector(src, labels, args.reps, path, {"subject": args.subject})
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
