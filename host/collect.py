"""手勢資料收集：照畫面提示做動作，自動錄下原始 PPG 和每次提示的時間點。

    python collect.py --subject tim                 # 5 個基本手勢，各 20 次
    python collect.py --preset asl --reps 40        # 論文 A 的 ASL 數字 1-9，各 40 次
    python collect.py --gestures fist,open,pinch    # 自訂手勢
    python collect.py --sim                         # 沒有硬體，先試流程

按鍵：空白鍵＝開始／暫停　Backspace＝刪掉上一筆（排到最後重做）　Esc＝存檔結束

畫面上每一次的標記：綠線＝提示（做！出現）、橘色虛線＝「做！」時間結束、綠色底色＝偵測到的動作範圍。
這些起訖點也會一起存進檔案（見 README 的錄音格式）。
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

from ppgkit import dsp
from ppgkit.fitbar import FitBar
from ppgkit.gesture import PRESETS, Detector, SegConfig, Trigger, calibrate, display_name, extract
from ppgkit.sources import GrowBuffer, add_source_args, load_session, open_source, save_session

CALIB, READY, GO, REST = 10.0, 1.5, 2.5, (2.0, 3.0)  # phase lengths in seconds
CALIB_MOTION = 1.5  # % motion during calibration that restarts it (see dsp.motion_percent)
VIEW = 9.0  # s of signal on screen, a bit more than one trial
Shortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut


class Collector(QtWidgets.QWidget):
    def __init__(self, src, labels, reps, path, meta, target=None, target_name=""):
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
        self.spans = {}  # cue index -> end markers of that trial, see ppgkit.sources.SPAN_KEYS
        self.current = None  # event of the trial in progress
        self.onsets = []  # live detector triggers, only used for feedback
        self.calib = None
        self.calib_start = 0
        self.det = self.trig = None
        self.phase, self.phase_end, self.paused = "wait", 0.0, False
        self.label = None
        self.note = ""
        self.check = None  # latest dsp.rest_check result, while waiting to start
        self.check_time = 0.0
        self.warned = False  # the user was told the checks fail and pressed Space once
        self.restarts = 0
        self.fit = FitBar()  # how snug the sensor sits; target = a reference recording, else this session's calibration
        self.fit.set_target(target, target_name)
        self.items = {}  # plot markers, keyed by (kind, cue index)

        self.setWindowTitle("PPG 手勢資料收集")
        self.resize(1100, 820)
        self.prompt = QtWidgets.QLabel()
        self.prompt.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sub = QtWidgets.QLabel()
        self.sub.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sub.setWordWrap(True)
        self.sub.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.sub.setStyleSheet("font-size: 17px; color: #bbb;")
        self.progress = QtWidgets.QLabel()
        self.progress.setStyleSheet("font-size: 14px; color: #999; padding: 6px;")
        self.progress.setWordWrap(True)

        glw = pg.GraphicsLayoutWidget()
        self.p_sig = glw.addPlot(row=0, col=0, title="相對於基準的變化（%）")
        self.p_sig.addLegend(offset=(8, -8))  # bottom left, clear of the cue labels at the top
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
        layout.addWidget(self.fit)
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
            bad = self.failed_checks()
            if bad and not self.warned:
                self.warned = True
                self.note = "還有項目沒過：" + "、".join(bad) + "。處理好再按空白鍵；仍要現在開始，請再按一次空白鍵。"
                return
            self.note = ""
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
        self.spans.pop(event[0], None)
        self.order.append(event[1])
        if event == self.current:
            self.current = None
        self.note = f"已刪除上一筆「{display_name(event[1])}」，排到最後重做"
        self.save()

    def failed_checks(self):
        names = {"contact": "感測器接觸", "pulse": "脈搏訊號", "still": "保持不動"}
        if self.check is None:
            return ["還沒有資料"]
        return [n for k, n in names.items() if not self.check[k][0]]

    def update_check(self):
        """Refresh the readiness check (about 3 times a second)."""
        now = time.monotonic()
        if now - self.check_time < 0.3 or len(self.buf) < int(5 * self.fs):
            return
        self.check_time = now
        red, ir = self.buf.data
        self.check = dsp.rest_check(red, ir, self.fs)
        if not self.failed_checks():
            self.warned, self.note = False, ""

    def watch_calibration(self):
        """Restart the rest calibration if the hand moved: a threshold measured while
        moving is too high and hides the gestures later."""
        n = len(self.buf)
        if n - self.calib_start < int(1.5 * self.fs):
            return
        red, ir = (x[self.calib_start:] for x in self.buf.data)
        if max(dsp.motion_percent(red, self.fs), dsp.motion_percent(ir, self.fs)) > CALIB_MOTION:
            self.restarts += 1
            self.calib_start = n
            self.set_phase("calib", CALIB)
            self.note = f"偵測到動作，校正重新計時（第 {self.restarts} 次）：請放鬆、完全不要動"

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
            if self.fit.target is None:  # no reference given: hold the strap at this session's calibration level
                self.fit.set_target(dsp.fit_metrics(red[c0:c1], ir[c0:c1], self.fs), "這次校正時的鬆緊")
            self.det, self.trig = Detector(self.cfg, cal), Trigger(self.cfg)
            self.note = ""
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
            if self.current:
                self.spans.setdefault(self.current[0], {})["go_end"] = self.src.total
            self.set_phase("rest", np.random.uniform(*REST))
        elif self.phase == "rest":
            if self.current:
                span = self.spans.setdefault(self.current[0], {})
                span["end"] = self.src.total
                span["move_start"], span["move_end"] = self.movement(self.current[0])
            self.save()
            self.next_trial()

    def movement(self, cue):
        """Start and end (sample indices) of the movement the live detector found
        for this cue, or (-1, -1). Uses the same segmentation as train.py's default."""
        near = [o for o in self.onsets if cue - 0.3 * self.fs <= o <= cue + 3.0 * self.fs]
        if near:
            red, ir = self.buf.data
            r = extract(red, ir, near[0], self.det.cal, self.cfg)
            if r is not None:
                return r["start"], r["end"]
        return -1, -1

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
        if self.phase in ("wait", "calib", "ready", "paused"):  # not while a gesture shifts the level
            self.fit.update_from(*self.buf.data, self.fs, time.monotonic())
        if self.phase == "wait":
            self.update_check()
        elif self.phase == "calib":
            self.watch_calibration()
        if self.src.stalled(1.5) and self.phase in ("ready", "go", "rest"):
            self.on_stall()
        enough = self.phase != "calib" or len(self.buf) - self.calib_start >= int(0.9 * CALIB * self.fs)
        if self.phase in ("calib", "ready", "go", "rest") and time.monotonic() >= self.phase_end and enough:
            self.advance()
        self.redraw()

    def on_stall(self):
        """The stream stopped (cable, board reset): drop the trial in progress and pause."""
        if self.phase in ("go", "rest") and self.events and self.events[-1] == self.current:
            self.on_undo()
        self.paused = True
        self.set_phase("paused")
        self.note = "⚠ 資料中斷（排線鬆了或板子重啟？）：這一筆不算。接好後按空白鍵繼續。"

    # ------------------------------------------------------- display ----

    def redraw(self):
        remaining = max(0.0, self.phase_end - time.monotonic())
        name = display_name(self.label) if self.label else ""
        texts = {
            "wait": self.wait_text(),
            "calib": ("放鬆不動", "#4fc3f7",
                      f"校正中，還有 {remaining:.0f} 秒：手放鬆平放桌上，不要動、不要說話。有動到會自動重新計時"),
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
        if self.phase == "wait" and self.note:
            sub = f'<span style="color:#ffa726">{self.note}</span><br><br>{sub}'
        elif self.note:
            sub = self.note
        self.sub.setText(sub)

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
        self._markers(t[0])

    def wait_text(self):
        """Title, colour and checklist for the screen before the calibration."""
        c = self.check
        if c is None:
            return ("等待資料…", "#888", "還沒收到資料：確認 ESP32 有接上、序列埠沒被別的程式佔用。")
        rows = []
        for key, label, value in (("contact", "感測器有貼好", f"IR 訊號 {c['dc']:.0f} % 滿格"),
                                  ("pulse", "看得到脈搏", f"脈搏幅度 {c['pi']:.2f} %"),
                                  ("still", "手保持不動", f"動作 {c['motion']:.1f} %，要小於 {dsp.STILL_LIMIT:g} %")):
            ok, msg = c[key]
            mark = '<span style="color:#66bb6a">✓</span>' if ok else '<span style="color:#ef5350">✗</span>'
            rows.append(f"{mark} {label}（{value}）{'　' + msg if msg else ''}")
        ready = not self.failed_checks()
        head = ("可以開始了：按空白鍵", "#66bb6a") if ready else ("先調整，全部打勾再按空白鍵", "#ffa726")
        tips = ("開始前：感測器綁在手腕內側（掌側），前臂平放桌上、手自然放鬆，感測器不要壓在桌面上。<br>"
                "接下來會有 10 秒的靜止校正，這段時間完全不要動；之後每次「做！」出現才動作。")
        return head[0], head[1], "<br>".join(rows) + "<br><br>" + tips

    def _markers(self, t0):
        """Green cue lines, dashed orange end-of-go lines and shaded detected movements
        for the trials that are still on screen (t0 = left edge in seconds)."""
        fs = self.fs
        green, orange = "#66bb6a", "#ffa726"
        wanted = {}
        for idx, label in self.events:
            span = self.spans.get(idx, {})
            if idx / fs >= t0:
                wanted[("cue", idx)] = lambda i=idx, l=label: pg.InfiniteLine(
                    i / fs, angle=90, pen=pg.mkPen(green, width=1),
                    label=display_name(l), labelOpts={"position": 0.92, "color": green})
            if span.get("go_end", -1) >= 0 and span["go_end"] / fs >= t0:
                wanted[("end", idx)] = lambda g=span["go_end"]: pg.InfiniteLine(
                    g / fs, angle=90, pen=pg.mkPen(orange, width=1, style=QtCore.Qt.PenStyle.DashLine),
                    label="結束", labelOpts={"position": 0.92, "color": orange})
            if span.get("move_start", -1) >= 0 and span["move_end"] / fs >= t0:
                wanted[("move", idx)] = lambda a=span["move_start"], b=span["move_end"]: self._region(a / fs, b / fs)
        for key in list(self.items):
            if key not in wanted:
                self.p_sig.removeItem(self.items.pop(key))
        for key, make in wanted.items():
            if key not in self.items:
                self.items[key] = make()
                self.p_sig.addItem(self.items[key])

    @staticmethod
    def _region(a, b):
        region = pg.LinearRegionItem((a, b), movable=False, brush=(102, 187, 106, 60), pen=pg.mkPen(None))
        region.setZValue(-10)  # behind the curves
        return region

    def save(self):
        if self.calib is None:
            return
        red, ir = self.buf.data
        meta = dict(self.meta, labels=self.labels, names={l: display_name(l) for l in self.labels},
                    saved=dt.datetime.now().isoformat(timespec="seconds"), source=self.src.info)
        spans = [self.spans.get(idx, {}) for idx, _ in self.events]
        save_session(self.path, red, ir, self.fs, self.events, self.calib, meta, spans)

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
    parser.add_argument("--match", metavar="FILE.npz", help="以這份錄音校正時的貼合度當目標（要用同樣的 --led / --range）")
    args = parser.parse_args()

    labels = [g.strip() for g in args.gestures.split(",") if g.strip()] if args.gestures else PRESETS[args.preset]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.subject}_{dt.datetime.now():%Y%m%d-%H%M%S}.npz")

    app = pg.mkQApp("PPG collect")
    src = open_source(args).start()
    target = name = None
    if args.match:
        ref = load_session(args.match)
        c0, c1 = ref["calib"]
        target, name = dsp.fit_metrics(ref["red"][c0:c1], ref["ir"][c0:c1], ref["fs"]), os.path.basename(args.match)
    win = Collector(src, labels, args.reps, path, {"subject": args.subject}, target, name)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
