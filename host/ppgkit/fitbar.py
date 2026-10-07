"""Live "how snug is the sensor" gauge for collect.py and predict.py.

The bar shows the IR level in % of full scale (more pressure or a smaller gap gives a
higher level). With a target (the level of the recording the model was trained on, or of
this session's calibration) the green band marks where the strap should sit; without one
the band is the generic 30-70 %. Matching the same level on every wearing is the point.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import dsp


class FitBar(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(54)
        self.dc = self.pi = None
        self.target = None  # {"dc": ..., "pi": ...} or None
        self.target_name = ""
        self._time = 0.0

    def set_target(self, target, name=""):
        self.target, self.target_name = target, name
        self.update()

    def update_from(self, red, ir, fs, now):
        """Refresh from the newest samples, at most 3 times a second."""
        if now - self._time < 0.3 or len(ir) < int(5 * fs):
            return
        self._time = now
        m = dsp.fit_metrics(red, ir, fs)
        self.dc, self.pi = m["dc"], m["pi"]
        self.update()

    def band(self):
        if self.target:
            t = self.target["dc"]
            return t * (1 - dsp.FIT_TOLERANCE), t * (1 + dsp.FIT_TOLERANCE)
        return 30.0, 70.0

    def verdict(self):
        if self.dc is None:
            return "等待資料…", "#888"
        lo, hi = self.band()
        if self.dc > 90:
            return "飽和了：調小 --led", "#ef5350"
        if self.dc < 10:
            return "沒貼到皮膚？", "#ef5350"
        if lo <= self.dc <= hi:
            return "貼合度 OK", "#66bb6a"
        word = "偏鬆（IR 偏低）：綁緊一點" if self.dc < lo else "偏緊（IR 偏高）：放鬆一點"
        return word, "#ffa726"

    def text(self):
        text, _ = self.verdict()
        if self.dc is None:
            return text
        goal = (f"　目標 {self.target['dc']:.0f} ±{100 * dsp.FIT_TOLERANCE:.0f}%（{self.target_name}）"
                if self.target else "　目標 30–70 %（還沒有參考，校正後會以這次為準）")
        pi = f"，脈搏 {self.pi:.2f} %" if self.pi is not None else ""
        return f"{text}　現在 IR {self.dc:.0f} %{pi}{goal}"

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width() - 20, 14
        x0, y0 = 10, 28
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor("#333"))
        p.drawRoundedRect(x0, y0, w, h, 4, 4)
        lo, hi = self.band()
        p.setBrush(QtGui.QColor(102, 187, 106, 120))
        p.drawRect(int(x0 + w * lo / 100), y0, max(2, int(w * (hi - lo) / 100)), h)
        if self.dc is not None:
            _, color = self.verdict()
            p.setBrush(QtGui.QColor(color))
            x = int(x0 + w * min(max(self.dc, 0), 100) / 100)
            p.drawPolygon(QtGui.QPolygon([QtCore.QPoint(x - 6, y0 - 6), QtCore.QPoint(x + 6, y0 - 6),
                                          QtCore.QPoint(x, y0 + 1)]))
        p.setPen(QtGui.QColor(self.verdict()[1]))
        font = p.font()
        font.setPointSize(13)
        p.setFont(font)
        p.drawText(QtCore.QRect(x0, 0, w, 20), QtCore.Qt.AlignmentFlag.AlignLeft, self.text())
