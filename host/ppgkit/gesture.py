"""PPG gesture recognition pipeline.

Follows Zhao et al., "PPG-based Finger-level Gesture Recognition Leveraging
Wearables", IEEE INFOCOM 2018 ("paper A"), plus the envelope + SVM / 1-NN
baseline of "Principal component analysis of photoplethysmography signals for
improved gesture recognition" (PMC9669422, MAX30105 on the wrist, "paper B").

    raw red / IR at 200 Hz
      -> coarse detection: 2 Hz high-pass, short-time energy, threshold
         mu + 3 sigma measured while the hand rests           (A, sec. VII)
      -> fine segmentation of the gesture                     (A, sec. V-A)
      -> features: statistics, FFT < 5 Hz, Haar DWT level 3,
         Wigner-Ville, AR coefficients, red/IR cross-correlation,
         2D-DTW distance to gesture profiles                  (A, table I)
      -> one binary gradient-boosting classifier per gesture  (A, sec. VI-B)

The paper used two green-LED sensors side by side; here the two channels are
the MAX30102's red and IR light, which reach different tissue depths.

The same Detector / Trigger / extract() code runs over whole recordings in
training and chunk by chunk in predict.py, so live segments match training.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfilt, sosfilt_zi, sosfiltfilt
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .sources import GrowBuffer

GESTURES = {
    "fist": "握拳",
    "open": "五指張開",
    "pinch": "拇指食指捏",
    "index": "食指彎曲",
    "wrist_up": "手腕上翹",
    "wrist_down": "手腕下壓",
    # paper A's gesture set: American Sign Language numbers 1-9
    "asl1": "ASL 1：食指伸直",
    "asl2": "ASL 2：食指中指 V",
    "asl3": "ASL 3：拇指＋食指＋中指",
    "asl4": "ASL 4：四指伸直、拇指收",
    "asl5": "ASL 5：五指張開",
    "asl6": "ASL 6：拇指碰小指",
    "asl7": "ASL 7：拇指碰無名指",
    "asl8": "ASL 8：拇指碰中指",
    "asl9": "ASL 9：拇指碰食指",
}
PRESETS = {
    "basic": ["fist", "open", "pinch", "wrist_up", "wrist_down"],
    "asl": [f"asl{i}" for i in range(1, 10)],
}


def display_name(label):
    return GESTURES.get(label, label)


@dataclass
class SegConfig:
    fs: float = 200.0  # input rate (firmware)
    feat_fs: float = 100.0  # features are computed at 100 Hz like paper A's prototype
    hp_cut: float = 2.0  # A: high-pass before the short-time energy
    lp_cut: float = 5.0  # B: gesture information sits in 0-5 Hz
    energy_win: float = 0.25  # s
    threshold: str = "robust"  # "robust": median + k_mad * MAD of the rest energy; "paper": mean + k_sigma * std
    k_mad: float = 10.0
    k_sigma: float = 3.0  # A: threshold = mu + 3 sigma of the energy at rest
    min_above: float = 0.03  # s above threshold before a trigger
    pre: float = 1.0  # s kept before the trigger
    wait: float = 3.0  # s collected after the trigger, long enough for action + release (A uses Wc = 4.5 s)
    quiet: float = 0.3  # s below threshold before the next trigger may fire
    mode: str = "threshold"  # "threshold": onset/offset of the energy burst, "energy": A's eq. (1)
    win: float = 1.2  # A: W, typical gesture length, used by mode "energy"
    bridge: float = 1.0  # s; quieter gaps inside one movement are bridged (e.g. action + release)
    min_len: float = 0.5
    max_len: float = 3.0
    pad: float = 0.1


@dataclass
class Calibration:
    dc: np.ndarray  # mean raw level at rest (red, IR)
    tau: np.ndarray  # energy threshold per channel


# ------------------------------------------------------------ detection ----

class Detector:
    """Causal short-time energy of the high-passed relative signal, divided by
    the calibration threshold: values above 1 mean "something is moving"."""

    def __init__(self, cfg: SegConfig, cal: Calibration):
        self.cal = cal
        self.sos = butter(4, cfg.hp_cut, "highpass", fs=cfg.fs, output="sos")
        self.w = max(1, int(round(cfg.energy_win * cfg.fs)))
        self.zi = None
        self.tail = np.zeros((2, self.w))

    def process(self, red, ir):
        """Returns the (2, n) normalised energy of the new samples."""
        u = np.vstack([red, ir]) / self.cal.dc[:, None] - 1.0
        n = u.shape[1]
        if n == 0:
            return np.zeros((2, 0))
        if self.zi is None:  # start in steady state instead of from zero
            self.zi = sosfilt_zi(self.sos)[:, None, :] * u[None, :, :1]
        h, self.zi = sosfilt(self.sos, u, axis=1, zi=self.zi)
        p = np.concatenate([self.tail, h ** 2], axis=1)
        c = np.concatenate([np.zeros((2, 1)), np.cumsum(p, axis=1)], axis=1)
        e = (c[:, self.w + 1:] - c[:, 1:n + 1]) / self.w
        self.tail = p[:, -self.w:]
        return e / self.cal.tau[:, None]


class Trigger:
    """Turns the energy score into gesture onsets (absolute sample indices)."""

    def __init__(self, cfg: SegConfig):
        self.need = max(1, int(round(cfg.min_above * cfg.fs)))
        self.hold = int(round(cfg.wait * cfg.fs))
        self.quiet_need = int(round(cfg.quiet * cfg.fs))
        self.armed = True
        self.run = 0
        self.quiet = 0
        self.blocked_until = -1

    def feed(self, score, start):
        """score[j] belongs to sample start + j; returns the onsets found."""
        onsets = []
        for j, s in enumerate(score):
            i = start + j
            if not self.armed:
                self.quiet = self.quiet + 1 if s < 1.0 else 0
                if i >= self.blocked_until and self.quiet >= self.quiet_need:
                    self.armed, self.run = True, 0
            elif s > 1.0:
                self.run += 1
                if self.run >= self.need:
                    onset = i - self.run + 1
                    onsets.append(onset)
                    self.armed, self.quiet = False, 0
                    self.blocked_until = onset + self.hold
            else:
                self.run = 0
        return onsets


def calibrate(red, ir, cfg: SegConfig) -> Calibration:
    """Energy threshold from a rest period.

    Paper A uses mu + 3 sigma. On real recordings one twitch during the rest
    period inflates sigma enough to hide most gestures, so the default uses
    the median and MAD instead, which ignore short movements.
    """
    x = np.vstack([red, ir]).astype(float)
    dc = x.mean(axis=1)
    e = Detector(cfg, Calibration(dc, np.ones(2))).process(x[0], x[1])
    e = e[:, int(1.0 * cfg.fs):]  # skip the filter start-up
    if cfg.threshold == "paper":
        tau = e.mean(axis=1) + cfg.k_sigma * e.std(axis=1)
    else:
        med = np.median(e, axis=1)
        tau = med + cfg.k_mad * 1.4826 * np.median(np.abs(e - med[:, None]), axis=1)
    return Calibration(dc, np.maximum(tau, 1e-12))


# --------------------------------------------------------- segmentation ----

def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _burst_window(score, k0, cfg):
    """Onset/offset of the energy burst nearest to the trigger (hysteresis 1.0 / 0.5)."""
    fs = cfg.fs
    runs = []
    for s, t in _runs(score > 1.0):  # bridge short gaps so one movement stays one burst
        if runs and s - runs[-1][1] < cfg.bridge * fs:
            runs[-1] = (runs[-1][0], t)
        else:
            runs.append((s, t))
    if not runs:
        return k0, k0 + int(cfg.win * fs)
    s, t = min(runs, key=lambda r: 0 if r[0] <= k0 < r[1] else min(abs(k0 - r[0]), abs(k0 - r[1])))
    while s > 0 and score[s - 1] > 0.5:
        s -= 1
    while t < len(score) and score[t] > 0.5:
        t += 1
    s = max(0, s - int(cfg.pad * fs))
    t = min(len(score), t + int(cfg.pad * fs))
    t = min(max(t, s + int(cfg.min_len * fs)), s + int(cfg.max_len * fs), len(score))
    return s, t


def _paper_window(score, cfg):
    """Paper A eq. (1): the length-W window whose energy exceeds the threshold the most."""
    w = int(round(cfg.win * cfg.fs))
    if len(score) <= w:
        return 0, len(score)
    c = np.concatenate([[0.0], np.cumsum(score - 1.0)])
    s = int(np.argmax(c[w:] - c[:-w]))
    return s, s + w


def extract(red, ir, onset, cal: Calibration, cfg: SegConfig):
    """Cut out the gesture that triggered at sample `onset`.

    Needs samples [onset - pre, onset + wait]. Returns None if they are not
    all there, else a dict with "seg": (2, m) relative change in % at
    cfg.feat_fs, and "start" / "end" as absolute sample indices at cfg.fs.
    """
    fs = cfg.fs
    a, b = onset - int(round(cfg.pre * fs)), onset + int(round(cfg.wait * fs))
    if a < 0 or b > len(red):
        return None
    x = np.vstack([red[a:b], ir[a:b]]).astype(float)
    base = np.median(x[:, :int(round((cfg.pre - 0.3) * fs))], axis=1)  # just before the movement
    u = x / base[:, None] - 1.0

    # energy again, zero-phase this time since the whole window is available
    h = sosfiltfilt(butter(4, cfg.hp_cut, "highpass", fs=fs, output="sos"), u, axis=1)
    w = max(1, int(round(cfg.energy_win * fs)))
    score = (uniform_filter1d(h ** 2, w, axis=1, mode="nearest") / cal.tau[:, None]).max(axis=0)
    if cfg.mode == "energy":
        s, t = _paper_window(score, cfg)
    else:
        s, t = _burst_window(score, onset - a, cfg)

    v = 100.0 * sosfiltfilt(butter(4, cfg.lp_cut, "lowpass", fs=fs, output="sos"), u, axis=1)
    step = max(1, int(round(fs / cfg.feat_fs)))
    return {"seg": v[:, s:t:step], "start": a + s, "end": a + t}


def segment_session(session, cfg: SegConfig):
    """Calibrate on the session's rest period, run the live detector over it and
    cut one segment per cue. Returns (segments, labels, report)."""
    red, ir = session["red"], session["ir"]
    if abs(session["fs"] - cfg.fs) > 1e-6:
        raise ValueError(f"sample rate {session['fs']} Hz != {cfg.fs} Hz")
    if session["calib"] is None:
        raise ValueError("session has no rest calibration")
    c0, c1 = session["calib"]
    cal = calibrate(red[c0:c1], ir[c0:c1], cfg)
    det, trig = Detector(cfg, cal), Trigger(cfg)
    onsets = trig.feed(det.process(red[c1:], ir[c1:]).max(axis=0), c1)

    segs, labels, used = [], [], set()
    for cue, label in session["events"]:
        near = [o for o in onsets if cue - 0.3 * cfg.fs <= o <= cue + 3.0 * cfg.fs and o not in used]
        if not near:
            continue
        r = extract(red, ir, near[0], cal, cfg)
        if r is None or r["seg"].shape[1] < 8:
            continue
        used.add(near[0])
        segs.append(r["seg"])
        labels.append(label)
    report = {"cues": len(session["events"]), "segments": len(segs),
              "missed": len(session["events"]) - len(segs), "extra_triggers": len(onsets) - len(used)}
    return segs, labels, report


def as_objects(segs):
    """List of (2, m) arrays -> 1-D object array (what the model pipelines take)."""
    out = np.empty(len(segs), dtype=object)
    for i, s in enumerate(segs):
        out[i] = s
    return out


# ------------------------------------------------------------- features ----

PAPER_A = ("stats", "fft", "dwt", "wvd", "ar", "xcorr", "dtw")
PAPER_B = ("env",)
XCORR_LAGS = (-8, -6, -4, -2, 0, 2, 4, 6, 8)  # samples at 100 Hz


def _stats(x):
    return [x.mean(), np.ptp(x), np.sqrt(np.mean(x ** 2)), x.var()]


def _fft(x, fs, fmax=5.0, nfft=256):
    mag = np.abs(np.fft.rfft(x - x.mean(), nfft))[1:]
    mag = mag[np.fft.rfftfreq(nfft, 1 / fs)[1:] <= fmax]
    return [skew(mag), kurtosis(mag), mag.mean(), np.median(mag), mag.var(), np.ptp(mag)]


def _haar(x, level=3):
    a = np.asarray(x, float)
    for _ in range(level):
        if len(a) % 2:
            a = np.append(a, a[-1])
        a, d = (a[0::2] + a[1::2]) / np.sqrt(2), (a[0::2] - a[1::2]) / np.sqrt(2)
    return a, d


def _dwt(x):
    approx, detail = _haar(x, 3)
    return _stats(approx) + _stats(detail)


def _ar(x, order=9):
    x = x - x.mean()
    n = len(x)
    if n <= 2 * order or not np.any(x):
        return [0.0] * order
    r = np.array([x[:n - k] @ x[k:] for k in range(order + 1)]) / n
    r[0] *= 1.0 + 1e-3  # small ridge: a 5 Hz low-passed signal at 100 Hz is nearly singular
    return list(solve_toeplitz(r[:order], r[1:]))


def _wvd(x, fs, lmax=32):
    """13 features of the pseudo Wigner-Ville distribution (Hann lag window)."""
    z = hilbert(x - x.mean())
    n = len(z)
    idx = np.arange(n)
    lags = np.arange(-lmax, lmax + 1)
    window = np.hanning(len(lags) + 2)[1:-1]
    kernel = np.zeros((n, len(lags)), complex)
    for j, m in enumerate(lags):
        ok = (idx - abs(m) >= 0) & (idx + abs(m) < n)
        kernel[ok, j] = z[idx[ok] + m] * np.conj(z[idx[ok] - m]) * window[j]
    tfr = np.real(np.fft.fft(np.fft.ifftshift(kernel, axes=1), axis=1))[:, :len(lags) // 2]
    freqs = np.arange(tfr.shape[1]) * fs / (2 * len(lags))
    ti, fi = np.unravel_index(np.argmax(tfr), tfr.shape)
    energy = tfr.sum(axis=1)
    eps = 1e-9 * (np.abs(energy).max() + 1e-12)

    def ratio(e):
        return np.log10((abs(e.max()) + eps) / (abs(e.min()) + eps))

    out = [np.mean(np.abs(np.diff(energy))) * fs, freqs[fi], ti / n, tfr.max(), ratio(energy)]
    for part in np.array_split(energy, 2):
        out += [ratio(part), part.max() - part.min(), part.std(), part.mean()]
    return out


def _xcorr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = len(a)
    return [a[max(0, -l):n - max(0, l)] @ b[max(0, l):n - max(0, -l)] / n for l in XCORR_LAGS]


def _envelope(seg, points=32):
    """Paper B: each channel resampled to 32 points and scaled to [-1, 1]."""
    out = []
    for x in seg:
        y = np.interp(np.linspace(0, len(x) - 1, points), np.arange(len(x)), x)
        span = np.ptp(y)
        out += list(2 * (y - y.min()) / span - 1 if span > 0 else np.zeros(points))
    return out


def dtw(a, b):
    """DTW distance between (n, d) and (m, d) point sequences, divided by n + m."""
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for s in range(2, n + m + 1):  # anti-diagonals only depend on the previous two
        i = np.arange(max(1, s - m), min(n, s - 1) + 1)
        j = s - i
        acc[i, j] = cost[i - 1, j - 1] + np.minimum(np.minimum(acc[i - 1, j], acc[i, j - 1]), acc[i - 1, j - 1])
    return acc[n, m] / (n + m)


def _dtw_seq(seg, step=4):
    """(2, m) at 100 Hz -> (m/4, 2) points at 25 Hz for 2D-DTW."""
    return np.ascontiguousarray(seg[:, ::step].T)


def select_profiles(seqs, labels, classes, per_class=3, pool=25, seed=0):
    """Paper A "2D-DTW profile selection": for each gesture keep the segments with the
    smallest mean DTW distance to the other segments of the same gesture."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    profiles = {}
    for c in classes:
        members = np.flatnonzero(labels == c)
        if len(members) > pool:
            members = rng.choice(members, pool, replace=False)
        k = len(members)
        dist = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                dist[i, j] = dist[j, i] = dtw(seqs[members[i]], seqs[members[j]])
        profiles[c] = [seqs[members[i]] for i in np.argsort(dist.sum(axis=1))[:per_class]]
    return profiles


class SegmentFeaturizer(BaseEstimator, TransformerMixin):
    """(2, m) gesture segments -> feature vectors; `groups` picks the feature families."""

    def __init__(self, groups=PAPER_A, fs=100.0):
        self.groups = groups
        self.fs = fs

    def fit(self, X, y=None):
        self.classes_ = sorted(set(y)) if y is not None else []
        if "dtw" in self.groups:
            self.profiles_ = select_profiles([_dtw_seq(s) for s in X], y, self.classes_)
        self.names_ = self._names()
        return self

    def transform(self, X):
        feats = np.array([self._features(s) for s in X], dtype=float)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def _features(self, seg):
        g, f = self.groups, []
        for x in seg:
            if "stats" in g:
                f += _stats(x)
            if "fft" in g:
                f += _fft(x, self.fs)
            if "dwt" in g:
                f += _dwt(x)
            if "wvd" in g:
                f += _wvd(x, self.fs)
            if "ar" in g:
                f += _ar(x)
        if "xcorr" in g:
            f += _xcorr(seg[0], seg[1])
        if "dtw" in g:
            q = _dtw_seq(seg)
            f += [np.mean([dtw(q, p) for p in self.profiles_[c]]) for c in self.classes_]
        if "env" in g:
            f += _envelope(seg)
        return f

    def _names(self):
        g, names = self.groups, []
        for ch in ("red", "ir"):
            if "stats" in g:
                names += [f"{ch}_{n}" for n in ("mean", "ptp", "rms", "var")]
            if "fft" in g:
                names += [f"{ch}_fft_{n}" for n in ("skew", "kurt", "mean", "median", "var", "ptp")]
            if "dwt" in g:
                names += [f"{ch}_dwt_{p}_{n}" for p in ("a3", "d3") for n in ("mean", "ptp", "rms", "var")]
            if "wvd" in g:
                names += [f"{ch}_wvd_{n}" for n in ("dE", "f_max", "t_max", "max", "ratio")]
                names += [f"{ch}_wvd_w{i}_{n}" for i in (1, 2) for n in ("ratio", "diff", "std", "mean")]
            if "ar" in g:
                names += [f"{ch}_ar{i}" for i in range(1, 10)]
        if "xcorr" in g:
            names += [f"xcorr_{l * 10}ms" for l in XCORR_LAGS]
        if "dtw" in g:
            names += [f"dtw_{c}" for c in self.classes_]
        if "env" in g:
            names += [f"{ch}_env{i}" for ch in ("red", "ir") for i in range(32)]
        return names


# --------------------------------------------------------------- models ----

class LinearSVM(BaseEstimator, ClassifierMixin):
    """Paper B's linear SVM (C = 1) with probabilities. The probability calibration
    uses as many folds as the smallest gesture allows, so small datasets still train."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        folds = min(3, min(Counter(y).values()))
        svm = SVC(kernel="linear", C=self.C)
        self.model_ = CalibratedClassifierCV(svm, cv=folds, ensemble=False) if folds >= 2 else svm
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        if isinstance(self.model_, CalibratedClassifierCV):
            return self.model_.predict_proba(X)
        return (self.model_.predict(X)[:, None] == self.classes_).astype(float)


MODELS = {
    "gbt": "論文 A：每個手勢一個二元 Gradient Boosting（特徵表 I）",
    "rf": "Random Forest（同一組特徵，對照用）",
    "svm": "論文 B：32 點波形包絡 + 線性 SVM",
    "knn": "論文 B：32 點波形包絡 + 1-NN",
}


def make_model(kind, n_estimators=300, seed=0):
    if kind == "gbt":
        # paper A: exponential loss, shrinkage 0.1, subsampling 0.5 (the paper runs 2000 rounds)
        gbt = GradientBoostingClassifier(loss="exponential", learning_rate=0.1, n_estimators=n_estimators,
                                         subsample=0.5, random_state=seed)
        return Pipeline([("feat", SegmentFeaturizer(PAPER_A)), ("clf", OneVsRestClassifier(gbt, n_jobs=-1))])
    if kind == "rf":
        rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
        return Pipeline([("feat", SegmentFeaturizer(PAPER_A)), ("clf", rf)])
    if kind == "svm":
        return Pipeline([("feat", SegmentFeaturizer(PAPER_B)), ("clf", LinearSVM())])
    if kind == "knn":
        return Pipeline([("feat", SegmentFeaturizer(PAPER_B)), ("clf", KNeighborsClassifier(n_neighbors=1))])
    raise ValueError(f"unknown model {kind!r}, choose from {', '.join(MODELS)}")


def feature_importance(model):
    """(name, importance) pairs for the tree models, largest first."""
    clf = model.named_steps["clf"]
    if isinstance(clf, OneVsRestClassifier):
        imp = np.mean([e.feature_importances_ for e in clf.estimators_], axis=0)
    elif hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    else:
        return []
    names = model.named_steps["feat"].names_
    return sorted(zip(names, imp), key=lambda p: -p[1])


# ----------------------------------------------------------------- live ----

class LivePredictor:
    """Streaming version of the training pipeline:
    rest calibration -> trigger -> wait for the window -> extract -> classify."""

    def __init__(self, bundle, calib_seconds=8.0, calib_range=None):
        self.cfg = SegConfig(**bundle["cfg"])
        self.model = bundle["model"]
        self.classes = list(self.model.classes_)
        self.buf = GrowBuffer(2)
        self.score = GrowBuffer(1)  # normalised energy per sample, 0 while calibrating
        self.calib_seconds = calib_seconds
        self.recalibrate()
        if calib_range:  # replaying a collect.py session: use its rest period, exactly like training
            self.cal_start, end = calib_range
            self.calib_n = end - self.cal_start

    def recalibrate(self):
        self.cal = self.det = self.trig = None
        self.cal_start = len(self.buf)
        self.calib_n = int(self.calib_seconds * self.cfg.fs)
        self.done = len(self.buf)
        self.pending = []

    @property
    def calib_progress(self):
        return min(1.0, (len(self.buf) - self.cal_start) / self.calib_n)

    def feed(self, red, ir):
        """Add raw samples; returns a result dict for every gesture completed."""
        self.buf.extend(red, ir)
        n = len(self.buf)
        red_all, ir_all = self.buf.data
        if self.cal is None:
            end = self.cal_start + self.calib_n
            if n < end:
                self._score_upto(n)
                return []
            self.cal = calibrate(red_all[self.cal_start:end], ir_all[self.cal_start:end], self.cfg)
            self.det, self.trig = Detector(self.cfg, self.cal), Trigger(self.cfg)
            self.done = end  # detection starts right after calibration, as in training
            self._score_upto(end)
        e = self.det.process(red_all[self.done:n], ir_all[self.done:n]).max(axis=0)
        self.score.extend(e)
        self.pending += self.trig.feed(e, self.done)
        self.done = n

        results = []
        wait = int(round(self.cfg.wait * self.cfg.fs))
        while self.pending and self.pending[0] + wait <= n:
            onset = self.pending.pop(0)
            r = extract(red_all, ir_all, onset, self.cal, self.cfg)
            if r is None or r["seg"].shape[1] < 8:
                continue
            proba = self.model.predict_proba(as_objects([r["seg"]]))[0]
            best = int(np.argmax(proba))
            r.update(onset=onset, proba=proba, label=self.classes[best], conf=float(proba[best]))
            results.append(r)
        return results

    def _score_upto(self, n):
        if n > len(self.score):
            self.score.extend(np.zeros(n - len(self.score)))


def bundle_for(model, labels, cfg: SegConfig, **extra):
    return {"model": model, "labels": list(labels), "cfg": asdict(cfg), "version": 1, **extra}
