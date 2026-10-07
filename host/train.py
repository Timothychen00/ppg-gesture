"""訓練手勢模型：讀 collect.py 錄的資料，切出手勢、做交叉驗證、存成 predict.py 用的模型。

    python train.py data/*.npz                  # 比較 4 種模型，存下最好的
    python train.py data/*.npz --model gbt      # 指定論文 A 的分類器
    python train.py data/*.npz --rounds 20 --trees 2000   # 照論文 A 的設定（很慢）
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from ppgkit.gesture import (MODELS, SegConfig, as_objects, bundle_for, display_name, feature_importance, make_model,
                            model_field, segment_session)
from ppgkit.gesture import NONE
from ppgkit import dsp
from ppgkit.sources import load_session


def blocked_cv(kind, X, y, blocks, labels, args):
    """Time-blocked cross-validation: each fold tests on one stretch of every recording and trains on the
    rest, so neighbouring trials (same strap pressure, same drift) never sit on both sides."""
    accs, cm = [], np.zeros((len(labels), len(labels)))
    for b in sorted(set(blocks)):
        train, test = blocks != b, blocks == b
        if len(set(y[train])) < 2 or not test.any():
            continue
        model = make_model(kind, n_estimators=args.trees).fit(X[train], y[train])
        pred = model.predict(X[test])
        accs.append(np.mean(pred == y[test]))
        cm += confusion_matrix(y[test], pred, labels=labels)
    return np.array(accs), cm


def time_blocks(groups, n_blocks):
    """Block index of every segment: each session's segments (in recording order) split into n_blocks."""
    blocks = np.zeros(len(groups), dtype=int)
    for g in set(groups):
        idx = np.flatnonzero(groups == g)
        for b, part in enumerate(np.array_split(idx, n_blocks)):
            blocks[part] = b
    return blocks


def monte_carlo(kind, X, y, labels, args):
    """Paper A's evaluation: repeated random splits, `train_frac` of the data for training."""
    split = StratifiedShuffleSplit(n_splits=args.rounds, train_size=args.train_frac, random_state=0)
    accs, cm = [], np.zeros((len(labels), len(labels)))
    for train, test in split.split(np.zeros(len(y)), y):
        model = make_model(kind, n_estimators=args.trees).fit(X[train], y[train])
        pred = model.predict(X[test])
        accs.append(np.mean(pred == y[test]))
        cm += confusion_matrix(y[test], pred, labels=labels)
    return np.array(accs), cm


def print_confusion(cm, labels):
    rows = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    width = max(len(l) for l in labels) + 2
    print(" " * width + "".join(f"{l[:7]:>8}" for l in labels) + "   ← 預測")
    for label, row in zip(labels, rows):
        print(f"{label:<{width}}" + "".join(f"{v:8.0%}" for v in row))


def save_confusion_png(cm, labels, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = 1.0 + 0.7 * len(labels)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{rows[i, j]:.2f}", ha="center", va="center",
                    color="white" if rows[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+", help="collect.py 錄的 .npz 檔")
    parser.add_argument("--model", default="auto", choices=["auto", *MODELS], help="auto＝交叉驗證最好的")
    parser.add_argument("--compare", default="wlr,wknn,rf,svm,knn,gbt", help="要比較的模型，逗號分隔")
    parser.add_argument("--blocks", type=int, default=4, help="時間分段交叉驗證：每個 session 切成幾段輪流當測試")
    parser.add_argument("--gestures", help="只訓練這幾種手勢（逗號分隔），例如 wrist_down,wrist_up,fist")
    parser.add_argument("--no-none", action="store_true", help="不要把多餘觸發（放開手的動作等）當成「無手勢」類別訓練")
    parser.add_argument("--rounds", type=int, default=10, help="Monte-Carlo 隨機切分次數（論文 A：20；只當樂觀參考）")
    parser.add_argument("--train-frac", type=float, default=0.5, help="每次拿來訓練的比例（論文 A：0.5）")
    parser.add_argument("--trees", type=int, default=300, help="GBT 回合數（論文 A：2000）")
    parser.add_argument("--mode", choices=["threshold", "energy"], default="threshold",
                        help="切段方式：threshold＝能量起訖點；energy＝論文 A 式 (1) 固定 1.2 s 視窗")
    parser.add_argument("--threshold", choices=["robust", "paper"], default="robust",
                        help="偵測門檻：robust＝中位數 + k×MAD（預設）；paper＝論文 A 的 μ + kσ")
    parser.add_argument("--k", type=float, help="門檻倍數（預設 robust 10、paper 3）。漏抓多就調小，多餘觸發多就調大")
    parser.add_argument("--hp", type=float, default=2.0, help="偵測用高通截止頻率 Hz（論文 A：2）")
    parser.add_argument("--out", default="models/gesture_model.joblib")
    args = parser.parse_args()

    kinds = [k.strip() for k in args.compare.split(",") if k.strip()]
    if args.model != "auto" and args.model not in kinds:
        kinds.append(args.model)
    fields = sorted({model_field(k) for k in kinds})

    print("== 切出手勢 ==")
    segs = {f: [] for f in fields}
    labels_all, groups = [], []
    fits = []  # sensor fit during each session's rest calibration
    cfg = None
    for k, path in enumerate(args.sessions):
        session = load_session(path)
        if session["calib"] is not None:
            c0, c1 = session["calib"]
            fits.append(dsp.fit_metrics(session["red"][c0:c1], session["ir"][c0:c1], session["fs"]))
        cfg = SegConfig(fs=session["fs"], mode=args.mode, threshold=args.threshold, hp_cut=args.hp)
        if args.k is not None:
            cfg.k_mad = cfg.k_sigma = args.k
        for f in fields:
            s, l, rep = segment_session(session, cfg, field=f, with_none=not args.no_none)
            segs[f] += s
        labels_all += l
        groups += [k] * len(l)
        n_none = sum(x == NONE for x in l)
        print(f"{os.path.basename(path)}：{rep['segments']}/{rep['cues']} 段　"
              f"沒偵測到 {rep['missed']}　多餘觸發 {rep['extra_triggers']}"
              + (f"（其中 {n_none} 個當「無手勢」訓練）" if n_none else ""))
    y, groups = np.array(labels_all), np.array(groups)
    blocks = time_blocks(groups, args.blocks)
    Xs = {f: as_objects(v) for f, v in segs.items()}
    labels = sorted(set(labels_all) - {NONE})
    counts = Counter(labels_all)
    print("每個手勢：" + "　".join(f"{display_name(l)} {counts[l]}" for l in labels)
          + (f"　{display_name(NONE)} {counts[NONE]}" if counts[NONE] else ""))
    if args.gestures:
        want = [g.strip() for g in args.gestures.split(",") if g.strip()]
        drop = [l for l in labels if l not in want]
        keep = ~np.isin(y, drop)
        Xs = {f: v[keep] for f, v in Xs.items()}
        y, groups, blocks = y[keep], groups[keep], blocks[keep]
        labels = [l for l in labels if l in want]
        counts = Counter(y)
    few = [l for l in labels if counts[l] < 4]
    if few:
        print("⚠ 這些手勢切出來不到 4 段，先不訓練：" + "、".join(display_name(l) for l in few))
        keep = ~np.isin(y, few)
        Xs = {f: v[keep] for f, v in Xs.items()}
        y, groups, blocks = y[keep], groups[keep], blocks[keep]
        labels = [l for l in labels if l not in few]
    if len(labels) < 2:
        raise SystemExit("資料不夠：至少要 2 種手勢、每種 4 段以上。")
    if NONE in y:
        labels = labels + [NONE]

    print(f"\n== 時間分段交叉驗證（每個 session 切 {args.blocks} 段，輪流當測試；不會讓相鄰的動作同時出現在訓練和測試）==")
    results = {}
    for kind in kinds:
        t0 = time.time()
        accs, cm = blocked_cv(kind, Xs[model_field(kind)], y, blocks, labels, args)
        results[kind] = (accs, cm)
        print(f"{kind:4s} {accs.mean():6.1%} ± {accs.std():4.1%}   {MODELS[kind]}（{time.time() - t0:.0f} s）")

    print("\n貼合度（校正時的 IR 位準 / 脈搏幅度）：" + "　".join(f"{f['dc']:.0f}% / {f['pi']:.2f}%" for f in fits))
    kind = args.model if args.model != "auto" else max(results, key=lambda k: results[k][0].mean())
    accs, cm = results[kind]
    X = Xs[model_field(kind)]
    print(f"\n選用 {kind}：{MODELS[kind]}")
    print_confusion(cm, labels)
    chance = max(counts[l] for l in labels) / len(y)
    print(f"（亂猜／全選最多的類別 ≈ {chance:.0%}）")

    mc, _ = monte_carlo(kind, X, y, labels, args)
    print(f"\n參考：論文 A 式隨機切分 {mc.mean():.1%} ± {mc.std():.1%}（同一次錄音的相鄰動作會洩漏，通常偏高，不要拿它當成績）")

    if len(set(groups)) > 1:
        print("\n== 跨 session（拿其他 session 訓練、這個 session 測試；重新戴過感測器最接近真實使用）==")
        for g in sorted(set(groups)):
            model = make_model(kind, n_estimators=args.trees).fit(X[groups != g], y[groups != g])
            acc = np.mean(model.predict(X[groups == g]) == y[groups == g])
            print(f"{os.path.basename(args.sessions[g])}：{acc:.1%}")

    model = make_model(kind, n_estimators=args.trees).fit(X, y)
    top = feature_importance(model)[:10]
    if top:
        print("\n最重要的特徵：" + "、".join(f"{n} {v:.3f}" for n, v in top))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(bundle_for(model, labels, cfg, kind=kind, field=model_field(kind), cv_accuracy=float(accs.mean()),
                           cv_kind="time-blocked", fit=dsp.fit_summary(fits), sessions=[os.path.basename(p) for p in args.sessions]), args.out)
    png = os.path.splitext(args.out)[0] + "_confusion.png"
    save_confusion_png(cm, labels, f"{kind}  blocked-CV accuracy {accs.mean():.1%}", png)
    print(f"\n模型存到 {args.out}，混淆矩陣圖 {png}")
    print(f"下一步：python predict.py --model {args.out}")


if __name__ == "__main__":
    main()
