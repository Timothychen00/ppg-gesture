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
                            segment_session)
from ppgkit.sources import load_session


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
    parser.add_argument("--compare", default=",".join(MODELS), help="要比較的模型，逗號分隔")
    parser.add_argument("--rounds", type=int, default=10, help="Monte-Carlo 交叉驗證次數（論文 A：20）")
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

    print("== 切出手勢 ==")
    segs, labels_all, groups = [], [], []
    cfg = None
    for k, path in enumerate(args.sessions):
        session = load_session(path)
        cfg = SegConfig(fs=session["fs"], mode=args.mode, threshold=args.threshold, hp_cut=args.hp)
        if args.k is not None:
            cfg.k_mad = cfg.k_sigma = args.k
        s, l, rep = segment_session(session, cfg)
        print(f"{os.path.basename(path)}：{rep['segments']}/{rep['cues']} 段　"
              f"沒偵測到 {rep['missed']}　多餘觸發 {rep['extra_triggers']}")
        segs += s
        labels_all += l
        groups += [k] * len(s)
    X, y, groups = as_objects(segs), np.array(labels_all), np.array(groups)
    labels = sorted(set(labels_all))
    counts = Counter(labels_all)
    print("每個手勢：" + "　".join(f"{display_name(l)} {counts[l]}" for l in labels))
    few = [l for l in labels if counts[l] < 4]
    if few:
        print("⚠ 這些手勢切出來不到 4 段，先不訓練：" + "、".join(display_name(l) for l in few))
        keep = ~np.isin(y, few)
        X, y, groups = X[keep], y[keep], groups[keep]
        labels = [l for l in labels if l not in few]
    if len(labels) < 2:
        raise SystemExit("資料不夠：至少要 2 種手勢、每種 4 段以上。")

    print(f"\n== Monte-Carlo 交叉驗證（{args.rounds} 次，每次 {args.train_frac:.0%} 訓練）==")
    results = {}
    for kind in [k.strip() for k in args.compare.split(",") if k.strip()]:
        t0 = time.time()
        accs, cm = monte_carlo(kind, X, y, labels, args)
        results[kind] = (accs, cm)
        print(f"{kind:4s} {accs.mean():6.1%} ± {accs.std():4.1%}   {MODELS[kind]}（{time.time() - t0:.0f} s）")

    kind = args.model if args.model != "auto" else max(results, key=lambda k: results[k][0].mean())
    if kind not in results:
        results[kind] = monte_carlo(kind, X, y, labels, args)
    accs, cm = results[kind]
    print(f"\n選用 {kind}：{MODELS[kind]}")
    print_confusion(cm, labels)

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
    joblib.dump(bundle_for(model, labels, cfg, kind=kind, cv_accuracy=float(accs.mean()),
                           sessions=[os.path.basename(p) for p in args.sessions]), args.out)
    png = os.path.splitext(args.out)[0] + "_confusion.png"
    save_confusion_png(cm, labels, f"{kind}  accuracy {accs.mean():.1%}", png)
    print(f"\n模型存到 {args.out}，混淆矩陣圖 {png}")
    print(f"下一步：python predict.py --model {args.out}")


if __name__ == "__main__":
    main()
