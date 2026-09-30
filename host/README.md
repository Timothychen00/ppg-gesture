# host

Python tools for the ESP32 + MAX30102 PPG project: a live heart-rate dashboard (`monitor.py`) and gesture recognition (`collect.py`, `train.py`, `predict.py`).

Setup, usage and options are described in the [main README](../README.md) ([繁體中文](../README.zh-TW.md)). To try everything without hardware:

```bash
uv run monitor.py --sim
uv run collect.py --sim --reps 10
uv run train.py data/*.npz
uv run predict.py --sim
```
