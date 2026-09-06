# Adaptive Cloud RL

The Python project is in [Adaptive_Cloud_RL](Adaptive_Cloud_RL/README.md). The original Component 1 PDF and presentation are preserved in this directory.

Run from PowerShell:

```powershell
cd "D:\Projects\Adaptive Cloud\Adaptive_Cloud_RL"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_all.py --quick
```

Results and graphs are written to `Adaptive_Cloud_RL/results/quick/`. Repeat runs automatically use a timestamped directory, or you can choose one with `--output results/quick_repeat`.
