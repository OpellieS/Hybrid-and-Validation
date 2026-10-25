# Reproducibility Instructions

## Environment

Use Python 3.12 on Windows, macOS, or Linux. The recorded execution used Python 3.12.15 on Windows 11. Create an isolated environment and install `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

On POSIX systems, activate `.venv/bin/activate` and use `export PYTHONPATH=src`.

## Verification and pipeline

Run the tests, then execute phases in order. Phase 04 must succeed before Phase 05.

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\run_phase01.py
.venv\Scripts\python scripts\run_phase02.py
.venv\Scripts\python scripts\run_phase03.py
.venv\Scripts\python scripts\run_phase04.py
.venv\Scripts\python scripts\run_phase05.py
.venv\Scripts\python scripts\run_phase06.py
```

Execute the notebook from a fresh kernel:

```powershell
.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute notebooks\Lab11C_Hybrid_and_Validation.ipynb --output Lab11C_Hybrid_and_Validation.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.cwd="."
```

The frozen configuration hash must validate before final-test evaluation. Resource timings will vary by hardware. The bootstrap is deterministic under seed 202406; model initialization is deterministic under seed 202411.

## Known limits

The Week 12 slide deck, `Recommendation System.pptx`, instructor companion script, reference notebook, and original reference outputs were unavailable. The implementation therefore declares its TF-IDF, CF, grid, and tie conventions. Do not claim exact instructor reproduction. Replace `STUDENTID_REQUIRED` in the staging ZIP filename with the actual student ID before submission.
