# Lab 11C Hybrid Recommendation and Validation

Reproducible implementation of Problems 1.1-3.3 for 961731 Business Data Analytics. The project uses a fixed chronological split, validation-only model selection, an immutable freeze gate, and untouched final-test evaluation.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\run_pipeline.py
.venv\Scripts\jupyter nbconvert --to notebook --execute notebooks\Lab11C_Hybrid_and_Validation.ipynb --output Lab11C_Hybrid_and_Validation.executed.ipynb --output-dir notebooks
```

The original four CSV files are under `data/raw/`. Evidence and machine-readable outputs are written under `artifacts/`. The assignment and available workflow reference are preserved under `references/`.

Important limitations: the Week 12 slide deck, instructor companion script, reference notebook, and original reference outputs were not present in the supplied workspace. Where those would have controlled an implementation choice, the project labels a fixed independent variant rather than claiming exact reproduction.
