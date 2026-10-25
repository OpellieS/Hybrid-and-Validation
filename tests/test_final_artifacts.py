from pathlib import Path
import zipfile

import nbformat
from pypdf import PdfReader

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "Lab11C_Hybrid_and_Validation.executed.ipynb"
PDF = ROOT / "output" / "pdf" / "Lab11C_Methodological_Defense.pdf"
ARCHIVE = ROOT / "submission" / "961731_STUDENTID_REQUIRED_Lab11C.zip"


def test_executed_notebook_has_ordered_counts_and_no_errors():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert errors == []
    joined = "\n".join(cell.source for cell in notebook.cells)
    for part in ("Problem 1.1", "Problems 1.2 and 1.3", "Problems 2.1 and 2.2", "Problem 2.3", "Problem 3.1", "Problem 3.2", "Problem 3.3"):
        assert part in joined


def test_methodological_defense_is_exactly_two_pages():
    reader = PdfReader(PDF)
    assert len(reader.pages) == 2
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for phrase in (
        "Modeling choices",
        "Factor objective and regularization",
        "Validation and freeze",
        "Cold start and reference reproduction",
        "Leakage and uncertainty",
        "Deployment limits",
        "Prespecified online test",
    ):
        assert phrase in text


def test_submission_archive_integrity_and_required_entries():
    assert ARCHIVE.is_file()
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
    required = {
        "notebooks/Lab11C_Hybrid_and_Validation.executed.ipynb",
        "output/pdf/Lab11C_Methodological_Defense.pdf",
        "data/raw/interaction_events-2.csv",
        "references/Lab_11C_Hybrid_and_Validation.docx",
        "artifacts/frozen_experiment_manifest.json",
        "docs/FINAL_QA_REPORT.md",
        "REPRODUCIBILITY.md",
    }
    assert required <= names
    assert not any(name.startswith((".git/", ".venv/", "tmp/", "submission/")) for name in names)
