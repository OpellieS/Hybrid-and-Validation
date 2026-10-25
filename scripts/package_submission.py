from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
ARCHIVE = ROOT / "submission" / "961731_STUDENTID_REQUIRED_Lab11C.zip"
MANIFEST = ROOT / "artifacts" / "evidence" / "submission_contents_manifest.csv"
INTEGRITY = ROOT / "artifacts" / "evidence" / "archive_integrity.json"

EXCLUDED_PARTS = {".git", ".venv", "tmp", "submission", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDED_NAMES = {"archive_integrity.json"}
SENSITIVE_SUFFIXES = {".pem", ".key"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def included_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name in EXCLUDED_NAMES or path.suffix.lower() in SENSITIVE_SUFFIXES:
            continue
        if path.name.startswith(".env") or "credential" in path.name.lower() or "token" in path.name.lower():
            continue
        files.append(path)
    return sorted(files, key=lambda value: value.relative_to(ROOT).as_posix())


def main() -> None:
    required = [
        ROOT / "notebooks" / "Lab11C_Hybrid_and_Validation.executed.ipynb",
        ROOT / "output" / "pdf" / "Lab11C_Methodological_Defense.pdf",
        ROOT / "data" / "raw" / "interaction_events-2.csv",
        ROOT / "references" / "Lab_11C_Hybrid_and_Validation.docx",
        ROOT / "artifacts" / "frozen_experiment_manifest.json",
        ROOT / "docs" / "FINAL_QA_REPORT.md",
        ROOT / "REPRODUCIBILITY.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required submission files missing: {missing}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    candidates = [path for path in included_files() if path != MANIFEST]
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Path", "Bytes", "SHA256"])
        writer.writeheader()
        for path in candidates:
            writer.writerow(
                {
                    "Path": path.relative_to(ROOT).as_posix(),
                    "Bytes": path.stat().st_size,
                    "SHA256": digest(path),
                }
            )

    files = included_files()
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad_member = archive.testzip()
        names = archive.namelist()
    if bad_member is not None:
        raise zipfile.BadZipFile(f"CRC failure: {bad_member}")
    required_names = {path.relative_to(ROOT).as_posix() for path in required}
    absent = sorted(required_names - set(names))
    if absent:
        raise AssertionError(f"required entries absent from archive: {absent}")

    archive_hash = digest(ARCHIVE)
    sidecar = ARCHIVE.with_suffix(ARCHIVE.suffix + ".sha256")
    sidecar.write_text(f"{archive_hash}  {ARCHIVE.name}\n", encoding="ascii")
    report = {
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "sha256": archive_hash,
        "bytes": ARCHIVE.stat().st_size,
        "entries": len(names),
        "crc_test_pass": True,
        "required_entries_pass": True,
        "excluded_roots": sorted(EXCLUDED_PARTS),
        "student_id_status": "missing; staging filename used",
        "missing_instructor_resources_not_fabricated": [
            "Week12 - Recommendation Systems.pdf",
            "Recommendation System.pptx",
            "instructor companion script",
            "reference notebook",
            "original reference outputs",
        ],
    }
    INTEGRITY.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
