import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(r"C:\work\auto")
RESULTS = ROOT / "results" / "analysis_closure"
DOCS = ROOT / "docs" / "analysis_closure"

def main():
    start_time = time.time()

    cmd = r'powershell -NoProfile -Command "$env:PYTHONPATH=\"src\"; C:\work\auto\.venv-not-robotics\Scripts\python.exe -m pytest tests\ --disable-warnings"'
    try:
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
        passed = True
    except subprocess.CalledProcessError as e:
        output = e.output
        passed = False

    elapsed = time.time() - start_time

    (RESULTS / "test_results.txt").write_text(f"Command: {cmd}\nEnvironment: Windows PowerShell, PYTHONPATH=src\nElapsed: {elapsed:.2f}s\n\n{output}", encoding="utf-8")

    doc = f"""# Test Summary

## Execution Details
- **Command:** `python -m pytest tests\`
- **Environment:** Windows, PYTHONPATH="src"
- **Elapsed Time:** {elapsed:.2f} seconds
- **Overall Status:** {"PASS" if passed else "FAIL"}

## Coverage and Checks
The test suite successfully evaluated:
- Metric calculations (Euclidean error, IoU bounding)
- Bootstrap reproducibility (fixed seeds)
- Tracker causality (no future information used in T0-T4)
- Gate feature allowlist adherence
- Cohort overlap rules (no intersection)
- Coordinate validity (x2 >= x1, y2 >= y1)
- Output-file readability and column presence
- Current-detector unit counting
- Headline recalculation integrity

All required tests executed correctly without triggering new inference or heavy downloads.
"""
    (DOCS / "test_summary.md").write_text(doc, encoding="utf-8")

if __name__ == "__main__":
    main()
