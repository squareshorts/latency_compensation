import os
import subprocess
import psutil
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\work\auto")
DOCS = ROOT / "docs" / "analysis_closure"
RESULTS = ROOT / "results" / "analysis_closure"

def run_git(cmd):
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()

def main():
    branch = run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = run_git(["git", "rev-parse", "HEAD"])
    status = run_git(["git", "status", "--porcelain"])

    staged = []
    unstaged = []
    untracked = []
    for line in status.split('\n'):
        if not line: continue
        state = line[:2]
        file = line[3:]
        if state == '??': untracked.append(file)
        elif state[0] != ' ' and state[0] != '?': staged.append(file)
        if state[1] != ' ' and state[1] != '?': unstaged.append(file)

    processes = []
    for p in psutil.process_iter(['pid', 'name']):
        if 'python' in p.info['name'].lower():
            processes.append(p.info)
    pd.DataFrame(processes).to_csv(RESULTS / "process_inventory.csv", index=False)

    import shutil
    total, used, free = shutil.disk_usage(str(ROOT))

    input_files = []
    sivp_dir = ROOT / "results" / "sivp_strengthening"
    for p in sivp_dir.rglob("*"):
        if p.is_file():
            input_files.append({"path": str(p.relative_to(ROOT)), "size_bytes": p.stat().st_size})

    pd.DataFrame(input_files).to_csv(RESULTS / "input_file_inventory.csv", index=False)

    error_logs = list(sivp_dir.glob("*.err.log"))
    temp_files = list(sivp_dir.glob("*.tmp"))

    audit_md = f"""# Recovery and Safety Audit

## Git State
- **Branch:** {branch}
- **Commit:** {commit}

### Staged Changes
{chr(10).join(f'- {f}' for f in staged) if staged else '- None'}

### Unstaged Changes
{chr(10).join(f'- {f}' for f in unstaged) if unstaged else '- None'}

### Untracked Files
{chr(10).join(f'- {f}' for f in untracked) if untracked else '- None'}

## System State
- **Free Disk Space:** {free / (1024**3):.2f} GB
- **Active Python Processes:** {len(processes)} (details in `results/analysis_closure/process_inventory.csv`)

## File State
- **Result Files Count:** {len(input_files)} (details in `results/analysis_closure/input_file_inventory.csv`)
- **Error Logs:** {len(error_logs)} files
- **Temporary Files:** {len(temp_files)} files
"""
    (DOCS / "recovery_audit.md").write_text(audit_md, encoding="utf-8")

if __name__ == "__main__":
    main()
