# Repository pivot audit

Audit time: 2026-07-17 (America/Fortaleza)

## Pre-pivot repository state

- Working directory: `C:\work\auto`
- Local folder name: `auto`
- Git repository: absent; all Git inspection commands returned “not a git repository”.
- Branch, HEAD, remotes, tracked files, and untracked classification: unavailable because `.git` did not exist.
- Existing remote URL: none.
- Running Python/analysis processes: none at audit time.
- GitHub CLI: 2.88.1; authenticated as `squareshorts` with `repo` scope.
- Remote rename permission: not assessable because no origin exists.
- Python: 3.10.11 in `C:\work\auto\.venv-not-robotics`.
- Platform: Windows 10 build 26200.
- Free space on `C:` at audit: 54,940,999,680 bytes (approximately 51.2 GiB).

## Relevant local data

- Real nuScenes root: `C:\work\auto\data\nuscenes`
- Permanent archives: `C:\work\auto\downloads\nuscenes`
- nuScenes devkit: `C:\work\auto\external\nuscenes-devkit`
- YOLO11n weights: `C:\work\auto\yolo11n.pt`
- Confirmed 500 ms pilot: `C:\work\auto\results\not_dual_loop_latency_500ms`
- Negative reliability-gate audit: `C:\work\auto\results\not_robotics_real_feasibility`
- Annotation-cadence audit: `C:\work\auto\results\not_dual_loop_latency`

## Protection action

Because no Git history existed, the current code and derived results will be recorded as a factual initial baseline commit. Raw data, archives, the external SDK, the virtual environment, and model weights are excluded. The safety tag is created on that baseline before cleanup.

No raw nuScenes file will be altered.
