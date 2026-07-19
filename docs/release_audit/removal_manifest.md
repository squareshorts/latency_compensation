- path: rtdetr-l.pt
  purpose: third-party detector weight
  reason_for_removal: non-redistributable; must not be committed in public release
  recoverable_from_git_history: yes
- path: yolo11s.pt
  purpose: third-party detector weight
  reason_for_removal: non-redistributable; must not be committed in public release
  recoverable_from_git_history: yes
- path: .venv-not-robotics
  purpose: local Python virtual environment
  reason_for_removal: environment and site-packages are not redistributable and are large
  recoverable_from_git_history: yes
