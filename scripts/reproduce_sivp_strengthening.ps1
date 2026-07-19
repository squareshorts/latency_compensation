$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'C:\work\auto\src'
$python = 'C:\work\auto\.venv-not-robotics\Scripts\python.exe'

& $python -m pytest -q
& $python scripts\run_sivp_tracker_baselines.py
& $python scripts\finalize_sivp_tracker_baselines.py
& $python scripts\run_sivp_end_to_end.py
& $python scripts\finalize_sivp_end_to_end.py
& $python scripts\analyze_sivp_failure_sensitivity.py
& $python scripts\run_third_detector_propagation.py --role heldout
& $python scripts\run_sivp_tracker_baselines.py --detector rtdetr_l
& $python scripts\fit_causal_gate.py
& $python scripts\build_av2_extension_pairs.py
& $python scripts\run_av2_extension_propagation.py --detector yolo11n
& $python scripts\run_av2_extension_propagation.py --detector yolo11s
& $python scripts\run_av2_extension_propagation.py --detector rtdetr_l
& $python scripts\evaluate_causal_gate_extension.py
