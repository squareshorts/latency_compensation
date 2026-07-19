$ErrorActionPreference = 'Stop'
$python = 'C:\work\auto\.venv-not-robotics\Scripts\python.exe'

& $python scripts\run_yolo_av2_extension.py --detector yolo11n
& $python scripts\run_yolo_av2_extension.py --detector yolo11s
& $python scripts\run_rtdetr_av2.py --role extension
& $python scripts\run_av2_extension_propagation.py --detector yolo11n
& $python scripts\run_av2_extension_propagation.py --detector yolo11s
& $python scripts\run_av2_extension_propagation.py --detector rtdetr_l
& $python scripts\evaluate_causal_gate_extension.py
