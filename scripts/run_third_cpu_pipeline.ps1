$ErrorActionPreference = 'Stop'
$python = 'C:\work\auto\.venv-not-robotics\Scripts\python.exe'

$propagation = @()
0..3 | ForEach-Object {
    $propagation += Start-Process -FilePath $python -ArgumentList @('scripts\run_third_detector_propagation.py','--role','heldout','--shard-index',"$_",'--shard-count','4') -WorkingDirectory 'C:\work\auto' -RedirectStandardOutput "results\sivp_strengthening\third_propagation_shard_$_.log" -RedirectStandardError "results\sivp_strengthening\third_propagation_shard_$_.err.log" -WindowStyle Hidden -PassThru
}
$propagation | Wait-Process
& $python scripts\run_sivp_tracker_baselines.py --detector rtdetr_l
$endToEnd = @()
0..3 | ForEach-Object {
    $endToEnd += Start-Process -FilePath $python -ArgumentList @('scripts\run_sivp_end_to_end.py','--detector','rtdetr_l','--shard-index',"$_",'--shard-count','4') -WorkingDirectory 'C:\work\auto' -RedirectStandardOutput "results\sivp_strengthening\third_end_to_end_shard_$_.log" -RedirectStandardError "results\sivp_strengthening\third_end_to_end_shard_$_.err.log" -WindowStyle Hidden -PassThru
}
$endToEnd | Wait-Process
& $python scripts\finalize_sivp_tracker_baselines.py
& $python scripts\analyze_sivp_failure_sensitivity.py

$jobs = @()
0..3 | ForEach-Object {
    $jobs += Start-Process -FilePath $python -ArgumentList @('scripts\finalize_sivp_end_to_end.py','--shard-index',"$_",'--shard-count','4','--detector','rtdetr_l') -WorkingDirectory 'C:\work\auto' -RedirectStandardOutput "results\sivp_strengthening\third_metric_shard_$_.log" -RedirectStandardError "results\sivp_strengthening\third_metric_shard_$_.err.log" -WindowStyle Hidden -PassThru
}
$jobs | Wait-Process
& $python scripts\consolidate_sivp_end_to_end.py --detector rtdetr_l
& $python scripts\finalize_third_detector_results.py
