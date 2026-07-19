$ErrorActionPreference = 'Stop'

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_rtdetr_av2.py --role heldout*'
}
if ($active) {
    Wait-Process -Id $active.ProcessId -ErrorAction SilentlyContinue
}

$powershell = (Get-Command powershell.exe).Source
$cpu = Start-Process -FilePath $powershell -ArgumentList @('-ExecutionPolicy','Bypass','-File','scripts\run_third_cpu_pipeline.ps1') -WorkingDirectory 'C:\work\auto' -RedirectStandardOutput 'results\sivp_strengthening\third_cpu_pipeline.log' -RedirectStandardError 'results\sivp_strengthening\third_cpu_pipeline.err.log' -WindowStyle Hidden -PassThru
$gpu = Start-Process -FilePath $powershell -ArgumentList @('-ExecutionPolicy','Bypass','-File','scripts\run_extension_gpu_pipeline.ps1') -WorkingDirectory 'C:\work\auto' -RedirectStandardOutput 'results\sivp_strengthening\extension_gpu_pipeline.log' -RedirectStandardError 'results\sivp_strengthening\extension_gpu_pipeline.err.log' -WindowStyle Hidden -PassThru
@($cpu, $gpu) | Wait-Process
