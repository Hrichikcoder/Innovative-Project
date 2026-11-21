# Quick Launch Script for Specialization Retention Experiments
param (
    [string]$Dataset = "cifar10",
    [string]$Method = "all",
    [int]$Rounds = 50,
    [switch]$Test
)

$VenvPython = "$PSScriptRoot\venv\Scripts\python.exe"
$Runner = "$PSScriptRoot\new_codes\run_specialization_experiments.py"

if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
}

if ($Test) {
    Write-Host "[Launch] Running fast test on $Dataset..." -ForegroundColor Cyan
    & $VenvPython $Runner --test --dataset $Dataset
} elseif ($Method -eq "all") {
    Write-Host "[Launch] Running ALL 3 methods (FedAvg, FedCurv, MI-IAA) on $Dataset ($Rounds rounds)..." -ForegroundColor Green
    & $VenvPython $Runner --all --dataset $Dataset --rounds $Rounds
} else {
    Write-Host "[Launch] Running $Method on $Dataset ($Rounds rounds)..." -ForegroundColor Yellow
    & $VenvPython $Runner --method $Method --dataset $Dataset --rounds $Rounds
}
