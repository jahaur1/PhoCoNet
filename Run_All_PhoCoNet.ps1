param(
    [string]$Python = "python",
    [string]$RunId = "phoconet_20261027",
    [ValidateSet("formal", "smoke")][string]$Profile = "formal",
    [string[]]$Stages = @("main", "noise", "missingness", "shap"),
    [int]$Gpu = 0,
    [switch]$Execute,
    [switch]$PrepareOnly
)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    $arguments = @("-B", "-m", "experiments.run_suite", "--run-id", $RunId, "--profile", $Profile, "--gpu", "$Gpu", "--stages") + $Stages
    if ($Execute) { $arguments += "--execute" }
    if ($PrepareOnly) { $arguments += "--prepare-only" }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "PhoCoNet suite failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
