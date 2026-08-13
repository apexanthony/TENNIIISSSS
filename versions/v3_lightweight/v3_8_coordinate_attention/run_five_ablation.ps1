param(
  [int]$Seed = 37,
  [int]$Epochs = 100,
  [int]$BatchSize = 8,
  [int]$NumWorkers = 0,
  [string]$Device = "cuda",
  [string]$OutputRoot = "exps\v38_essay_ablation"
)

$ErrorActionPreference = "Stop"
$variants = @("baseline", "ca", "hardneg", "aux", "full")

foreach ($variant in $variants) {
  $runName = "${variant}_seed${Seed}"
  $runDir = Join-Path $OutputRoot $runName
  New-Item -ItemType Directory -Force $runDir | Out-Null
  Write-Host "=== Training $runName ==="
  $resumeArgs = @()
  $statePath = Join-Path $runDir "training_state.pt"
  if (Test-Path $statePath) {
    $resumeArgs = @("--resume", $statePath)
  }
  $ErrorActionPreference = "Continue"
  & python versions\v3_lightweight\v3_8_coordinate_attention\train_v38_ablation.py `
    --variant $variant `
    --run-name $runName `
    --output-root $OutputRoot `
    --seed $Seed `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --val-batch-size $BatchSize `
    --num-workers $NumWorkers `
    --device $Device `
    @resumeArgs `
    2>&1 | Tee-Object -FilePath (Join-Path $runDir "train.log")
  $trainExitCode = $LASTEXITCODE
  $ErrorActionPreference = "Stop"
  if ($trainExitCode -ne 0) {
    throw "Training $runName failed with exit code $trainExitCode"
  }

  Write-Host "=== Evaluating $runName ==="
  $ErrorActionPreference = "Continue"
  python versions\v3_lightweight\v3_8_coordinate_attention\evaluate_v38.py `
    --run-dir $runDir `
    --batch-size $BatchSize `
    --device $Device `
    2>&1 | Tee-Object -FilePath (Join-Path $runDir "evaluate.log")
  $evaluateExitCode = $LASTEXITCODE
  $ErrorActionPreference = "Stop"
  if ($evaluateExitCode -ne 0) {
    throw "Evaluation $runName failed with exit code $evaluateExitCode"
  }
}
