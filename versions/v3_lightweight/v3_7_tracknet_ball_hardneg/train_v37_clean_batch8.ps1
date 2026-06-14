$ErrorActionPreference = "Stop"

$python = "E:\python\python.exe"
$trainScript = ".\versions\v3_lightweight\v3_7_tracknet_ball_hardneg\main_v37_hardneg_ball.py"
$expId = "lite_heatmap_v37_clean_hardneg_batch8_360x640"
$expDir = ".\exps\$expId"

New-Item -ItemType Directory -Force -Path $expDir | Out-Null

$preflightId = "v37_batch8_preflight"
$preflightDir = ".\exps\$preflightId"
if (Test-Path $preflightDir) {
    Remove-Item -LiteralPath $preflightDir -Recurse -Force
}

Write-Host "Running one-step batch-8 GPU preflight..."
& $python @(
    $trainScript,
    "--exp-id", $preflightId,
    "--mapped-csv", ".\datasets\tennis_all_v4i_mapped\annotations_hardneg_cleaned_strict.csv",
    "--num-epochs", "1",
    "--steps-per-epoch", "1",
    "--val-intervals", "1",
    "--batch-size", "8",
    "--val-batch-size", "8",
    "--input-height", "360",
    "--input-width", "640",
    "--base-channels", "24",
    "--heatmap-radius", "5",
    "--heatmap-sigma", "2.0",
    "--peak-window", "9",
    "--max-val-batches", "1",
    "--device", "cuda",
    "--num-workers", "0",
    "--augment",
    "--amp"
)
if ($LASTEXITCODE -ne 0) {
    throw "Batch-8 GPU preflight failed. Use batch 4 with gradient accumulation instead."
}
Remove-Item -LiteralPath $preflightDir -Recurse -Force
Write-Host "Batch-8 GPU preflight passed."

$arguments = @(
    $trainScript,
    "--exp-id", $expId,
    "--train-csv", ".\datasets\tracknet_v37_clip_split\train.csv",
    "--valid-csv", ".\datasets\tracknet_v37_clip_split\valid.csv",
    "--mapped-csv", ".\datasets\tennis_all_v4i_mapped\annotations_hardneg_cleaned_strict.csv",
    "--num-epochs", "100",
    "--steps-per-epoch", "0",
    "--val-intervals", "1",
    "--batch-size", "8",
    "--val-batch-size", "8",
    "--input-height", "360",
    "--input-width", "640",
    "--label-height", "720",
    "--label-width", "1280",
    "--base-channels", "24",
    "--heatmap-radius", "5",
    "--heatmap-sigma", "2.0",
    "--peak-window", "9",
    "--min-dist", "5",
    "--lr", "0.001",
    "--weight-decay", "0.00001",
    "--scheduler", "plateau",
    "--warmup-epochs", "5",
    "--warmup-start-lr", "0.00001",
    "--scheduler-patience", "8",
    "--scheduler-factor", "0.5",
    "--early-stop-patience", "40",
    "--pos-weight", "120",
    "--mse-weight", "1.0",
    "--hardneg-weight", "0.08",
    "--hardneg-player-weight", "1.0",
    "--hardneg-shoe-weight", "1.4",
    "--hardneg-court-weight", "0.85",
    "--hardneg-bright-weight", "0.35",
    "--hardneg-edge-weight", "0.18",
    "--hardneg-ball-clear-radius", "18",
    "--augment-min-ball-contrast", "4.0",
    "--augment-min-contrast-ratio", "0.55",
    "--thresholds", "0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    "--target-precision", "0.95",
    "--target-recall", "0.95",
    "--target-f1", "0.95",
    "--augment",
    "--amp",
    "--device", "cuda",
    "--num-workers", "2",
    "--print-interval", "100",
    "--val-print-interval", "100"
)

Write-Host "Starting V3.7 clean batch-8 training: $expId"
& $python @arguments 2>&1 | Tee-Object -FilePath "$expDir\train.log"
