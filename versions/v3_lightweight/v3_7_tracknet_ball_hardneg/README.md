# V3.7 TrackNet Ball-Only Hard Negative

V3.7 returns to the original TrackNet images and labels, then rebuilds the
dataset split at clip level so one clip cannot cross train, valid, and test.

The model remains ball-only:

```text
input: 3 RGB frames -> 9 channels
output: 1 ball heatmap
```

Player and court annotations are used only to construct training-time hard
negative masks. They do not add inference heads or increase model computation.

## Hard Negatives

```text
player body: low/medium negative weight
lower body and shoes: strongest negative weight
court keypoints and court lines: strong negative weight
bright white line/highlight regions: medium negative weight
upper and side edge structures: weak negative weight for ads/scoreboards/audience
```

The positive ball neighborhood is cleared from every hard-negative mask.

The auxiliary mapping uses strict cleaning: rows with `visibility=0`, missing
TrackNet matches, ambiguous orientation, or ball-center alignment error above
5 pixels are discarded. Duplicate Roboflow augmentations are reduced to the
single candidate with the smallest verified alignment error.

Strong motion-blur, JPEG, and noise augmentations are checked at the labelled
ball neighborhood. An operation is rolled back when local ball contrast drops
below `4.0` or below `55%` of its pre-operation value, preventing synthetic
augmentation from erasing the target.

```powershell
python tools\clean_v37_hardneg_mapping.py
```

Generated mapping:

```text
datasets/tennis_all_v4i_mapped/annotations_hardneg_cleaned_strict.csv
```

## Dataset Split

```powershell
python tools\prepare_tracknet_v37_split.py ^
  --out-dir datasets\tracknet_v37_clip_split ^
  --seed 37
```

Generated split:

```text
train: 13775 frames, 65 clips
valid: 2935 frames, 15 clips
test:  2935 frames, 15 clips
```

## Scratch Training

The current main experiment is trained from random initialization:

```powershell
python versions\v3_lightweight\v3_7_tracknet_ball_hardneg\main_v37_hardneg_ball.py ^
  --exp-id lite_heatmap_v37_tracknet_hardneg_ballonly_scratch_360x640 ^
  --train-csv datasets\tracknet_v37_clip_split\train.csv ^
  --valid-csv datasets\tracknet_v37_clip_split\valid.csv ^
  --num-epochs 100 ^
  --steps-per-epoch 0 ^
  --val-intervals 1 ^
  --batch-size 8 ^
  --val-batch-size 8 ^
  --input-height 360 ^
  --input-width 640 ^
  --base-channels 24 ^
  --heatmap-radius 5 ^
  --heatmap-sigma 2.0 ^
  --peak-window 9 ^
  --lr 0.001 ^
  --scheduler plateau ^
  --warmup-epochs 5 ^
  --scheduler-patience 8 ^
  --early-stop-patience 40 ^
  --hardneg-weight 0.08 ^
  --augment ^
  --amp ^
  --device cuda ^
  --num-workers 4
```

The same production configuration is available as a PowerShell launcher:

```powershell
powershell -ExecutionPolicy Bypass -File versions\v3_lightweight\v3_7_tracknet_ball_hardneg\train_v37_clean_batch8.ps1
```

`--steps-per-epoch 0` traverses the complete shuffled training DataLoader,
including the final incomplete batch. Validation runs after every epoch. The
run is capped at 100 epochs and stops early only when validation precision,
recall, and F1 all reach at least `0.95`, or when the configured no-improvement
patience is exhausted. The test split remains untouched until final model
selection is complete.

## Dynamic Learning Rate

New runs default to per-epoch validation with linear warmup followed by
`ReduceLROnPlateau` on validation F1:

```text
warmup: 5 epochs, 1e-5 -> configured lr
plateau patience: 8 validations
plateau factor: 0.5
minimum lr: 1e-6
early stopping: 40 validations without F1 improvement
```

The legacy fixed cosine schedule remains available through
`--scheduler cosine`.

## Hyperparameter Search

Run a short, reproducible grid screening before the full 100-epoch training:

```powershell
python tools\search_v37_hyperparams.py ^
  --max-trials 12 ^
  --trial-epochs 30 ^
  --steps-per-epoch 100
```

Results are ranked by validation F1 in:

```text
exps/v37_hyperparam_search/results.csv
```

The best configuration should then be trained from scratch using the full
training budget and evaluated once on the untouched test split.

## Adaptive Video Post-processing

V3.7 accepted detections are already accurately localized, while its main
errors are weak responses and occasional far peaks. The dedicated video
pipeline therefore preserves observed coordinates and uses motion prediction
only for candidate association:

```text
low-threshold top-k candidates
-> two-frame high-confidence track confirmation
-> constant-velocity prediction and speed-adaptive motion gate
-> recover weak candidates near the predicted trajectory
-> require look-ahead confirmation for suspicious acceleration/turns
-> expire the old track after a long missing run
-> interpolate only geometrically plausible gaps of at most three frames
-> remove isolated spikes and fragments shorter than three frames
```

```powershell
python versions\v3_lightweight\v3_7_tracknet_ball_hardneg\infer_video_v37_adaptive.py ^
  --video_path 示例视频1.mp4 ^
  --video_out_path exps\v37_adaptive\sample1.mp4 ^
  --csv_out_path exps\v37_adaptive\sample1.csv ^
  --fps_adaptive
```

Defaults separate detection confidence from temporal tracking confidence:

- `seed_threshold=0.96`: initialize/reinitialize only after two consistent peaks.
- `high_threshold=0.88`: accept strong observations inside the wider motion gate.
- `low_threshold=0.35`: recover weak peaks only near the predicted trajectory.
- Dynamic gate: recent five-frame median speed, normally constrained to `60-220 px`.
- Weak/strong gate factors: `0.85/1.5`; weak recovery stops after four missing frames.
- Track expiry: eight missing frames; suspicious turns use one-frame look-ahead confirmation.
- Isolated static cleanup: remove disconnected fragments of at most three frames
  when their total displacement radius is at most `8 px`.

No Kalman-smoothed point replaces a real observation, preserving V3.7's
measured localization accuracy. Missing runs longer than three frames remain
missing instead of being filled with fabricated coordinates.

On `示例视频1.mp4`, the balanced defaults plus isolated-static cleanup retained
`319/370` frames in the main court-view segment (`86.2%`). The cleanup removed
the confirmed three-frame false track at `F112-F114`, while all other tracked
coordinates remained unchanged. Adjacent jumps above `120 px` were reduced
from four to one and jumps above `180 px` from one to zero. Close-up scene
changes are intentionally allowed to lose the track and reinitialize.
