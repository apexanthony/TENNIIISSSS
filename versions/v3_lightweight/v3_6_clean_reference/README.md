# V3.6 Clean Reference Training

V3.6 returns to the reference TrackNet principle:

```text
3 RGB frames -> 9-channel input -> single-channel ball heatmap
```

The model only learns ball localization. Player, court, hit, bounce, and status reasoning are handled by V5 post-processing.

## Main Idea

```text
Keep:
  lightweight V3 depthwise model
  Gaussian ball heatmap target
  clip-level train/valid/test split
  optional small hard-negative penalty from player boxes

Remove:
  status classification
  player mask head
  court keypoint head
  hit/bounce labels from model training
```

## Smoke Test

```powershell
python versions\v3_lightweight\v3_6_clean_reference\main_v36_clean_reference.py ^
  --exp-id v36_smoke ^
  --input-height 270 ^
  --input-width 480 ^
  --batch-size 1 ^
  --val-batch-size 8 ^
  --num-epochs 1 ^
  --steps-per-epoch 2 ^
  --max-val-batches 4 ^
  --val-intervals 1 ^
  --device cuda
```

## 360x640 Accuracy Run

```powershell
python versions\v3_lightweight\v3_6_clean_reference\main_v36_clean_reference.py ^
  --exp-id lite_heatmap_v36_clean_360x640 ^
  --input-height 360 ^
  --input-width 640 ^
  --heatmap-radius 8 ^
  --heatmap-sigma 3.0 ^
  --batch-size 2 ^
  --val-batch-size 8 ^
  --num-epochs 120 ^
  --steps-per-epoch 300 ^
  --val-intervals 5 ^
  --augment ^
  --amp ^
  --device cuda
```

## Cleaned Clip Split

The Roboflow context labels contain horizontally mirrored samples. The ball
TrackNet coordinate and Roboflow ball center showed this pattern:

```text
tracknet_x + ball_cx ~= image_width
tracknet_y ~= ball_cy
```

About 22% of rows were affected. Clean the split before using player boxes,
court keypoints, or Roboflow ball boxes as context:

```powershell
python tools\clean_clip_split_mirror.py ^
  --input-dir datasets\tennis_all_v4i_clip_split ^
  --output-dir datasets\tennis_all_v4i_clip_split_cleaned
```

Current cleaned summary:

```text
train: 12540 rows, 2772 mirror-corrected
valid: 2678 rows, 622 mirror-corrected
test : 2691 rows, 604 mirror-corrected
```

After cleaning, TrackNet coordinates and Roboflow ball centers differ by at most
4 px on the full 1280x720 label scale.

## Cleaned Fine-Tune

Use the previous V3.6 best checkpoint as a warm start, but train against the
cleaned context labels:

```powershell
python versions\v3_lightweight\v3_6_clean_reference\main_v36_clean_reference.py ^
  --exp-id lite_heatmap_v36_cleaned_finetune_360x640 ^
  --train-csv datasets\tennis_all_v4i_clip_split_cleaned\train.csv ^
  --valid-csv datasets\tennis_all_v4i_clip_split_cleaned\valid.csv ^
  --pretrained exps\lite_heatmap_v36_clean_360x640\model_best_f1.pt ^
  --input-height 360 ^
  --input-width 640 ^
  --heatmap-radius 8 ^
  --heatmap-sigma 3.0 ^
  --batch-size 2 ^
  --val-batch-size 8 ^
  --num-epochs 80 ^
  --steps-per-epoch 300 ^
  --val-intervals 5 ^
  --lr 3e-4 ^
  --hardneg-weight 0.05 ^
  --augment ^
  --amp ^
  --device cuda
```

## Distance Sensitivity

The strict default validation radius is:

```text
min_dist = 5 px on 1280x720 label scale
```

Evaluate the same checkpoint with multiple hit radii:

```powershell
python versions\v3_lightweight\v3_6_clean_reference\eval_v36_sensitivity.py ^
  --checkpoint exps\lite_heatmap_v36_clean_360x640\model_best_f1.pt ^
  --valid-csv datasets\tennis_all_v4i_clip_split_cleaned\valid.csv ^
  --output-csv exps\lite_heatmap_v36_clean_360x640\sensitivity_cleaned_valid.csv ^
  --input-height 360 ^
  --input-width 640 ^
  --device cuda ^
  --min-dists 5,8,10,15
```

Current sensitivity result:

```text
min_dist=5   F1=0.7744
min_dist=8   F1=0.8938
min_dist=10  F1=0.9182
min_dist=15  F1=0.9452
```

This means the model is often near the ball, but the strict 5 px metric counts
small localization offsets as errors.

## 270x480 Deployment Run

```powershell
python versions\v3_lightweight\v3_6_clean_reference\main_v36_clean_reference.py ^
  --exp-id lite_heatmap_v36_clean_270x480 ^
  --input-height 270 ^
  --input-width 480 ^
  --heatmap-radius 6 ^
  --heatmap-sigma 2.25 ^
  --batch-size 4 ^
  --val-batch-size 8 ^
  --num-epochs 120 ^
  --steps-per-epoch 300 ^
  --val-intervals 5 ^
  --augment ^
  --amp ^
  --device cuda
```

## Validation Metrics

V3.6 reports both detection and trajectory-quality metrics:

```text
precision
recall
f1
valid_ratio
jump80
jump120
jump240
max_jump
```

This is important because V5 event detection depends more on stable trajectories than on frame-level detection alone.
