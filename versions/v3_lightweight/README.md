# V3 Lightweight Versions

V3 is the lightweight TrackNet line for tennis ball trajectory tracking. The files are split by version so each folder can be read and run independently.

```text
v3_0_base/
  Depthwise heatmap tracker baseline.
  Main scripts: main_v3.py, infer_on_video_v3_batch.py, infer_on_video_v3_onnx.py, export_onnx_v3.py

v3_1_stable_postprocess/
  Same V3 model, improved inference post-processing.
  Main script: infer_on_video_v31_stable.py

v3_2_hard_negative/
  V3 fine-tuning with hard-negative suppression for shoes, lines, and false peaks.
  Main script: main_v32_hardneg.py

v3_3_aux_positive/
  V3 fine-tuning with auxiliary positive samples.
  Main script: main_v33_auxpos.py

v3_4_multitask_context/
  Multi-task context training with ball heatmap, player mask, court keypoints, and status head.
  Main script: main_v34_multitask.py

v3_6_clean_reference/
  Clean reference-style ball-only heatmap training.
  Main script: main_v36_clean_reference.py
```

## Current Best Checkpoints

```text
V3.0 recommended deployment model:
exps/lite_heatmap_v3_270x480_from240/model_best_thr070_pw15.pt

V3.4 best context model:
exps/lite_heatmap_v34_multitask_clip_fromscratch/model_best_f1.pt

V3.4 context fine-tune best:
exps/lite_heatmap_v34_ball_finetune_context_200e/model_best_f1.pt
```

## Current Training Direction

V3.6 is the recommended next training direction:

```text
3-frame input
single-channel Gaussian ball heatmap
clip-level split
optional small hard-negative penalty
no status/player/court multi-task heads
```

Important V3.6 data fix:

```text
datasets/tennis_all_v4i_clip_split_cleaned/
```

The cleaned split fixes horizontally mirrored Roboflow context labels in about
22% of rows. Use it for any run that uses player boxes, court keypoints, or
Roboflow ball boxes as context.

Current V3.6 sensitivity on cleaned valid:

```text
min_dist=5   F1=0.7744
min_dist=8   F1=0.8938
min_dist=10  F1=0.9182
min_dist=15  F1=0.9452
```

## Recommended Direction

The reference project suggests that the next practical gain should come from a stronger pipeline, not only from training longer:

```text
TrackNet heatmap -> peak/circle candidates -> motion gating -> player/court context -> hit/bounce event logic
```

The short-term target is V3.5/V4.4: keep V3 as the ball tracker, add reference-style trajectory stabilization, player-nearby suppression, and hit/bounce event detection based on geometry.
