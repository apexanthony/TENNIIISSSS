# Version Directory

The project now uses a continuous V4-series naming scheme for bounce-event work:

```text
v1_original/
  Original TrackNet baseline.

v2_heatmap/
  Single-channel heatmap output version.

v3_lightweight/
  Lightweight heatmap tracker for deployment and video trajectory inference.
  v3_0_base/: baseline lightweight heatmap tracker.
  v3_1_stable_postprocess/: stable inference post-processing.
  v3_2_hard_negative/: hard-negative fine-tuning.
  v3_3_aux_positive/: auxiliary-positive fine-tuning.
  v3_4_multitask_context/: ball/player/court/status multi-task context training.
  v3_6_clean_reference/: reference-style clean ball-only heatmap training.
  v3_7_tracknet_ball_hardneg/: scratch-trained TrackNet ball-only model with clip-level splits and player/shoe/court/ad hard negatives.
  V3.1 stable inference script: versions/v3_lightweight/v3_1_stable_postprocess/infer_on_video_v31_stable.py
  Notes: docs/V3_1_TRACK_STABILIZATION.md
  Next reference-style pipeline: docs/NEXT_STEP_REFERENCE_PIPELINE.md

v4/
  v4_1_bounce_rule/
    V4.1: rule-based bounce detection from V3 trajectory CSVs.

  v4_2_bounce_classifier/
    V4.2: lightweight trajectory-feature classifier built on V4.1 candidates.

  v4_3_event_classifier/
    V4.3: event-level classifier with learned frame-offset correction.

v5_reference_pipeline/
  V5: integrated reference-style pipeline.
  It reads video, runs V3 TrackNet tracking, optionally uses MediaPipe/player context and court regions, then detects hit/bounce events with geometry.
  Main script: versions/v5_reference_pipeline/infer_video_v5_pipeline.py

v5_1_reference_pipeline/
  V5.1: current reference-assisted product pipeline.
  Event windows use seconds and are converted from the input FPS at runtime.
  Serve tosses are suppressed before bounce scoring when a new point starts.
  Bounce scoring fuses image-space and mini-court trajectory evidence.
  Homography/player/track reliability controls dynamic event weights.
  Event CSV includes a refined frame interval and timing confidence.
  It copies the V5 baseline and continues development on automatic MediaPipe player crops, Canny/Hough court homography, mini-court mapping, and reference-style event detection.
  Main script: versions/v5_1_reference_pipeline/infer_video_v51_pipeline.py
  CSV adapter: versions/v5_1_reference_pipeline/apply_reference_assist.py

v5_2_integrated_pipeline/
  V5.2: self-contained stable inference package.
  It consolidates V3.7 ball tracking, MediaPipe player context, automatic court
  mapping, V5.1 bounce recognition, rendering, and evaluation under one folder.
  Recommended entry: versions/v5_2_integrated_pipeline/run_pipeline.py
  Architecture: versions/v5_2_integrated_pipeline/README.md
```

## Naming Note

Older notes used `V5` for the trajectory candidate classifier and `V6` for the event-level offset classifier. They are now integrated into the V4 line:

```text
old V4 -> V4.1
old V5 -> V4.2
old V6 -> V4.3
```

Inside `v4_1_bounce_rule`, flags such as `--enable-v41`, `--enable-v42`, and `--enable-v43` are retained as legacy rule-preset switches for reproducibility.

## Main Commands

V4.1 rule baseline:

```powershell
python versions\v4\v4_1_bounce_rule\eval_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --region-root .\configs\court_regions ^
  --region-bonus 0 ^
  --region-penalty 4 ^
  --min-score 9.5 ^
  --enable-v41 ^
  --enable-v42
```

V4.2 trajectory candidate classifier:

```powershell
python versions\v4\v4_2_bounce_classifier\train_eval_bounce_classifier.py ^
  --labels-root .\datasets\trackNet\images ^
  --region-root .\configs\court_regions ^
  --out-dir .\exps\v4_2_bounce_classifier ^
  --candidate-min-score 3.0 ^
  --positive-tolerance 3 ^
  --negative-gap 6
```

V4.3 event classifier with offset correction:

```powershell
python versions\v4\v4_3_event_classifier\train_eval_event_classifier.py ^
  --out-dir .\exps\v4_3_event_classifier_cand2_final ^
  --candidate-min-score 2.0 ^
  --offset-min -3 ^
  --offset-max 10 ^
  --negative-gap 12 ^
  --match-tolerance 3 ^
  --event-trees 500 ^
  --offset-trees 400
```

V4.3 example-video inference:

```powershell
python versions\v4\v4_3_event_classifier\infer_event_classifier.py ^
  --track-csv .\exps\demo_best_v5\demo_v3_track_thr030.csv ^
  --model-path .\exps\v4_3_event_classifier_cand2_final\model_event_offset.pkl ^
  --out-csv .\exps\demo_best_v5\demo_bounces_v43.csv ^
  --video-path .\示例视频1.mp4 ^
  --video-out-path .\exps\demo_best_v5\demo_bounces_v43.mp4 ^
  --court-region-json .\configs\court_regions\game1.json ^
  --jump-hard-max 240 ^
  --codec mp4v
```

V5.1 reference-style integrated inference:

```powershell
python versions\v5_1_reference_pipeline\infer_video_v51_pipeline.py ^
  --model-path .\exps\lite_heatmap_v3_270x480_from240\model_best_thr070_pw15.pt ^
  --video-path .\示例视频1.mp4 ^
  --video-out-path .\exps\v5_1_reference_pipeline\demo_v51.mp4 ^
  --track-csv-out .\exps\v5_1_reference_pipeline\demo_v51_track.csv ^
  --event-csv-out .\exps\v5_1_reference_pipeline\demo_v51_events.csv ^
  --input-height 270 ^
  --input-width 480 ^
  --threshold 0.30 ^
  --top-k 5 ^
  --fps-adaptive ^
  --codec mp4v ^
  --device cuda
```

V5.1 product-style automatic mode:

```powershell
python versions\v5_1_reference_pipeline\infer_video_v51_pipeline.py ^
  --auto-product-mode ^
  --model-path .\exps\lite_heatmap_v3_270x480_from240\model_best_thr070_pw15.pt ^
  --video-path .\input.mp4 ^
  --video-out-path .\exps\v5_1_reference_pipeline\auto_v51.mp4 ^
  --track-csv-out .\exps\v5_1_reference_pipeline\auto_v51_track.csv ^
  --event-csv-out .\exps\v5_1_reference_pipeline\auto_v51_events.csv ^
  --input-height 270 ^
  --input-width 480 ^
  --threshold 0.30 ^
  --top-k 5 ^
  --codec mp4v ^
  --device cuda
```

This automatic mode uses V3 TrackNet for the ball, MediaPipe Pose for players, and Canny + HoughLines for court-line homography.

Optional V5.1 player/court context:

```powershell
  --annotations-csv .\datasets\tennis_all_v4i_clip_split\annotations_clip_split.csv ^
  --game game1 ^
  --clip Clip1 ^
  --court-region-json .\configs\court_regions\game1.json ^
  --draw-players ^
  --draw-court
```

With `--annotations-csv`, V5.1 also reads `court_1...court_14` and writes mini-court mapped `court_x,court_y` into the track CSV.

Optional automatic court-line fallback:

```powershell
  --auto-court-lines ^
  --court-detect-stride 15
```
