# V4.3 Event Classifier

V4.3 is the current recommended bounce-event module. It replaces hand-tuned single-video repair rules with a data-driven event-level classifier and a learned frame-offset classifier:

```text
trajectory candidates -> event classifier -> offset classifier -> NMS
```

The model still uses only trajectory data. It does not read images and does not introduce YOLO or another visual detector.

## Relation To Previous Names

This module was previously called `v6_event_classifier`. It is now integrated as:

```text
old V6 -> V4.3
```

## Recommended Model

```text
exps/v4_3_event_classifier_cand2_final/model_event_offset.pkl
```

## Train And Evaluate

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

## Current Results

Raw validation on `game8-game10`:

```text
Precision = 0.9923
Recall    = 0.8431
F1        = 0.9117
AvgErr    = 0.465 frames
```

Observable validation, excluding clip-boundary events with margin 12:

```text
Precision = 0.9919
Recall    = 0.9389
F1        = 0.9647
AvgErr    = 0.447 frames
```

Observable all-data result:

```text
Precision = 0.9769
Recall    = 0.9358
F1        = 0.9559
AvgErr    = 0.277 frames
```

Boundary events are reported separately because the model uses a local temporal feature window. Events at the first or last few frames of a clip may be impossible to infer reliably from a clipped trajectory.

## Summarize Observable Metrics

```powershell
python versions\v4\v4_3_event_classifier\summarize_eval.py ^
  --detail-csv .\exps\v4_3_event_classifier_cand2_final\detail_val.csv ^
  --boundary-margin 12 ^
  --out-prefix .\exps\v4_3_event_classifier_cand2_final\observable_val_m12
```

## Single-Video Inference

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

## Optimization Route

V4.3 optimization should follow the full-dataset route:

```text
1. Evaluate on full TrackNet labels.
2. Report raw metrics and observable metrics.
3. Analyze FP/FN by game and clip.
4. Tune candidate generation only when validation metrics improve.
5. Keep single-video repair rules out of the default path.
```

The best generalizable change found so far is lowering `candidate-min-score` from `3.0` to `2.0`, then letting the event classifier reject false candidates.
