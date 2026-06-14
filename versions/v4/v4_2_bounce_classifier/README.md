# V4.2 Bounce Classifier

V4.2 adds a lightweight trajectory-feature classifier on top of V4.1 rule candidates. It does not read image frames and does not add a visual detector. It only classifies local trajectory geometry around candidate points, so it remains cheap enough for edge deployment.

## Relation To Previous Names

This module was previously called `v5_bounce_classifier`. It is now integrated as:

```text
old V5 -> V4.2
```

## Train And Evaluate

```powershell
python versions\v4\v4_2_bounce_classifier\train_eval_bounce_classifier.py ^
  --labels-root .\datasets\trackNet\images ^
  --region-root .\configs\court_regions ^
  --out-dir .\exps\v4_2_bounce_classifier ^
  --candidate-min-score 3.0 ^
  --positive-tolerance 3 ^
  --negative-gap 6
```

## Outputs

```text
exps/v4_2_bounce_classifier/
  model_random_forest.pkl
  metrics.csv
  train_samples.csv
  val_samples.csv
```

## Current Role

V4.2 is a stable, low-cost improvement over pure rule-based V4.1. It is useful when simple deployment and interpretability matter more than maximum event-level accuracy.

For the current best generalization result, use V4.3 event classification with learned offset correction.

## Single-Video Inference

```powershell
python versions\v4\v4_2_bounce_classifier\infer_bounce_classifier.py ^
  --track-csv .\exps\demo_best_v5\demo_v3_track_thr030.csv ^
  --model-path .\exps\v4_2_bounce_classifier\model_random_forest.pkl ^
  --out-csv .\exps\demo_best_v5\demo_bounces_v42.csv ^
  --video-path .\示例视频1.mp4 ^
  --video-out-path .\exps\demo_best_v5\demo_bounces_v42.mp4 ^
  --court-region-json .\configs\court_regions\game1.json ^
  --codec mp4v
```
