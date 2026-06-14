# V4.1 Bounce Rule

V4.1 is the rule-based bounce detector built on V3 trajectory CSVs. It does not add any new model. It uses trajectory cleaning, short-gap interpolation, velocity-angle change, acceleration change, jump distance, ROI filtering, and hit-like penalties to detect bounce events.

## Relation To Previous Names

This module was previously the project's `v4_bounce_rule` directory. It is now integrated as:

```text
old V4 -> V4.1
```

Inside this directory, legacy switches such as `--enable-v41`, `--enable-v42`, and `--enable-v43` are retained for reproducibility. They should be read as rule presets:

```text
--enable-v41  hit-like guard preset
--enable-v42  event-merge and weak-bounce scoring preset
--enable-v43  adaptive per-clip statistics experiment
```

They are not the same as the top-level V4.1/V4.2/V4.3 module names.

## Evaluate Rule Baseline

```powershell
python versions\v4\v4_1_bounce_rule\eval_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --region-root .\configs\court_regions ^
  --region-bonus 0 ^
  --region-penalty 4 ^
  --min-score 9.5 ^
  --enable-v41 ^
  --enable-v42 ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_candidates.csv
```

## Tune Rules

```powershell
python versions\v4\v4_1_bounce_rule\tune_bounce_rule.py ^
  --split val ^
  --match-tolerance 3 ^
  --region-root .\configs\court_regions ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_tuning.csv
```

## Single-Video Inference

```powershell
python versions\v4\v4_1_bounce_rule\infer_video_bounce_rule.py ^
  --track-csv .\exps\demo_best_v5\demo_v3_track_thr030.csv ^
  --out-csv .\exps\demo_best_v5\demo_bounces_v41.csv ^
  --video-path .\示例视频1.mp4 ^
  --video-out-path .\exps\demo_best_v5\demo_bounces_v41.mp4 ^
  --court-region-json .\configs\court_regions\game1.json ^
  --region-bonus 0 ^
  --region-penalty 4 ^
  --min-score 9.5 ^
  --enable-v41 ^
  --enable-v42 ^
  --codec mp4v
```

## Current Role

V4.1 is useful as an interpretable baseline and for debugging trajectory geometry. The current recommended production route is V4.3, which learns event classification and frame offset from the full labeled dataset.
