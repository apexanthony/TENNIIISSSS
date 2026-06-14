# V5.2 Integrated Pipeline

V5.2 packages the current stable V3.7 tracker and V5.1 bounce recognizer into
one self-contained version directory. Historical version folders remain
unchanged; new inference should use `run_pipeline.py`.

## Architecture

```text
input video
  -> tracking/model.py
     three RGB frames -> 9-channel input -> single-channel ball heatmap
  -> tracking/heatmap_candidates.py
     batch inference -> top-k heatmap peaks -> sub-pixel peak refinement
  -> tracking/track_video.py
     speed-adaptive motion gate -> turn confirmation -> short-gap interpolation
     -> temporal spike/static-fragment cleanup
  -> tracking/trajectory.csv
  -> events/cleanup.py
     bidirectional trajectory outlier cleanup
  -> context/player.py
     MediaPipe Pose -> player boxes, hands, feet and temporal stabilization
  -> context/court.py
     Canny/Hough court lines -> stabilized homography -> mini-court coordinates
  -> events/detector.py
     image/court speed, acceleration, turn, player distance, trajectory support,
     serve-toss suppression and FPS-based event windows
  -> pipeline/runtime.py
     render trajectory, mini court and bounce markers
  -> result.mp4 + events/bounce_events.csv
```

## Directory Layout

```text
v5_2_integrated_pipeline/
  run_pipeline.py                 single recommended entry point
  tracking/
    model.py                      lightweight TrackNet heatmap network
    heatmap_candidates.py         frame loading, inference and top-k peaks
    track_video.py                stable V3.7 trajectory post-processing
  context/
    player.py                     MediaPipe player and hand context
    court.py                      court lines, homography and mini court
  events/
    cleanup.py                    bidirectional trajectory cleanup
    detector.py                   hit/bounce feature and event logic
    timing_refiner.py             conservative local contact-frame refinement
  pipeline/
    recognize_events.py           trajectory CSV -> event recognition
    runtime.py                    context construction and video rendering
  evaluation/
    evaluate_events.py            Precision/Recall/F1 evaluation
```

The model checkpoint is intentionally not duplicated. By default V5.2 uses:

```text
exps/lite_heatmap_v37_clean_hardneg_batch8_360x640/
  snapshots/best_before_error_audit_20260613_191025/model_best_f1.pt
```

## Run

```powershell
python versions\v5_2_integrated_pipeline\run_pipeline.py `
  --video-path ".\示例视频1.mp4" `
  --output-dir ".\exps\v5_2" `
  --device cuda
```

Optional output name:

```powershell
  --prefix sample1
```

Useful switches:

```text
--batch-size 2
--bounce-threshold 0.55
--disable-mediapipe
--disable-auto-court
--disable-mini-court
--disable-serve-toss-suppression
--disable-fps-adaptive
```

## Output

For `--prefix sample1`, files are organized as:

```text
exps/v5_2/sample1/
  result.mp4
  tracking/
    trajectory.mp4
    trajectory.csv
  events/
    context_trajectory.csv
    bounce_events.csv
  cache/
    player_context.pkl
```

## Evaluation

```powershell
python versions\v5_2_integrated_pipeline\evaluation\evaluate_events.py `
  --predictions ".\exps\v5_2\sample1\events\bounce_events.csv" `
  --manual ".\manual_bounces.csv" `
  --tolerances "3,5,8"
```

V5.2 currently recognizes bounce events with `event_mode=v37_bounce`. MediaPipe
and automatic court mapping provide contextual evidence; the V3.7 heatmap model
remains the only neural network used for ball localization.

## Contact-frame refinement

Event detection and exact contact timing are separate stages. After an event is
found, `events/timing_refiner.py` fits the trajectory before and after each
candidate as two local motion segments. It may move the representative frame
only when all of the following hold:

- the existing timing confidence is below `0.60`;
- the existing `frame_start..frame_end` interval is at least two frames wide;
- the alternative frame stays inside that interval;
- the piecewise-motion score improves by at least `0.06`.

The event CSV records `original_frame`, `frame_offset`, and `timing_method`.
Disable this stage with `--disable-contact-timing-refinement`.

Existing outputs can be refined without rerunning TrackNet:

```powershell
python versions\v5_2_integrated_pipeline\events\timing_refiner.py `
  --events-in events\bounce_events.csv `
  --track-csv events\context_trajectory.csv `
  --events-out events\bounce_events_refined.csv `
  --fps 59.94
```
