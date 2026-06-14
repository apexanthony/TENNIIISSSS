# V5.1 Reference-Assisted Pipeline

V5.1 is an integrated event-analysis pipeline inspired by the reference project.

It keeps the lightweight TrackNet model focused on ball localization and moves event understanding into geometry-based post-processing.

```text
video
  -> V3 TrackNet heatmap tracker
  -> V3.1 candidate stabilization + heatmap shape quality
  -> optional MediaPipe/player-box context
  -> optional court region / court homography context
  -> hit/bounce event scoring
  -> annotated video + track/event CSV
```

## Why V5.1

Previous V4 bounce-only logic was too dependent on trajectory shape. It could confuse a player hit with a bounce. V5.1 adds player context, so trajectory changes near the player are treated as hit candidates first, while trajectory changes away from the player remain bounce candidates.

## Main Command

V3.7 + V5.1 recommended end-to-end command:

```powershell
python versions\v5_1_reference_pipeline\infer_video_v37_v51.py ^
  --video-path .\input.mp4 ^
  --output-dir .\exps\v5_1_v37 ^
  --prefix demo
```

This command runs:

```text
V3.7 360x640 ball heatmap
-> adaptive motion gate and short/static false-positive cleanup
-> MediaPipe player contact context
-> automatic court-line homography
-> v37_bounce flight-segment state machine
-> FPS-adaptive event windows + serve-toss suppression
-> image/mini-court dual-space bounce evidence
-> local contact-frame refinement
-> final track/event CSV and annotated video
```

`v37_bounce` uses V3.7 metadata (`source`, `accepted`, interpolation state and
track support). Interpolated coordinates may provide temporal context but can
never become the center of a bounce event. When an automatic court map is
available, frames before court initialization and clearly out-of-court racket
contacts are excluded from bounce candidates.

Product-style automatic mode:

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

This mode enables:

```text
V3 TrackNet ball tracking
MediaPipe Pose player hands/feet/body detection
Canny + HoughLines automatic court-line homography
V5.1 hit/bounce geometry
reference-style rendering with a mini-court panel
```

Reference-style bounce-only mode:

```powershell
python versions\v5_1_reference_pipeline\infer_video_v51_pipeline.py ^
  --auto-product-mode ^
  --event-mode reference_bounce ^
  --model-path .\exps\lite_heatmap_v3_270x480_from240\model_best_thr070_pw15.pt ^
  --video-path .\input.mp4 ^
  --video-out-path .\exps\v5_1_reference_pipeline\reference_bounce.mp4 ^
  --track-csv-out .\exps\v5_1_reference_pipeline\reference_bounce_track.csv ^
  --event-csv-out .\exps\v5_1_reference_pipeline\reference_bounce_events.csv ^
  --input-height 270 ^
  --input-width 480 ^
  --threshold 0.30 ^
  --top-k 5 ^
  --bounce-threshold 0.72 ^
  --min-event-gap-seconds 0.60 ^
  --min-flight-seconds 0.15 ^
  --max-bounce-step 260 ^
  --codec mp4v ^
  --device cuda
```

This mode follows the reference pipeline more closely:

```text
TrackNet finds the ball.
MediaPipe estimates player hands/feet/body.
Canny + HoughLines estimates court homography.
mini-court coordinates provide geometry context.
near-hand/contact zones reset flying state.
bounce is detected only in flying state from y-acceleration / y-turn evidence.
```

Apply reference-style auxiliary modules to an existing V3/V5.1 track CSV:

```powershell
python versions\v5_1_reference_pipeline\apply_reference_assist.py ^
  --auto-product-mode ^
  --event-mode reference_bounce ^
  --video-path .\input.mp4 ^
  --track-csv-in .\exps\v5_1_reference_pipeline\input_track.csv ^
  --track-csv-out .\exps\v5_1_reference_pipeline\reference_assist_track.csv ^
  --event-csv-out .\exps\v5_1_reference_pipeline\reference_assist_events.csv ^
  --video-out-path .\exps\v5_1_reference_pipeline\reference_assist.mp4 ^
  --bounce-threshold 0.72 ^
  --min-event-gap-seconds 0.60 ^
  --min-flight-seconds 0.15 ^
  --max-bounce-step 260 ^
  --codec mp4v
```

This is useful when TrackNet output already exists. It skips model inference and
only adds:

```text
MediaPipe player hands/feet/body context
Canny + HoughLines court homography
mini-court coordinate output
reference-style hit/bounce geometry
annotated video rendering
```

## FPS-Adaptive Event Timing

V5.1 event parameters are expressed in seconds and converted with the input
video FPS at runtime. The defaults preserve the previous 30 FPS behavior:

```text
feature sample interval:       0.033 s
feature discontinuity window:  0.067 s
minimum event gap:             0.600 s
minimum flying duration:       0.150 s
refinement search:            -0.120 s to +0.200 s
fast refinement search:       -0.120 s to +0.500 s
```

For example, `--min-event-gap-seconds 0.60` becomes 18 frames at 30 FPS and
36 frames at 60 FPS. The deprecated `--min-event-gap` and
`--min-flight-frames` options remain available as explicit fixed-frame
overrides for reproducing old experiments.

## Serve-Toss Suppression

Serve-toss suppression is enabled by default in `v37_bounce` mode. It detects
an upward, mostly vertical ball release near the serving player when a new
track/point starts. The toss segment through the next player contact is
excluded from bounce candidates. It does not activate during an established
rally, so ordinary upward trajectories after a hit are retained.

Useful controls:

```text
--disable-serve-toss-suppression
--serve-toss-max-seconds 1.20
--serve-toss-min-rise-seconds 0.10
--serve-toss-min-rise-player-ratio 0.22
--serve-toss-max-lateral-player-ratio 0.85
```

## Mini-Court Bounce Evidence

The trajectory drawn in the upper-right mini-court is also used by the event
detector. V5.1 measures mapped direction change, y-direction reversal,
acceleration and speed change, then compares those signals with the original
image trajectory.

```text
image change + mini-court change agree: bounce score increases
weak image change + stable mini-court change: weak bounce evidence is recovered
mini-court-only discontinuity: homography-jitter penalty is applied
```

The mini-court path is an auxiliary signal rather than an independent judge.
This is important because automatic Canny/Hough court homographies can change
slightly between updates. Event CSV files expose `court_evidence` and
`court_agreement`; strongly supported events use the source
`v37_dual_space_refined`.

## Six-Stage Robustness Upgrade

The current V5.1 pipeline includes the following six coordinated upgrades:

```text
1. court homography validation, temporal corner smoothing and quality decay
2. waiting/toss/contact/flying/approaching-player event state machine
3. dual-space contact-onset refinement with event frame intervals
4. reliability-aware fusion of court, player and track evidence
5. two-player temporal pose stabilization and invalid-person filtering
6. conservative forward/backward trajectory outlier cleanup
```

Automatic court mapping writes `court_quality` to the track CSV. Event CSV
files additionally write `frame_start`, `frame_end` and `timing_confidence`.
Low-quality court mappings can still be rendered but contribute less to event
scoring. A mapped ball outside the court does not itself terminate a flight,
because airborne balls do not lie on the court homography plane.

MediaPipe contexts can be cached for repeated event tuning:

```powershell
--mediapipe-complexity 0 ^
--mediapipe-stride 2 ^
--player-context-cache .\exps\demo\player_context.pkl
```

The first run creates the cache. Later runs load it without initializing
MediaPipe. One Pose graph is reused sequentially for both player crops to keep
memory usage bounded.

Evaluate an event CSV against manually verified frame ranges:

```powershell
python tools\evaluate_event_predictions.py ^
  --predictions .\exps\demo\events.csv ^
  --manual .\exps\demo\manual_validation.csv ^
  --tolerances 3,5,8 ^
  --out-csv .\exps\demo\metrics.csv
```

Research/evaluation mode:

```powershell
python versions\v5_1_reference_pipeline\infer_video_v51_pipeline.py ^
  --model-path .\exps\lite_heatmap_v3_270x480_from240\model_best_thr070_pw15.pt ^
  --video-path .\绀轰緥瑙嗛1.mp4 ^
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

Enable MediaPipe player-pose context:

```powershell
python -m pip install mediapipe
python versions\v5_1_reference_pipeline\infer_video_v51_pipeline.py ... --use-mediapipe
```

Use a manually annotated playable court region:

```powershell
--court-region-json .\configs\court_regions\game1.json
```

Use best-effort automatic court-line homography:

```powershell
--auto-court-lines --court-detect-stride 15
```

Debug MediaPipe player crops before event detection:

```powershell
python tools\debug_mediapipe_crops.py ^
  --video-path .\input.mp4 ^
  --player-crop-json .\configs\player_crops\broadcast_court_only.json ^
  --frames 0,80,160,240,320,400 ^
  --out-dir .\exps\v5_1_reference_pipeline\mediapipe_debug
```

The debug images draw:

```text
gray boxes: MediaPipe search crops
orange boxes/circles: detected player body, hands, and hit radius
```

If MediaPipe detects spectators, ball kids, or line judges, narrow the crop JSON
so each crop mostly covers one playable half-court/player area.

Use mapped Roboflow player boxes without MediaPipe:

```powershell
python versions\v5_1_reference_pipeline\infer_video_v51_pipeline.py ^
  --model-path .\exps\lite_heatmap_v3_270x480_from240\model_best_thr070_pw15.pt ^
  --video-path .\input.mp4 ^
  --video-out-path .\exps\v5_1_reference_pipeline\demo_v51_context.mp4 ^
  --track-csv-out .\exps\v5_1_reference_pipeline\demo_v51_context_track.csv ^
  --event-csv-out .\exps\v5_1_reference_pipeline\demo_v51_context_events.csv ^
  --annotations-csv .\datasets\tennis_all_v4i_clip_split\annotations_clip_split.csv ^
  --game game1 ^
  --clip Clip1 ^
  --court-region-json .\configs\court_regions\game1.json ^
  --draw-players ^
  --draw-court ^
  --device cuda
```

The player/court context affects candidate selection:

```text
final_candidate_score =
  heatmap_score
  - motion_distance_penalty
  - player_body_or_shoe_penalty
  - non_court_region_penalty
  + heatmap_shape_quality_bonus
  + optional_hough_circle_bonus
```

This is intended to reduce false peaks on player shoes/body and non-court regions before hit/bounce event scoring.

When `--annotations-csv --game --clip` are provided, V5.1 also reads `court_1...court_14` and maps image coordinates to normalized mini-court coordinates.

## Outputs

```text
track CSV:
  frame,x,y,score,raw_x,raw_y,raw_score,shape_quality,hough_quality,court_x,court_y,court_quality,accepted,source,gap

event CSV:
  frame,event_type,x,y,score,near_player,accel,angle_change,track_support,observed_count,court_evidence,court_agreement,source,frame_start,frame_end,timing_confidence

video:
  red/yellow trajectory trail
  orange HIT markers
  cyan BOUNCE markers
  optional player/court context overlays
```

Thin trajectory rendering can be controlled with:

```text
--trace-radius 3
--trace-min-radius 1
--trace-decay 5
```

Rendering modes:

```text
--render-style reference
  Default in V5.1. Draws a clearer reference-like output:
  red/yellow dotted ball trail, highlighted current ball, compact event rings,
  and an optional mini-court panel.

--draw-mini-court
  Draws the right-side mini-court panel. It is enabled automatically by
  --auto-product-mode.

--render-style legacy
  Uses the earlier V5 overlay style with larger trajectory dots and labels.
```

## Current Status

This is a V5.1 prototype. It is intended for rapid validation of the reference-style idea:

```text
TrackNet finds the ball.
Geometry explains hit and bounce events.
```

Implemented:

```text
1. Player bbox context from simple CSV or mapped Roboflow annotations.
2. Upper-body / lower-body / inside-body split from player bbox.
3. Player-body/shoe penalty during ball-candidate selection.
4. Heatmap connected-component shape quality during ball-candidate selection.
5. Optional HoughCircles quality during ball-candidate selection.
6. Configurable MediaPipe player crop regions via `--player-crop-json`.
7. near-player hit state machine with local-minimum hit confirmation.
8. court-region penalty during ball-candidate selection.
9. court keypoints -> mini-court coordinate mapping in track CSV.
10. optional automatic court-line homography fallback.
11. event-level hit/bounce sequencing and global mutual suppression.
12. clipped court-coordinate event features to reduce homography jitter.
13. `reference_bounce` mode with flying-state bounce detection.
14. configurable thin trajectory rendering.
15. CSV adapter for applying reference-style auxiliary modules to existing V3/V5.1 tracks.
16. MediaPipe crop visualization and duplicate-player suppression for overlapping crops.
```

Known limitation:

```text
If TrackNet trajectory has large false jumps or MediaPipe does not detect hands,
reference_bounce can still over-detect. The next priority is improving V3/V3.1
track stability before adding more bounce thresholds.
```

For `v37_bounce`, MediaPipe still detects one pose per configured crop and may
occasionally select the wrong person. The flight-segment logic therefore uses
player proximity together with court coordinates and trajectory support rather
than treating MediaPipe as a mandatory single source of truth.
