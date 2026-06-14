import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[2]
V5_DIR = Path(__file__).resolve().parent
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))

from court_context import (  # noqa: E402
    load_annotation_court_maps,
    load_playable_polygons,
    transform_point,
)
from event_detector import detect_events, seconds_to_frames, write_event_csv  # noqa: E402
from infer_video_v51_pipeline import (  # noqa: E402
    build_auto_court_maps,
    build_player_contexts,
    draw_outputs,
    nearest_court_map,
    nearest_court_quality,
    track_stats,
    write_track_csv,
)
from trajectory_cleanup import clean_bidirectional_outliers  # noqa: E402


Point = Tuple[float, float]


def _float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return float(text)


def _point_from_row(row, x_name: str, y_name: str) -> Optional[Point]:
    x = _float_or_none(row.get(x_name))
    y = _float_or_none(row.get(y_name))
    if x is None or y is None:
        return None
    return x, y


def _read_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return width, height, fps, frame_count


def read_track_csv(path: str, frame_count: int):
    rows = {}
    max_frame = frame_count - 1
    with open(path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            frame = int(float(row.get("frame") or row.get("frame_id") or len(rows)))
            max_frame = max(max_frame, frame)
            rows[frame] = row

    size = max_frame + 1
    stable = {
        "tracks": [None] * size,
        "scores": [None] * size,
        "shape_quality": [None] * size,
        "hough_quality": [None] * size,
        "raw_tracks": [None] * size,
        "raw_scores": [None] * size,
        "accepted": [0] * size,
        "source": ["track_csv"] * size,
        "gaps": [0] * size,
    }

    gap = 0
    for frame in range(size):
        row = rows.get(frame)
        if row is None:
            gap += 1
            stable["gaps"][frame] = gap
            continue

        point = _point_from_row(row, "x", "y")
        raw_point = _point_from_row(row, "raw_x", "raw_y") or point
        score = _float_or_none(row.get("score") or row.get("confidence"))
        raw_score = _float_or_none(row.get("raw_score")) or score
        stable["tracks"][frame] = point
        stable["scores"][frame] = score if score is not None else (1.0 if point is not None else None)
        stable["raw_tracks"][frame] = raw_point
        stable["raw_scores"][frame] = raw_score
        stable["shape_quality"][frame] = _float_or_none(row.get("shape_quality"))
        stable["hough_quality"][frame] = _float_or_none(row.get("hough_quality"))
        stable["accepted"][frame] = int(float(row.get("accepted") or (1 if point is not None else 0)))
        stable["source"][frame] = row.get("source") or "track_csv"
        gap = 0 if point is not None else gap + 1
        stable["gaps"][frame] = gap

    stable["max_step"] = 0.0
    stable["hard_step"] = 0.0
    return stable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply reference-style MediaPipe/court/mini-court event logic to an existing V3/V5.1 track CSV."
    )
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--track-csv-in", required=True)
    parser.add_argument("--track-csv-out", required=True)
    parser.add_argument("--event-csv-out", required=True)
    parser.add_argument("--video-out-path", required=True)
    parser.add_argument("--player-box-csv", default="")
    parser.add_argument("--annotations-csv", default="")
    parser.add_argument("--game", default="")
    parser.add_argument("--clip", default="")
    parser.add_argument("--court-region-json", default="")
    parser.add_argument("--auto-product-mode", action="store_true")
    parser.add_argument("--auto-court-lines", action="store_true")
    parser.add_argument("--court-detect-stride", type=int, default=15)
    parser.add_argument("--use-mediapipe", action="store_true")
    parser.add_argument("--mediapipe-complexity", type=int, default=1)
    parser.add_argument("--mediapipe-stride", type=int, default=1)
    parser.add_argument("--player-crop-json", default="")
    parser.add_argument("--player-context-cache", default="")
    parser.add_argument(
        "--event-mode",
        choices=["all", "bounce_only", "reference_bounce", "v37_bounce"],
        default="reference_bounce",
    )
    parser.add_argument("--hit-threshold", type=float, default=0.55)
    parser.add_argument("--bounce-threshold", type=float, default=0.72)
    parser.add_argument("--min-event-gap-seconds", type=float, default=0.60)
    parser.add_argument("--min-event-gap", type=int, default=None, help="Deprecated fixed-frame override.")
    parser.add_argument("--event-frame-margin", type=float, default=4.0)
    parser.add_argument("--contact-threshold", type=float, default=0.18)
    parser.add_argument("--min-flight-seconds", type=float, default=0.15)
    parser.add_argument("--min-flight-frames", type=int, default=None, help="Deprecated fixed-frame override.")
    parser.add_argument("--feature-sample-seconds", type=float, default=1.0 / 30.0)
    parser.add_argument("--feature-gap-seconds", type=float, default=2.0 / 30.0)
    parser.add_argument("--refine-backward-seconds", type=float, default=0.12)
    parser.add_argument("--refine-forward-seconds", type=float, default=6.0 / 30.0)
    parser.add_argument("--refine-fast-forward-seconds", type=float, default=0.50)
    parser.add_argument("--refine-merge-seconds", type=float, default=2.0 / 30.0)
    parser.add_argument("--disable-serve-toss-suppression", action="store_true")
    parser.add_argument("--disable-bidirectional-cleanup", action="store_true")
    parser.add_argument("--serve-toss-max-seconds", type=float, default=1.20)
    parser.add_argument("--serve-toss-min-rise-seconds", type=float, default=0.10)
    parser.add_argument("--serve-toss-min-rise-player-ratio", type=float, default=0.22)
    parser.add_argument("--serve-toss-max-lateral-player-ratio", type=float, default=0.85)
    parser.add_argument("--max-bounce-step", type=float, default=260.0)
    parser.add_argument("--min-bounce-y-accel", type=float, default=8.0)
    parser.add_argument("--trace", type=int, default=14)
    parser.add_argument("--trace-radius", type=int, default=3)
    parser.add_argument("--trace-min-radius", type=int, default=1)
    parser.add_argument("--trace-decay", type=int, default=5)
    parser.add_argument("--render-style", choices=["reference", "legacy"], default="reference")
    parser.add_argument("--draw-mini-court", action="store_true")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--draw-raw", action="store_true")
    parser.add_argument("--draw-players", action="store_true")
    parser.add_argument("--draw-court", action="store_true")
    args = parser.parse_args()
    if args.auto_product_mode:
        args.use_mediapipe = True
        args.auto_court_lines = True
        args.draw_players = True
        args.draw_court = True
        args.draw_mini_court = True
    return args


def main() -> None:
    args = parse_args()
    Path(args.track_csv_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.event_csv_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.video_out_path).parent.mkdir(parents=True, exist_ok=True)

    width, height, fps, frame_count = _read_video_meta(args.video_path)
    stable = read_track_csv(args.track_csv_in, frame_count)
    if not args.disable_bidirectional_cleanup:
        stable = clean_bidirectional_outliers(stable, fps)
    court_polygons = load_playable_polygons(args.court_region_json)
    court_maps = load_annotation_court_maps(args.annotations_csv, args.game, args.clip)
    if not court_maps and args.auto_court_lines:
        court_maps = build_auto_court_maps(args)
    player_contexts = build_player_contexts(args, width, height)

    court_tracks = [transform_point(nearest_court_map(court_maps, idx), point) for idx, point in enumerate(stable["tracks"])]
    court_qualities = [nearest_court_quality(court_maps, idx) for idx in range(len(stable["tracks"]))]
    events = detect_events(
        stable["tracks"],
        stable["scores"],
        player_contexts,
        court_polygons,
        fps,
        court_tracks=court_tracks,
        court_qualities=court_qualities,
        hit_threshold=args.hit_threshold,
        bounce_threshold=args.bounce_threshold,
        min_event_gap=args.min_event_gap,
        min_event_gap_seconds=args.min_event_gap_seconds,
        event_mode=args.event_mode,
        frame_width=width,
        frame_height=height,
        frame_margin=args.event_frame_margin,
        contact_threshold=args.contact_threshold,
        min_flight_frames=args.min_flight_frames,
        min_flight_seconds=args.min_flight_seconds,
        feature_sample_seconds=args.feature_sample_seconds,
        feature_gap_seconds=args.feature_gap_seconds,
        refine_backward_seconds=args.refine_backward_seconds,
        refine_forward_seconds=args.refine_forward_seconds,
        refine_fast_forward_seconds=args.refine_fast_forward_seconds,
        refine_merge_seconds=args.refine_merge_seconds,
        suppress_serve_toss=not args.disable_serve_toss_suppression,
        serve_toss_max_seconds=args.serve_toss_max_seconds,
        serve_toss_min_rise_seconds=args.serve_toss_min_rise_seconds,
        serve_toss_min_rise_player_ratio=args.serve_toss_min_rise_player_ratio,
        serve_toss_max_lateral_player_ratio=args.serve_toss_max_lateral_player_ratio,
        max_bounce_step=args.max_bounce_step,
        min_bounce_y_accel=args.min_bounce_y_accel,
        track_sources=stable["source"],
        accepted_flags=stable["accepted"],
    )

    write_track_csv(args.track_csv_out, stable, court_maps)
    write_event_csv(args.event_csv_out, events)
    draw_outputs(args, stable, events, player_contexts, court_polygons, court_maps)

    stats = track_stats(stable["tracks"])
    hit_count = sum(1 for event in events if event.event_type == "hit")
    bounce_count = sum(1 for event in events if event.event_type == "bounce")
    print(f"track_csv_in={args.track_csv_in}", flush=True)
    print(f"track_csv_out={args.track_csv_out}", flush=True)
    print(f"event_csv={args.event_csv_out}", flush=True)
    print(f"video_out={args.video_out_path}", flush=True)
    event_gap_frames = args.min_event_gap or seconds_to_frames(args.min_event_gap_seconds, fps)
    flight_frames = args.min_flight_frames or seconds_to_frames(args.min_flight_seconds, fps)
    print(
        "event_timing: fps={:.3f}, min_gap={}f/{:.3f}s, min_flight={}f/{:.3f}s, toss_suppression={}".format(
            fps,
            event_gap_frames,
            event_gap_frames / max(fps, 1.0),
            flight_frames,
            flight_frames / max(fps, 1.0),
            int(not args.disable_serve_toss_suppression),
        ),
        flush=True,
    )
    print(f"bidirectional_removed={stable.get('bidirectional_removed', 0)}", flush=True)
    print(
        "track_stats: valid={}, jump>80={}, jump>120={}, jump>240={}, max_jump={:.2f}".format(
            stats["valid"],
            stats["jump80"],
            stats["jump120"],
            stats["jump240"],
            stats["max_jump"],
        ),
        flush=True,
    )
    print(f"events: hit={hit_count}, bounce={bounce_count}, total={len(events)}", flush=True)


if __name__ == "__main__":
    main()
