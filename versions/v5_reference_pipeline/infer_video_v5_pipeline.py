import argparse
import csv
import sys
from collections import deque
from pathlib import Path
from typing import Dict

import cv2

ROOT = Path(__file__).resolve().parents[2]
V31_DIR = ROOT / "versions" / "v3_lightweight" / "v3_1_stable_postprocess"
if str(V31_DIR) not in sys.path:
    sys.path.insert(0, str(V31_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from court_context import (  # noqa: E402
    detect_court_homography_from_frame,
    draw_polygons,
    load_annotation_court_maps,
    load_playable_polygons,
    transform_point,
)
from court_context import point_in_polygons  # noqa: E402
from event_detector import detect_events, write_event_csv  # noqa: E402
from infer_on_video_v31_stable import (  # noqa: E402
    choose_candidate,
    distance_xy,
    infer_candidate_rows,
    interpolate_short_gaps,
    track_stats,
)
from player_context import (  # noqa: E402
    MediaPipePoseProvider,
    PlayerFrameContext,
    draw_player_context,
    empty_player_context,
    load_annotation_player_contexts,
    load_player_box_csv,
    player_candidate_penalty,
)


def build_auto_court_maps(args):
    if not args.auto_court_lines:
        return {}
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for court-line detection: {args.video_path}")
    maps = {}
    frame_id = 0
    last_matrix = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % max(1, args.court_detect_stride) == 0:
                matrix = detect_court_homography_from_frame(frame)
                if matrix is not None:
                    last_matrix = matrix
            if last_matrix is not None:
                maps[frame_id] = last_matrix
            frame_id += 1
    finally:
        cap.release()
    return maps


def build_player_contexts(args, width: int, height: int) -> Dict[int, PlayerFrameContext]:
    contexts = load_player_box_csv(args.player_box_csv)
    contexts.update(load_annotation_player_contexts(args.annotations_csv, args.game, args.clip))
    if not args.use_mediapipe:
        return contexts

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for MediaPipe: {args.video_path}")

    provider = MediaPipePoseProvider(width, height, args.mediapipe_complexity, args.player_crop_json)
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % max(1, args.mediapipe_stride) == 0:
                contexts[frame_id] = provider.process(frame)
            frame_id += 1
    finally:
        provider.close()
        cap.release()
    return contexts


def nearest_context(contexts: Dict[int, PlayerFrameContext], frame_id: int, max_gap: int = 2) -> PlayerFrameContext:
    if frame_id in contexts:
        return contexts[frame_id]
    for gap in range(1, max_gap + 1):
        if frame_id - gap in contexts:
            return contexts[frame_id - gap]
        if frame_id + gap in contexts:
            return contexts[frame_id + gap]
    return empty_player_context()


def choose_candidate_with_context(candidates, predicted, max_step, hard_step, args, ctx, court_polygons):
    best = None
    best_value = -1e9
    for cand in candidates:
        point = (cand["x"], cand["y"])
        dist = distance_xy(point, predicted)
        if dist > hard_step:
            continue
        value = cand["score"] - args.motion_weight * min(dist / max(max_step, 1e-6), 4.0)

        if args.player_candidate_penalty > 0:
            value -= args.player_candidate_penalty * player_candidate_penalty(point, ctx)

        if court_polygons and not point_in_polygons(point, court_polygons):
            value -= args.court_candidate_penalty

        shape_quality = float(cand.get("shape_quality", 0.5))
        value += args.shape_quality_weight * (shape_quality - 0.5)
        hough_quality = float(cand.get("hough_quality", 0.0))
        value += args.hough_quality_weight * hough_quality

        if value > best_value:
            best = cand
            best_value = value
    return best


def stabilize_candidates_with_context(candidate_rows, width, height, fps, args, player_contexts, court_polygons):
    import math

    diag = math.hypot(width, height)
    base_step = args.max_step if args.max_step > 0 else diag * args.max_step_ratio
    if args.fps_adaptive:
        base_step *= 30.0 / max(float(fps), 1.0)
    max_step = max(args.min_step, base_step)
    hard_step = max_step * args.hard_step_factor

    tracks = []
    scores = []
    shape_quality = []
    hough_quality = []
    raw_tracks = []
    raw_scores = []
    accepted_flags = []
    gap_lengths = []

    last_pos = None
    last_out = None
    velocity = (0.0, 0.0)
    gap = 0

    for frame_id, candidates in enumerate(candidate_rows):
        raw = candidates[0] if candidates else None
        raw_tracks.append(None if raw is None else (raw["x"], raw["y"]))
        raw_scores.append(None if raw is None else raw["score"])

        ctx = nearest_context(player_contexts, frame_id)
        chosen = None
        predicted = None
        if last_pos is None or gap > args.reinit_gap:
            predicted = (raw["x"], raw["y"]) if raw is not None else (0.0, 0.0)
            chosen = choose_candidate_with_context(
                candidates,
                predicted,
                max_step,
                hard_step,
                args,
                ctx,
                court_polygons,
            ) if candidates else None
        elif candidates:
            dt = gap + 1
            predicted = (last_pos[0] + velocity[0] * dt, last_pos[1] + velocity[1] * dt)
            chosen = choose_candidate_with_context(
                candidates,
                predicted,
                max_step,
                hard_step,
                args,
                ctx,
                court_polygons,
            )

        if chosen is None:
            tracks.append(None)
            scores.append(None if raw is None else raw["score"])
            shape_quality.append(None if raw is None else raw.get("shape_quality"))
            hough_quality.append(None if raw is None else raw.get("hough_quality"))
            accepted_flags.append(0)
            gap += 1
            gap_lengths.append(gap)
            continue

        point = (chosen["x"], chosen["y"])
        if last_pos is not None and gap <= args.reinit_gap:
            dt = gap + 1
            measured_v = ((point[0] - last_pos[0]) / dt, (point[1] - last_pos[1]) / dt)
            velocity = (
                args.velocity_alpha * measured_v[0] + (1.0 - args.velocity_alpha) * velocity[0],
                args.velocity_alpha * measured_v[1] + (1.0 - args.velocity_alpha) * velocity[1],
            )
        else:
            velocity = (0.0, 0.0)

        if last_out is not None and predicted is not None and args.smooth_alpha < 1.0:
            out = (
                args.smooth_alpha * point[0] + (1.0 - args.smooth_alpha) * predicted[0],
                args.smooth_alpha * point[1] + (1.0 - args.smooth_alpha) * predicted[1],
            )
        else:
            out = point

        tracks.append(out)
        scores.append(chosen["score"])
        shape_quality.append(chosen.get("shape_quality"))
        hough_quality.append(chosen.get("hough_quality"))
        accepted_flags.append(1)
        gap = 0
        gap_lengths.append(gap)
        last_pos = point
        last_out = out

    return {
        "tracks": tracks,
        "scores": scores,
        "shape_quality": shape_quality,
        "hough_quality": hough_quality,
        "raw_tracks": raw_tracks,
        "raw_scores": raw_scores,
        "accepted": accepted_flags,
        "source": ["accepted" if flag else "missing" for flag in accepted_flags],
        "gaps": gap_lengths,
        "max_step": max_step,
        "hard_step": hard_step,
    }


def nearest_court_map(court_maps, frame_id: int, max_gap: int = 2):
    if frame_id in court_maps:
        return court_maps[frame_id]
    for gap in range(1, max_gap + 1):
        if frame_id - gap in court_maps:
            return court_maps[frame_id - gap]
        if frame_id + gap in court_maps:
            return court_maps[frame_id + gap]
    return None


def write_track_csv(path: str, stable, court_maps=None) -> None:
    court_maps = court_maps or {}
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "frame",
                "x",
                "y",
                "score",
                "raw_x",
                "raw_y",
                "raw_score",
                "shape_quality",
                "hough_quality",
                "court_x",
                "court_y",
                "accepted",
                "source",
                "gap",
            ]
        )
        for frame_id, point in enumerate(stable["tracks"]):
            raw = stable["raw_tracks"][frame_id]
            score = stable["scores"][frame_id]
            raw_score = stable["raw_scores"][frame_id]
            shape_quality = stable.get("shape_quality", [None] * len(stable["tracks"]))[frame_id]
            hough_quality = stable.get("hough_quality", [None] * len(stable["tracks"]))[frame_id]
            court_point = transform_point(nearest_court_map(court_maps, frame_id), point)
            writer.writerow(
                [
                    frame_id,
                    "" if point is None else f"{point[0]:.2f}",
                    "" if point is None else f"{point[1]:.2f}",
                    "" if score is None else f"{score:.6f}",
                    "" if raw is None else f"{raw[0]:.2f}",
                    "" if raw is None else f"{raw[1]:.2f}",
                    "" if raw_score is None else f"{raw_score:.6f}",
                    "" if shape_quality is None else f"{shape_quality:.6f}",
                    "" if hough_quality is None else f"{hough_quality:.6f}",
                    "" if court_point is None else f"{court_point[0]:.2f}",
                    "" if court_point is None else f"{court_point[1]:.2f}",
                    stable["accepted"][frame_id],
                    stable["source"][frame_id],
                    stable["gaps"][frame_id],
                ]
            )


def draw_outputs(args, stable, events, player_contexts, court_polygons) -> None:
    event_by_frame = {}
    for event in events:
        event_by_frame.setdefault(event.frame, []).append(event)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.video_out_path, cv2.VideoWriter_fourcc(*args.codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {args.video_out_path}")

    trace = deque(maxlen=args.trace)
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id >= len(stable["tracks"]):
                break

            if args.draw_court:
                draw_polygons(frame, court_polygons)
            if args.draw_players:
                draw_player_context(frame, nearest_context(player_contexts, frame_id))

            point = stable["tracks"][frame_id]
            raw = stable["raw_tracks"][frame_id]
            trace.append(point)
            if args.draw_raw and raw is not None:
                cv2.circle(frame, (int(raw[0]), int(raw[1])), 5, (255, 180, 0), 1)
            for age, pt in enumerate(reversed(trace)):
                if pt is None:
                    continue
                radius = max(args.trace_min_radius, args.trace_radius - age // max(1, args.trace_decay))
                color = (0, 0, 255) if age == 0 else (0, 220, 255)
                cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)

            for event in event_by_frame.get(frame_id, []):
                color = (0, 165, 255) if event.event_type == "hit" else (255, 255, 0)
                label = "HIT" if event.event_type == "hit" else "BOUNCE"
                cv2.circle(frame, (int(event.x), int(event.y)), 16, color, 3)
                cv2.putText(
                    frame,
                    f"{label} {event.score:.2f}",
                    (int(event.x) + 12, max(20, int(event.y) - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            frame_id += 1
    finally:
        writer.release()
        cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="V5 reference-style TrackNet + player/court event pipeline.")
    parser.add_argument(
        "--auto-product-mode",
        action="store_true",
        help="Enable product-style automatic mode: TrackNet + MediaPipe players + automatic court-line homography.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--video-out-path", required=True)
    parser.add_argument("--track-csv-out", required=True)
    parser.add_argument("--event-csv-out", required=True)
    parser.add_argument("--player-box-csv", default="")
    parser.add_argument("--annotations-csv", default="")
    parser.add_argument("--game", default="")
    parser.add_argument("--clip", default="")
    parser.add_argument("--court-region-json", default="")
    parser.add_argument("--auto-court-lines", action="store_true")
    parser.add_argument("--court-detect-stride", type=int, default=15)
    parser.add_argument("--use-mediapipe", action="store_true")
    parser.add_argument("--mediapipe-complexity", type=int, default=1)
    parser.add_argument("--mediapipe-stride", type=int, default=1)
    parser.add_argument("--player-crop-json", default="")
    parser.add_argument("--base-channels", dest="base_channels", type=int, default=24)
    parser.add_argument("--input-height", dest="input_height", type=int, default=270)
    parser.add_argument("--input-width", dest="input_width", type=int, default=480)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--peak-window", dest="peak_window", type=int, default=15)
    parser.add_argument("--top-k", dest="top_k", type=int, default=5)
    parser.add_argument("--suppress-window", dest="suppress_window", type=int, default=21)
    parser.add_argument("--max-step", dest="max_step", type=float, default=240.0)
    parser.add_argument("--max-step-ratio", dest="max_step_ratio", type=float, default=0.08)
    parser.add_argument("--min-step", dest="min_step", type=float, default=45.0)
    parser.add_argument("--hard-step-factor", dest="hard_step_factor", type=float, default=6.0)
    parser.add_argument("--motion-weight", dest="motion_weight", type=float, default=0.12)
    parser.add_argument("--player-candidate-penalty", type=float, default=0.45)
    parser.add_argument("--court-candidate-penalty", type=float, default=0.20)
    parser.add_argument("--shape-quality-weight", type=float, default=0.12)
    parser.add_argument("--hough-quality-weight", type=float, default=0.08)
    parser.add_argument("--enable-hough-quality", action="store_true")
    parser.add_argument("--velocity-alpha", dest="velocity_alpha", type=float, default=0.60)
    parser.add_argument("--smooth-alpha", dest="smooth_alpha", type=float, default=1.00)
    parser.add_argument("--reinit-gap", dest="reinit_gap", type=int, default=6)
    parser.add_argument("--interp-gap", dest="interp_gap", type=int, default=4)
    parser.add_argument("--fps-adaptive", dest="fps_adaptive", action="store_true")
    parser.add_argument("--hit-threshold", type=float, default=0.55)
    parser.add_argument("--bounce-threshold", type=float, default=0.60)
    parser.add_argument("--event-mode", choices=["all", "bounce_only", "reference_bounce"], default="all")
    parser.add_argument("--event-frame-margin", type=float, default=4.0)
    parser.add_argument("--contact-threshold", type=float, default=0.18)
    parser.add_argument("--min-flight-frames", type=int, default=6)
    parser.add_argument("--max-bounce-step", type=float, default=260.0)
    parser.add_argument("--min-bounce-y-accel", type=float, default=8.0)
    parser.add_argument("--min-event-gap", type=int, default=10)
    parser.add_argument("--trace", type=int, default=14)
    parser.add_argument("--trace-radius", type=int, default=3)
    parser.add_argument("--trace-min-radius", type=int, default=1)
    parser.add_argument("--trace-decay", type=int, default=5)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--draw-raw", action="store_true")
    parser.add_argument("--draw-players", action="store_true")
    parser.add_argument("--draw-court", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--print-interval", dest="print_interval", type=int, default=100)
    args = parser.parse_args()
    if args.auto_product_mode:
        args.use_mediapipe = True
        args.auto_court_lines = True
        args.draw_players = True
        args.draw_court = True
        args.fps_adaptive = True
        args.enable_hough_quality = True
    return args


def main() -> None:
    args = parse_args()
    Path(args.video_out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.track_csv_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.event_csv_out).parent.mkdir(parents=True, exist_ok=True)

    candidate_rows, width, height, fps = infer_candidate_rows(args)
    print(
        "v5_mode: auto_product_mode={}, mediapipe={}, auto_court_lines={}, annotations={}".format(
            int(args.auto_product_mode),
            int(args.use_mediapipe),
            int(args.auto_court_lines),
            int(bool(args.annotations_csv)),
        ),
        flush=True,
    )
    player_contexts = build_player_contexts(args, width, height)
    court_polygons = load_playable_polygons(args.court_region_json)
    court_maps = load_annotation_court_maps(args.annotations_csv, args.game, args.clip)
    if not court_maps:
        court_maps = build_auto_court_maps(args)
    stable = stabilize_candidates_with_context(candidate_rows, width, height, fps, args, player_contexts, court_polygons)
    stable = interpolate_short_gaps(stable, args.interp_gap)
    court_tracks = [transform_point(nearest_court_map(court_maps, idx), point) for idx, point in enumerate(stable["tracks"])]
    events = detect_events(
        stable["tracks"],
        stable["scores"],
        player_contexts,
        court_polygons,
        fps,
        court_tracks=court_tracks,
        hit_threshold=args.hit_threshold,
        bounce_threshold=args.bounce_threshold,
        min_event_gap=args.min_event_gap,
        event_mode=args.event_mode,
        frame_width=width,
        frame_height=height,
        frame_margin=args.event_frame_margin,
        contact_threshold=args.contact_threshold,
        min_flight_frames=args.min_flight_frames,
        max_bounce_step=args.max_bounce_step,
        min_bounce_y_accel=args.min_bounce_y_accel,
    )

    write_track_csv(args.track_csv_out, stable, court_maps)
    write_event_csv(args.event_csv_out, events)
    draw_outputs(args, stable, events, player_contexts, court_polygons)

    stats = track_stats(stable["tracks"])
    hit_count = sum(1 for event in events if event.event_type == "hit")
    bounce_count = sum(1 for event in events if event.event_type == "bounce")
    print(f"track_csv={args.track_csv_out}", flush=True)
    print(f"event_csv={args.event_csv_out}", flush=True)
    print(f"video_out={args.video_out_path}", flush=True)
    print(
        "stable_stats: valid={}, jump>80={}, jump>120={}, jump>240={}, max_jump={:.2f}".format(
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
