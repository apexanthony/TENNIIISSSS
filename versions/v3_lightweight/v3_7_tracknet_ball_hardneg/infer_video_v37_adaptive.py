import argparse
import csv
import math
import statistics
import sys
from collections import deque
from pathlib import Path

import cv2


CURRENT_DIR = Path(__file__).resolve().parent
V31_DIR = CURRENT_DIR.parent / "v3_1_stable_postprocess"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(V31_DIR) not in sys.path:
    sys.path.append(str(V31_DIR))

import infer_on_video_v31_stable as v31


def clamp_point(point, width, height):
    return (
        min(max(float(point[0]), 0.0), float(width - 1)),
        min(max(float(point[1]), 0.0), float(height - 1)),
    )


def candidate_value(candidate, predicted, gate, args):
    point = (candidate["x"], candidate["y"])
    distance = v31.distance_xy(point, predicted)
    if distance > gate:
        return None
    shape_bonus = args.shape_weight * candidate.get("shape_quality", 0.0)
    hough_bonus = args.hough_weight * candidate.get("hough_quality", 0.0)
    motion_penalty = args.motion_weight * min(distance / max(gate, 1e-6), 1.0)
    return candidate["score"] + shape_bonus + hough_bonus - motion_penalty


def choose_tracking_candidate(candidates, predicted, strong_gate, weak_gate, args, allow_weak=True):
    best = None
    best_value = -1e9
    best_distance = None
    for candidate in candidates:
        strong = candidate["score"] >= args.high_threshold
        if not strong:
            if not allow_weak:
                continue
            if candidate["score"] < args.low_threshold:
                continue
            if candidate.get("shape_quality", 0.0) < args.min_weak_shape:
                continue
        gate = strong_gate if strong else weak_gate
        value = candidate_value(candidate, predicted, gate, args)
        if value is None:
            continue
        distance = v31.distance_xy((candidate["x"], candidate["y"]), predicted)
        if value > best_value:
            best = candidate
            best_value = value
            best_distance = distance
    return best, best_distance


def dynamic_motion_gate(recent_speeds, args):
    if len(recent_speeds) < 2:
        return args.gate_max
    median_speed = statistics.median(recent_speeds)
    gate = median_speed * args.speed_gate_factor + args.gate_margin
    return min(max(gate, args.gate_min), args.gate_max)


def abrupt_change_confirmed(candidate_rows, frame_id, point, measured_velocity, gate, args):
    next_frame = frame_id + 1
    if next_frame >= len(candidate_rows):
        return False
    expected = (point[0] + measured_velocity[0], point[1] + measured_velocity[1])
    confirmation_gate = min(
        args.gate_max,
        max(args.gate_min, gate, math.hypot(*measured_velocity) * 0.75 + args.gate_margin),
    )
    for candidate in candidate_rows[next_frame]:
        if candidate["score"] < args.high_threshold:
            continue
        candidate_point = (candidate["x"], candidate["y"])
        if v31.distance_xy(candidate_point, expected) <= confirmation_gate:
            return True
    return False


def vector_angle_degrees(a, b):
    norm_a = math.hypot(a[0], a[1])
    norm_b = math.hypot(b[0], b[1])
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    cosine = (a[0] * b[0] + a[1] * b[1]) / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def stabilize_v37(candidate_rows, width, height, fps, args):
    fps_scale = 30.0 / max(float(fps), 1.0) if args.fps_adaptive else 1.0
    gate_min = args.gate_min * fps_scale
    gate_max = args.gate_max * fps_scale
    args.gate_min = min(gate_min, gate_max)
    args.gate_max = max(gate_min, gate_max)

    count = len(candidate_rows)
    tracks = [None] * count
    scores = [None] * count
    accepted = [0] * count
    sources = ["missing"] * count
    raw_tracks = []
    raw_scores = []
    predicted_tracks = [None] * count
    innovations = [None] * count
    shape_quality = [None] * count
    gaps = [0] * count
    motion_gates = [None] * count

    confirmed = False
    seed = None
    last_position = None
    velocity = (0.0, 0.0)
    gap = 0
    recent_speeds = deque(maxlen=args.speed_window)

    for frame_id, candidates in enumerate(candidate_rows):
        raw = candidates[0] if candidates else None
        raw_tracks.append(None if raw is None else (raw["x"], raw["y"]))
        raw_scores.append(None if raw is None else raw["score"])

        if not confirmed:
            strong = next((candidate for candidate in candidates if candidate["score"] >= args.seed_threshold), None)
            if strong is None:
                seed = None
                gaps[frame_id] = gap
                continue
            point = (strong["x"], strong["y"])
            if seed is None:
                seed = (frame_id, strong)
                sources[frame_id] = "seed_pending"
                scores[frame_id] = strong["score"]
                continue

            seed_frame, seed_candidate = seed
            seed_point = (seed_candidate["x"], seed_candidate["y"])
            frame_gap = frame_id - seed_frame
            seed_gate = args.gate_max * args.strong_gate_factor * max(1, frame_gap)
            if frame_gap <= args.seed_max_gap and v31.distance_xy(point, seed_point) <= seed_gate:
                tracks[seed_frame] = seed_point
                scores[seed_frame] = seed_candidate["score"]
                accepted[seed_frame] = 1
                sources[seed_frame] = "seed_confirmed"
                shape_quality[seed_frame] = seed_candidate.get("shape_quality", 0.0)

                tracks[frame_id] = point
                scores[frame_id] = strong["score"]
                accepted[frame_id] = 1
                sources[frame_id] = "high_observed"
                shape_quality[frame_id] = strong.get("shape_quality", 0.0)
                velocity = (
                    (point[0] - seed_point[0]) / max(frame_gap, 1),
                    (point[1] - seed_point[1]) / max(frame_gap, 1),
                )
                recent_speeds.clear()
                recent_speeds.append(math.hypot(*velocity))
                last_position = point
                confirmed = True
                seed = None
                gap = 0
            else:
                seed = (frame_id, strong)
                sources[frame_id] = "seed_pending"
                scores[frame_id] = strong["score"]
            continue

        if gap >= args.max_reconnect_gap:
            confirmed = False
            seed = None
            last_position = None
            velocity = (0.0, 0.0)
            recent_speeds.clear()
            strong = next((candidate for candidate in candidates if candidate["score"] >= args.seed_threshold), None)
            if strong is not None:
                seed = (frame_id, strong)
                sources[frame_id] = "seed_pending"
                scores[frame_id] = strong["score"]
            else:
                sources[frame_id] = "track_expired"
            continue

        gap += 1
        predicted = clamp_point(
            (last_position[0] + velocity[0] * gap, last_position[1] + velocity[1] * gap),
            width,
            height,
        )
        predicted_tracks[frame_id] = predicted
        base_gate = dynamic_motion_gate(recent_speeds, args)
        strong_gate = min(args.gate_max, base_gate * args.strong_gate_factor)
        weak_gate = min(args.gate_max, base_gate * args.weak_gate_factor)
        motion_gates[frame_id] = base_gate
        allow_weak = gap <= args.weak_reconnect_gap
        chosen, innovation = choose_tracking_candidate(
            candidates,
            predicted,
            strong_gate,
            weak_gate,
            args,
            allow_weak=allow_weak,
        )

        if chosen is None:
            gaps[frame_id] = gap
            if not allow_weak:
                sources[frame_id] = "high_conf_reconnect_only"
            continue

        point = (chosen["x"], chosen["y"])
        measured_velocity = (
            (point[0] - last_position[0]) / max(gap, 1),
            (point[1] - last_position[1]) / max(gap, 1),
        )
        turn_angle = vector_angle_degrees(velocity, measured_velocity)
        measured_speed = math.hypot(*measured_velocity)
        baseline_speed = statistics.median(recent_speeds) if len(recent_speeds) >= 3 else None
        acceleration_jump = (
            baseline_speed is not None
            and baseline_speed >= args.min_speed_for_accel_check
            and measured_speed > baseline_speed * args.max_accel_ratio
        )
        strong = chosen["score"] >= args.high_threshold
        abrupt_change = turn_angle >= args.turn_reset_angle or acceleration_jump
        requires_confirmation = (
            (turn_angle >= args.turn_reset_angle and acceleration_jump)
            or turn_angle >= args.extreme_turn_angle
        )
        if abrupt_change and not strong:
            gaps[frame_id] = gap
            sources[frame_id] = "weak_abrupt_rejected"
            continue
        if requires_confirmation and not abrupt_change_confirmed(
            candidate_rows, frame_id, point, measured_velocity, base_gate, args
        ):
            gaps[frame_id] = gap
            sources[frame_id] = "abrupt_unconfirmed"
            continue
        if strong and abrupt_change:
            velocity = measured_velocity
            source = "high_abrupt_confirmed"
        else:
            alpha = args.velocity_alpha if strong else args.weak_velocity_alpha
            velocity = (
                alpha * measured_velocity[0] + (1.0 - alpha) * velocity[0],
                alpha * measured_velocity[1] + (1.0 - alpha) * velocity[1],
            )
            source = "high_observed" if strong else "weak_motion_recovered"

        # V3.7 accepted peaks are already accurately localized. Prediction guides
        # candidate association but never replaces a real observed coordinate.
        tracks[frame_id] = point
        scores[frame_id] = chosen["score"]
        accepted[frame_id] = 1
        sources[frame_id] = source
        innovations[frame_id] = innovation
        shape_quality[frame_id] = chosen.get("shape_quality", 0.0)
        gaps[frame_id] = 0
        last_position = point
        recent_speeds.append(measured_speed)
        gap = 0

    return {
        "tracks": tracks,
        "scores": scores,
        "accepted": accepted,
        "source": sources,
        "raw_tracks": raw_tracks,
        "raw_scores": raw_scores,
        "predicted_tracks": predicted_tracks,
        "innovations": innovations,
        "shape_quality": shape_quality,
        "gaps": gaps,
        "motion_gates": motion_gates,
    }


def remove_temporal_spikes(stable, args):
    tracks = list(stable["tracks"])
    accepted = list(stable["accepted"])
    sources = list(stable["source"])
    for frame_id in range(1, len(tracks) - 1):
        previous = tracks[frame_id - 1]
        current = tracks[frame_id]
        following = tracks[frame_id + 1]
        if previous is None or current is None or following is None:
            continue
        if (
            v31.distance_xy(previous, following) <= args.spike_return_radius
            and v31.distance_xy(previous, current) >= args.spike_distance
            and v31.distance_xy(current, following) >= args.spike_distance
        ):
            tracks[frame_id] = None
            accepted[frame_id] = 0
            sources[frame_id] = "temporal_spike_removed"
    result = dict(stable)
    result["tracks"] = tracks
    result["accepted"] = accepted
    result["source"] = sources
    return result


def remove_short_fragments(stable, args):
    tracks = list(stable["tracks"])
    accepted = list(stable["accepted"])
    sources = list(stable["source"])
    index = 0
    while index < len(tracks):
        if tracks[index] is None:
            index += 1
            continue
        start = index
        while index < len(tracks) and tracks[index] is not None:
            index += 1
        end = index
        fragment = tracks[start:end]
        length = end - start
        fragment_radius = max(v31.distance_xy(fragment[0], point) for point in fragment)
        isolated_static_fragment = (
            length <= args.isolated_static_max_length
            and fragment_radius <= args.isolated_static_radius
        )
        static_fragment = (
            length <= args.static_fragment_max_length
            and fragment_radius <= args.static_radius
        )
        if length < args.min_fragment_length or isolated_static_fragment or static_fragment:
            if isolated_static_fragment:
                reason = "isolated_static_removed"
            elif static_fragment:
                reason = "static_fragment_removed"
            else:
                reason = "short_fragment_removed"
            for frame_id in range(start, end):
                tracks[frame_id] = None
                accepted[frame_id] = 0
                sources[frame_id] = reason
    result = dict(stable)
    result["tracks"] = tracks
    result["accepted"] = accepted
    result["source"] = sources
    return result


def adjacent_track_stats(tracks):
    distances = []
    for previous, current in zip(tracks, tracks[1:]):
        if previous is not None and current is not None:
            distances.append(v31.distance_xy(previous, current))
    return {
        "pairs": len(distances),
        "jump80": sum(distance > 80 for distance in distances),
        "jump120": sum(distance > 120 for distance in distances),
        "jump180": sum(distance > 180 for distance in distances),
        "max_jump": max(distances, default=0.0),
    }


def interpolate_guarded(stable, max_gap, max_step):
    if max_gap <= 0:
        return stable
    tracks = list(stable["tracks"])
    scores = list(stable["scores"])
    accepted = list(stable["accepted"])
    sources = list(stable["source"])
    n = len(tracks)
    index = 0

    while index < n:
        if tracks[index] is not None:
            index += 1
            continue
        start = index
        while index < n and tracks[index] is None:
            index += 1
        end = index - 1
        gap = end - start + 1
        previous_index = start - 1
        next_index = index
        if gap > max_gap or previous_index < 0 or next_index >= n:
            continue
        previous = tracks[previous_index]
        following = tracks[next_index]
        if previous is None or following is None:
            continue
        if v31.distance_xy(previous, following) > max_step * (gap + 1) * 1.5:
            continue

        span = next_index - previous_index
        previous_score = scores[previous_index] or 0.0
        following_score = scores[next_index] or previous_score
        for frame_id in range(start, end + 1):
            ratio = (frame_id - previous_index) / float(span)
            tracks[frame_id] = (
                previous[0] * (1.0 - ratio) + following[0] * ratio,
                previous[1] * (1.0 - ratio) + following[1] * ratio,
            )
            scores[frame_id] = previous_score * (1.0 - ratio) + following_score * ratio
            accepted[frame_id] = 2
            sources[frame_id] = "guarded_interpolation"

    result = dict(stable)
    result["tracks"] = tracks
    result["scores"] = scores
    result["accepted"] = accepted
    result["source"] = sources
    return result


def write_outputs(args, stable):
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.video_out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.video_out_path, cv2.VideoWriter_fourcc(*args.codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {args.video_out_path}")

    output_path = Path(args.csv_out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace = deque(maxlen=args.trace)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame", "x", "y", "score", "raw_x", "raw_y", "raw_score",
                "predicted_x", "predicted_y", "innovation", "shape_quality",
                "motion_gate", "accepted", "source", "gap",
            ]
        )
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            point = stable["tracks"][frame_id]
            raw = stable["raw_tracks"][frame_id]
            predicted = stable["predicted_tracks"][frame_id]
            trace.append(point)
            writer.write(v31.draw_trace(frame, trace, raw if args.draw_raw else None))

            def pair(value):
                return ("", "") if value is None else (f"{value[0]:.2f}", f"{value[1]:.2f}")

            x, y = pair(point)
            raw_x, raw_y = pair(raw)
            predicted_x, predicted_y = pair(predicted)
            score = stable["scores"][frame_id]
            raw_score = stable["raw_scores"][frame_id]
            innovation = stable["innovations"][frame_id]
            quality = stable["shape_quality"][frame_id]
            motion_gate = stable["motion_gates"][frame_id]
            csv_writer.writerow(
                [
                    frame_id, x, y, "" if score is None else f"{score:.6f}",
                    raw_x, raw_y, "" if raw_score is None else f"{raw_score:.6f}",
                    predicted_x, predicted_y,
                    "" if innovation is None else f"{innovation:.3f}",
                    "" if quality is None else f"{quality:.4f}",
                    "" if motion_gate is None else f"{motion_gate:.2f}",
                    stable["accepted"][frame_id], stable["source"][frame_id], stable["gaps"][frame_id],
                ]
            )
            frame_id += 1

    writer.release()
    cap.release()
    stats = adjacent_track_stats(stable["tracks"])
    source_counts = {}
    for source in stable["source"]:
        source_counts[source] = source_counts.get(source, 0) + 1
    print(f"video_out={args.video_out_path}", flush=True)
    print(f"csv_out={args.csv_out_path}", flush=True)
    print(f"source_counts={source_counts}", flush=True)
    print(
        "adjacent_stats: pairs={}, jump>80={}, jump>120={}, jump>180={}, max_jump={:.2f}".format(
            stats["pairs"], stats["jump80"], stats["jump120"], stats["jump180"], stats["max_jump"]
        ),
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="V3.7 adaptive video post-processing for precise but occasionally weak heatmaps.")
    parser.add_argument(
        "--model_path",
        default="exps/lite_heatmap_v37_clean_hardneg_batch8_360x640/snapshots/"
        "best_before_error_audit_20260613_191025/model_best_f1.pt",
    )
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--video_out_path", required=True)
    parser.add_argument("--csv_out_path", required=True)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--input_height", type=int, default=360)
    parser.add_argument("--input_width", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.35, help="Candidate extraction floor.")
    parser.add_argument("--low_threshold", type=float, default=0.35)
    parser.add_argument("--high_threshold", type=float, default=0.88)
    parser.add_argument(
        "--seed_threshold",
        type=float,
        default=0.96,
        help="Stricter threshold used only to initialize or reinitialize a track.",
    )
    parser.add_argument("--peak_window", type=int, default=9)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--suppress_window", type=int, default=15)
    parser.add_argument("--min_weak_shape", type=float, default=0.05)
    parser.add_argument("--shape_weight", type=float, default=0.05)
    parser.add_argument("--hough_weight", type=float, default=0.00)
    parser.add_argument("--gate_min", type=float, default=60.0)
    parser.add_argument("--gate_max", type=float, default=220.0)
    parser.add_argument("--speed_window", type=int, default=5)
    parser.add_argument("--speed_gate_factor", type=float, default=2.2)
    parser.add_argument("--gate_margin", type=float, default=25.0)
    parser.add_argument("--strong_gate_factor", type=float, default=1.5)
    parser.add_argument("--weak_gate_factor", type=float, default=0.85)
    parser.add_argument("--motion_weight", type=float, default=0.20)
    parser.add_argument("--velocity_alpha", type=float, default=0.65)
    parser.add_argument("--weak_velocity_alpha", type=float, default=0.35)
    parser.add_argument("--turn_reset_angle", type=float, default=70.0)
    parser.add_argument("--extreme_turn_angle", type=float, default=120.0)
    parser.add_argument("--max_accel_ratio", type=float, default=3.0)
    parser.add_argument("--min_speed_for_accel_check", type=float, default=5.0)
    parser.add_argument("--seed_max_gap", type=int, default=3)
    parser.add_argument("--weak_reconnect_gap", type=int, default=4)
    parser.add_argument("--max_reconnect_gap", type=int, default=8)
    parser.add_argument("--fps_adaptive", action="store_true")
    parser.add_argument("--interp_gap", type=int, default=3)
    parser.add_argument("--min_fragment_length", type=int, default=3)
    parser.add_argument("--isolated_static_max_length", type=int, default=3)
    parser.add_argument("--isolated_static_radius", type=float, default=8.0)
    parser.add_argument("--static_fragment_max_length", type=int, default=8)
    parser.add_argument("--static_radius", type=float, default=5.0)
    parser.add_argument("--spike_return_radius", type=float, default=35.0)
    parser.add_argument("--spike_distance", type=float, default=80.0)
    parser.add_argument("--trace", type=int, default=12)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--draw_raw", action="store_true")
    parser.add_argument("--enable_hough_quality", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--print_interval", type=int, default=100)
    return parser


def main():
    args = build_parser().parse_args()
    if args.threshold > args.low_threshold:
        args.threshold = args.low_threshold
    rows, width, height, fps = v31.infer_candidate_rows(args)
    stable = stabilize_v37(rows, width, height, fps, args)
    stable = remove_temporal_spikes(stable, args)
    stable = interpolate_guarded(stable, args.interp_gap, args.gate_max)
    stable = remove_short_fragments(stable, args)
    write_outputs(args, stable)


if __name__ == "__main__":
    main()
