import argparse
import csv
import math
import time
from collections import deque

import cv2
import numpy as np
import torch

from model_v3 import BallTrackerNetV3


def make_input(frame, prev_frame, preprev_frame, input_width, input_height):
    img = cv2.resize(frame, (input_width, input_height))
    img_prev = cv2.resize(prev_frame, (input_width, input_height))
    img_preprev = cv2.resize(preprev_frame, (input_width, input_height))
    imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = np.rollaxis(imgs, 2, 0)
    return imgs


def refine_peak(heatmap, peak_x, peak_y, threshold, peak_window, scale_x, scale_y):
    peak_window = max(3, int(peak_window))
    if peak_window % 2 == 0:
        peak_window += 1
    radius = peak_window // 2
    x0 = max(0, peak_x - radius)
    x1 = min(heatmap.shape[1], peak_x + radius + 1)
    y0 = max(0, peak_y - radius)
    y1 = min(heatmap.shape[0], peak_y + radius + 1)

    crop = heatmap[y0:y1, x0:x1]
    weights = np.maximum(crop - threshold, 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return float(peak_x) * scale_x, float(peak_y) * scale_y

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    x_model = float(np.sum(weights * xs[None, :]) / weight_sum)
    y_model = float(np.sum(weights * ys[:, None]) / weight_sum)
    return x_model * scale_x, y_model * scale_y


def peak_shape_quality(heatmap, peak_x, peak_y, threshold, suppress_window):
    radius = max(3, int(suppress_window) // 2)
    x0 = max(0, peak_x - radius)
    x1 = min(heatmap.shape[1], peak_x + radius + 1)
    y0 = max(0, peak_y - radius)
    y1 = min(heatmap.shape[0], peak_y + radius + 1)
    crop = heatmap[y0:y1, x0:x1]
    if crop.size == 0:
        return {"area": 0.0, "radius": 0.0, "compactness": 0.0, "shape_quality": 0.0}

    binary = (crop >= threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if num_labels <= 1:
        return {"area": 0.0, "radius": 0.0, "compactness": 0.0, "shape_quality": 0.0}

    local_x = peak_x - x0
    local_y = peak_y - y0
    label = labels[local_y, local_x]
    if label <= 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    area = float(stats[label, cv2.CC_STAT_AREA])
    width = float(stats[label, cv2.CC_STAT_WIDTH])
    height = float(stats[label, cv2.CC_STAT_HEIGHT])
    radius_est = 0.5 * max(width, height)
    aspect = min(width, height) / max(width, height, 1.0)
    ideal_area = np.pi * max(radius_est, 1.0) * max(radius_est, 1.0)
    fill_ratio = min(area / max(ideal_area, 1.0), 1.5)
    compactness = max(0.0, min(1.0, aspect)) * max(0.0, min(1.0, fill_ratio))
    size_score = max(0.0, 1.0 - abs(radius_est - 4.0) / 9.0)
    shape_quality = 0.65 * compactness + 0.35 * size_score
    return {
        "area": area,
        "radius": radius_est,
        "compactness": float(compactness),
        "shape_quality": float(max(0.0, min(1.0, shape_quality))),
    }


def hough_circle_quality(heatmap, peak_x, peak_y, threshold, suppress_window):
    radius = max(8, int(suppress_window) // 2)
    x0 = max(0, peak_x - radius)
    x1 = min(heatmap.shape[1], peak_x + radius + 1)
    y0 = max(0, peak_y - radius)
    y1 = min(heatmap.shape[0], peak_y + radius + 1)
    crop = heatmap[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    norm = np.clip((crop - threshold) / max(1e-6, float(crop.max() - threshold)), 0.0, 1.0)
    img = (norm * 255).astype(np.uint8)
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=2,
        param1=40,
        param2=6,
        minRadius=2,
        maxRadius=max(3, radius),
    )
    if circles is None:
        return 0.0
    local_peak = np.asarray([peak_x - x0, peak_y - y0], dtype=np.float32)
    best = 0.0
    for circle in circles[0]:
        center = np.asarray([circle[0], circle[1]], dtype=np.float32)
        dist = float(np.linalg.norm(center - local_peak))
        radius_score = max(0.0, 1.0 - abs(float(circle[2]) - 4.0) / 8.0)
        center_score = max(0.0, 1.0 - dist / max(1.0, radius))
        best = max(best, 0.55 * center_score + 0.45 * radius_score)
    return float(min(1.0, best))


def topk_heatmap_candidates(heatmap, args, scale_x, scale_y):
    if heatmap.ndim != 2:
        heatmap = np.squeeze(heatmap)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    work = heatmap.copy()
    candidates = []
    suppress = max(3, int(args.suppress_window))
    if suppress % 2 == 0:
        suppress += 1
    radius = suppress // 2

    for _ in range(max(1, int(args.top_k))):
        _, max_value, _, max_loc = cv2.minMaxLoc(work)
        score = float(max_value)
        if score < args.threshold:
            break
        peak_x, peak_y = max_loc
        x, y = refine_peak(
            heatmap,
            peak_x,
            peak_y,
            args.threshold,
            args.peak_window,
            scale_x,
            scale_y,
        )
        shape = peak_shape_quality(heatmap, peak_x, peak_y, args.threshold, args.suppress_window)
        if getattr(args, "enable_hough_quality", False):
            shape["hough_quality"] = hough_circle_quality(heatmap, peak_x, peak_y, args.threshold, args.suppress_window)
        else:
            shape["hough_quality"] = 0.0
        candidates.append({"x": x, "y": y, "score": score, **shape})

        x0 = max(0, peak_x - radius)
        x1 = min(work.shape[1], peak_x + radius + 1)
        y0 = max(0, peak_y - radius)
        y1 = min(work.shape[0], peak_y + radius + 1)
        work[y0:y1, x0:x1] = -1.0

    return candidates


def run_batch(model, device, batch_inputs, width, height, args):
    tensor = torch.from_numpy(np.stack(batch_inputs, axis=0)).float().to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        heatmaps = torch.sigmoid(logits).detach().cpu().numpy()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    scale_x = width / float(args.input_width)
    scale_y = height / float(args.input_height)
    results = []
    for i in range(heatmaps.shape[0]):
        candidates = topk_heatmap_candidates(heatmaps[i, 0], args, scale_x, scale_y)
        results.append(candidates)
    return results, (t1 - t0) * 1000.0


def distance_xy(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def choose_candidate(candidates, predicted, max_step, hard_step, motion_weight):
    best = None
    best_value = -1e9
    for cand in candidates:
        point = (cand["x"], cand["y"])
        dist = distance_xy(point, predicted)
        if dist > hard_step:
            continue
        value = cand["score"] - motion_weight * min(dist / max(max_step, 1e-6), 4.0)
        if value > best_value:
            best = cand
            best_value = value
    return best


def stabilize_candidates(candidate_rows, width, height, fps, args):
    diag = math.hypot(width, height)
    base_step = args.max_step if args.max_step > 0 else diag * args.max_step_ratio
    if args.fps_adaptive:
        base_step *= 30.0 / max(float(fps), 1.0)
    max_step = max(args.min_step, base_step)
    hard_step = max_step * args.hard_step_factor

    tracks = []
    scores = []
    raw_tracks = []
    raw_scores = []
    accepted_flags = []
    gap_lengths = []

    last_pos = None
    last_out = None
    velocity = (0.0, 0.0)
    gap = 0

    for candidates in candidate_rows:
        raw = candidates[0] if candidates else None
        raw_tracks.append(None if raw is None else (raw["x"], raw["y"]))
        raw_scores.append(None if raw is None else raw["score"])

        chosen = None
        predicted = None
        if last_pos is None or gap > args.reinit_gap:
            chosen = raw
        elif candidates:
            dt = gap + 1
            predicted = (last_pos[0] + velocity[0] * dt, last_pos[1] + velocity[1] * dt)
            chosen = choose_candidate(candidates, predicted, max_step, hard_step, args.motion_weight)

        if chosen is None:
            tracks.append(None)
            scores.append(None if raw is None else raw["score"])
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
        accepted_flags.append(1)
        gap = 0
        gap_lengths.append(gap)
        last_pos = point
        last_out = out

    return {
        "tracks": tracks,
        "scores": scores,
        "raw_tracks": raw_tracks,
        "raw_scores": raw_scores,
        "accepted": accepted_flags,
        "source": ["accepted" if flag else "missing" for flag in accepted_flags],
        "gaps": gap_lengths,
        "max_step": max_step,
        "hard_step": hard_step,
    }


def interpolate_short_gaps(stable, max_gap):
    if max_gap <= 0:
        return stable

    tracks = list(stable["tracks"])
    scores = list(stable["scores"])
    accepted = list(stable["accepted"])
    source = list(stable["source"])
    n = len(tracks)
    i = 0
    while i < n:
        if tracks[i] is not None:
            i += 1
            continue
        start = i
        while i < n and tracks[i] is None:
            i += 1
        end = i - 1
        gap = end - start + 1
        prev_i = start - 1
        next_i = i
        if gap > max_gap or prev_i < 0 or next_i >= n:
            continue
        if tracks[prev_i] is None or tracks[next_i] is None:
            continue

        p0 = tracks[prev_i]
        p1 = tracks[next_i]
        s0 = scores[prev_i] if scores[prev_i] is not None else 0.0
        s1 = scores[next_i] if scores[next_i] is not None else s0
        span = next_i - prev_i
        for j in range(start, end + 1):
            ratio = (j - prev_i) / float(span)
            tracks[j] = (
                p0[0] * (1.0 - ratio) + p1[0] * ratio,
                p0[1] * (1.0 - ratio) + p1[1] * ratio,
            )
            scores[j] = s0 * (1.0 - ratio) + s1 * ratio
            accepted[j] = 2
            source[j] = "interpolated"

    out = dict(stable)
    out["tracks"] = tracks
    out["scores"] = scores
    out["accepted"] = accepted
    out["source"] = source
    return out


def infer_candidate_rows(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = BallTrackerNetV3(base_channels=args.base_channels)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = deque(maxlen=3)
    candidate_rows = []
    batch_inputs = []
    batch_indices = []
    infer_ms = []
    processed = 0
    t0_total = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if len(frames) < 3:
            candidate_rows.append([])
        else:
            batch_inputs.append(make_input(frames[-1], frames[-2], frames[-3], args.input_width, args.input_height))
            batch_indices.append(processed)
            if len(batch_inputs) == args.batch_size:
                results, ms = run_batch(model, device, batch_inputs, width, height, args)
                infer_ms.append(ms)
                for idx, result in zip(batch_indices, results):
                    while len(candidate_rows) <= idx:
                        candidate_rows.append([])
                    candidate_rows[idx] = result
                batch_inputs = []
                batch_indices = []
        processed += 1
        if processed % args.print_interval == 0:
            recent = infer_ms[-max(1, args.print_interval // max(1, args.batch_size)) :]
            avg_run_ms = float(np.mean(recent)) if recent else 0.0
            print(f"pass1 processed {processed}/{total}, avg_recent_run_ms={avg_run_ms:.2f}", flush=True)

    if batch_inputs:
        results, ms = run_batch(model, device, batch_inputs, width, height, args)
        infer_ms.append(ms)
        for idx, result in zip(batch_indices, results):
            while len(candidate_rows) <= idx:
                candidate_rows.append([])
            candidate_rows[idx] = result

    cap.release()
    while len(candidate_rows) < processed:
        candidate_rows.append([])

    wall_s = time.perf_counter() - t0_total
    avg_run_ms = float(np.mean(infer_ms)) if infer_ms else 0.0
    avg_frame_ms = avg_run_ms / max(1, args.batch_size)
    raw_detected = sum(1 for row in candidate_rows if row)
    print(
        f"pass1 done: frames={processed}, raw_detected={raw_detected}, "
        f"avg_run_ms={avg_run_ms:.2f}, approx_model_ms_per_frame={avg_frame_ms:.2f}, wall_sec={wall_s:.2f}",
        flush=True,
    )
    return candidate_rows, width, height, fps


def draw_trace(frame, trace_points, raw_point=None):
    if raw_point is not None:
        cv2.circle(frame, (int(raw_point[0]), int(raw_point[1])), radius=5, color=(255, 180, 0), thickness=1)
    for age, point in enumerate(reversed(trace_points)):
        if point is None:
            continue
        x, y = point
        thickness = max(2, 10 - age)
        cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=thickness)
    return frame


def track_stats(tracks):
    valid = [p for p in tracks if p is not None]
    jumps = []
    prev = None
    for point in tracks:
        if point is None:
            prev = None
            continue
        if prev is not None:
            jumps.append(distance_xy(point, prev))
        prev = point
    return {
        "valid": len(valid),
        "jump80": sum(j > 80 for j in jumps),
        "jump120": sum(j > 120 for j in jumps),
        "jump240": sum(j > 240 for j in jumps),
        "max_jump": max(jumps) if jumps else 0.0,
    }


def write_outputs(args, stable):
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        args.video_out_path,
        cv2.VideoWriter_fourcc(*args.codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {args.video_out_path}")

    csv_file = open(args.csv_out_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "x", "y", "score", "raw_x", "raw_y", "raw_score", "accepted", "source", "gap"])

    trace_points = deque(maxlen=args.trace)
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            point = stable["tracks"][frame_id]
            score = stable["scores"][frame_id]
            raw = stable["raw_tracks"][frame_id]
            raw_score = stable["raw_scores"][frame_id]
            trace_points.append(point)
            writer.write(draw_trace(frame, trace_points, raw if args.draw_raw else None))

            if point is None:
                x_out, y_out, score_out = "", "", "" if score is None else f"{score:.6f}"
            else:
                x_out, y_out, score_out = f"{point[0]:.2f}", f"{point[1]:.2f}", f"{score:.6f}"
            if raw is None:
                raw_x, raw_y, raw_s = "", "", ""
            else:
                raw_x, raw_y, raw_s = f"{raw[0]:.2f}", f"{raw[1]:.2f}", f"{raw_score:.6f}"

            csv_writer.writerow(
                [
                    frame_id,
                    x_out,
                    y_out,
                    score_out,
                    raw_x,
                    raw_y,
                    raw_s,
                    stable["accepted"][frame_id],
                    stable["source"][frame_id],
                    stable["gaps"][frame_id],
                ]
            )
            frame_id += 1
    finally:
        csv_file.close()
        writer.release()
        cap.release()

    stats = track_stats(stable["tracks"])
    print(f"video_out={args.video_out_path}", flush=True)
    print(f"csv_out={args.csv_out_path}", flush=True)
    print(
        "stable_stats: valid={}, jump>80={}, jump>120={}, jump>240={}, max_jump={:.2f}, max_step={:.2f}, hard_step={:.2f}".format(
            stats["valid"],
            stats["jump80"],
            stats["jump120"],
            stats["jump240"],
            stats["max_jump"],
            stable["max_step"],
            stable["hard_step"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackNet V3.1 inference with top-k association and jump filtering.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--csv_out_path", type=str, required=True)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--input_height", type=int, default=180)
    parser.add_argument("--input_width", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--peak_window", type=int, default=15)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--suppress_window", type=int, default=21)
    parser.add_argument("--max_step", type=float, default=240.0)
    parser.add_argument("--max_step_ratio", type=float, default=0.08)
    parser.add_argument("--min_step", type=float, default=45.0)
    parser.add_argument("--hard_step_factor", type=float, default=6.0)
    parser.add_argument("--motion_weight", type=float, default=0.12)
    parser.add_argument("--velocity_alpha", type=float, default=0.60)
    parser.add_argument("--smooth_alpha", type=float, default=1.00)
    parser.add_argument("--reinit_gap", type=int, default=6)
    parser.add_argument("--fps_adaptive", action="store_true")
    parser.add_argument("--trace", type=int, default=12)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--interp_gap", type=int, default=4)
    parser.add_argument("--draw_raw", action="store_true")
    parser.add_argument("--enable_hough_quality", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--print_interval", type=int, default=100)
    parsed = parser.parse_args()

    rows, video_width, video_height, video_fps = infer_candidate_rows(parsed)
    stable_rows = stabilize_candidates(rows, video_width, video_height, video_fps, parsed)
    stable_rows = interpolate_short_gaps(stable_rows, parsed.interp_gap)
    write_outputs(parsed, stable_rows)
