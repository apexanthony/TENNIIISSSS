import argparse
import csv
import time
from collections import deque

import cv2
import numpy as np
import torch

from general_v3 import postprocess_heatmap
from model_v3 import BallTrackerNetV3


def make_input(frame, prev_frame, preprev_frame, input_width, input_height):
    img = cv2.resize(frame, (input_width, input_height))
    img_prev = cv2.resize(prev_frame, (input_width, input_height))
    img_preprev = cv2.resize(preprev_frame, (input_width, input_height))
    imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = np.rollaxis(imgs, 2, 0)
    return imgs


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

    results = []
    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i, 0]
        score = float(np.max(heatmap))
        x_pred, y_pred = postprocess_heatmap(
            heatmap,
            threshold=args.threshold,
            scale_x=width / float(args.input_width),
            scale_y=height / float(args.input_height),
            peak_window=args.peak_window,
        )
        results.append((x_pred, y_pred, score))
    return results, (t1 - t0) * 1000.0


def draw_trace(frame, trace_points):
    for age, point in enumerate(reversed(trace_points)):
        if point is None:
            continue
        x, y = point
        thickness = max(2, 10 - age)
        cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=thickness)
    return frame


def infer_tracks(args):
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
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = deque(maxlen=3)
    tracks = []
    scores = []
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
            tracks.append((None, None))
            scores.append(None)
        else:
            batch_inputs.append(make_input(frames[-1], frames[-2], frames[-3], args.input_width, args.input_height))
            batch_indices.append(processed)
            if len(batch_inputs) == args.batch_size:
                results, ms = run_batch(model, device, batch_inputs, width, height, args)
                infer_ms.append(ms)
                for idx, result in zip(batch_indices, results):
                    while len(tracks) <= idx:
                        tracks.append((None, None))
                        scores.append(None)
                    tracks[idx] = (result[0], result[1])
                    scores[idx] = result[2]
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
            while len(tracks) <= idx:
                tracks.append((None, None))
                scores.append(None)
            tracks[idx] = (result[0], result[1])
            scores[idx] = result[2]

    cap.release()
    while len(tracks) < processed:
        tracks.append((None, None))
        scores.append(None)

    detected = sum(1 for x, _ in tracks if x is not None)
    wall_s = time.perf_counter() - t0_total
    avg_run_ms = float(np.mean(infer_ms)) if infer_ms else 0.0
    avg_frame_ms = avg_run_ms / max(1, args.batch_size)
    print(
        f"pass1 done: frames={processed}, detected={detected}, "
        f"avg_run_ms={avg_run_ms:.2f}, approx_model_ms_per_frame={avg_frame_ms:.2f}, wall_sec={wall_s:.2f}",
        flush=True,
    )
    return tracks, scores


def write_outputs(args, tracks, scores):
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
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
    csv_writer.writerow(["frame", "x", "y", "score"])

    trace_points = deque(maxlen=args.trace)
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            x, y = tracks[frame_id]
            score = scores[frame_id]
            point = None if x is None else (x, y)
            trace_points.append(point)
            writer.write(draw_trace(frame, trace_points))

            if x is None:
                csv_writer.writerow([frame_id, "", "", "" if score is None else f"{score:.6f}"])
            else:
                csv_writer.writerow([frame_id, f"{x:.2f}", f"{y:.2f}", f"{score:.6f}"])
            frame_id += 1
    finally:
        csv_file.close()
        writer.release()
        cap.release()

    print(f"video_out={args.video_out_path}", flush=True)
    print(f"csv_out={args.csv_out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch TrackNet V3 inference on video.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--csv_out_path", type=str, required=True)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--input_height", type=int, default=180)
    parser.add_argument("--input_width", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--peak_window", type=int, default=15)
    parser.add_argument("--trace", type=int, default=7)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--print_interval", type=int, default=100)
    parsed = parser.parse_args()

    track_points, confidence_scores = infer_tracks(parsed)
    write_outputs(parsed, track_points, confidence_scores)
