import argparse
import csv
import time
from collections import deque

import cv2
import numpy as np

from general_v3 import postprocess_heatmap


def make_input(frame, prev_frame, preprev_frame, input_width, input_height):
    img = cv2.resize(frame, (input_width, input_height))
    img_prev = cv2.resize(prev_frame, (input_width, input_height))
    img_preprev = cv2.resize(preprev_frame, (input_width, input_height))
    imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = np.rollaxis(imgs, 2, 0)
    return imgs[None, :, :, :]


def draw_trace(frame, trace_points):
    for age, point in enumerate(reversed(trace_points)):
        if point is None:
            continue
        x, y = point
        thickness = max(2, 10 - age)
        cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=thickness)
    return frame


def infer_tracks(args):
    net = cv2.dnn.readNetFromONNX(args.onnx_path)
    if args.backend == "opencv":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    if args.target == "cpu":
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    elif args.target == "cuda":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(args.video_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = deque(maxlen=3)
    tracks = []
    scores = []
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
            inp = make_input(frames[-1], frames[-2], frames[-3], args.input_width, args.input_height)
            t0 = time.perf_counter()
            net.setInput(inp)
            heatmap = net.forward()[0, 0]
            t1 = time.perf_counter()
            infer_ms.append((t1 - t0) * 1000.0)

            score = float(np.max(heatmap))
            x_pred, y_pred = postprocess_heatmap(
                heatmap,
                threshold=args.threshold,
                scale_x=width / float(args.input_width),
                scale_y=height / float(args.input_height),
                peak_window=args.peak_window,
            )
            tracks.append((x_pred, y_pred))
            scores.append(score)

        processed += 1
        if processed % args.print_interval == 0:
            recent = infer_ms[-args.print_interval :]
            avg_ms = float(np.mean(recent)) if recent else 0.0
            print("pass1 processed {}/{}, avg_recent_ms={:.2f}".format(processed, total, avg_ms), flush=True)

    cap.release()
    detected = sum(1 for x, _ in tracks if x is not None)
    wall_s = time.perf_counter() - t0_total
    avg_ms = float(np.mean(infer_ms)) if infer_ms else 0.0
    print(
        "pass1 done: frames={}, detected={}, avg_model_ms_per_frame={:.2f}, wall_sec={:.2f}".format(
            processed, detected, avg_ms, wall_s
        ),
        flush=True,
    )
    return tracks, scores


def write_outputs(args, tracks, scores):
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(args.video_path))

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
        raise RuntimeError("Cannot open video writer: {}".format(args.video_out_path))

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
                csv_writer.writerow([frame_id, "", "", "" if score is None else "{:.6f}".format(score)])
            else:
                csv_writer.writerow([frame_id, "{:.2f}".format(x), "{:.2f}".format(y), "{:.6f}".format(score)])
            frame_id += 1
    finally:
        csv_file.close()
        writer.release()
        cap.release()

    print("video_out={}".format(args.video_out_path), flush=True)
    print("csv_out={}".format(args.csv_out_path), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TrackNet V3 ONNX inference on video with OpenCV DNN.")
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--csv_out_path", type=str, required=True)
    parser.add_argument("--input_height", type=int, default=270)
    parser.add_argument("--input_width", type=int, default=480)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--peak_window", type=int, default=15)
    parser.add_argument("--trace", type=int, default=7)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--backend", choices=["opencv"], default="opencv")
    parser.add_argument("--target", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--print_interval", type=int, default=100)
    parsed = parser.parse_args()

    track_points, confidence_scores = infer_tracks(parsed)
    write_outputs(parsed, track_points, confidence_scores)
