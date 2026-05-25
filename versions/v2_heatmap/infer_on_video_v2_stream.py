import argparse
import csv
import time
from collections import deque

import cv2
import numpy as np
import torch

from general_v2 import postprocess_heatmap
from model_v2 import BallTrackerNetLite


MODEL_HEIGHT = 360
MODEL_WIDTH = 640


def make_input(frame, prev_frame, preprev_frame):
    img = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
    img_prev = cv2.resize(prev_frame, (MODEL_WIDTH, MODEL_HEIGHT))
    img_preprev = cv2.resize(preprev_frame, (MODEL_WIDTH, MODEL_HEIGHT))
    imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = np.rollaxis(imgs, 2, 0)
    return np.expand_dims(imgs, axis=0)


def draw_trace(frame, trace_points):
    for age, point in enumerate(reversed(trace_points)):
        if point is None:
            continue
        x, y = point
        thickness = max(2, 10 - age)
        cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=thickness)
    return frame


def infer_stream(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = BallTrackerNetLite(base_channels=args.base_channels)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(args.video_out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {args.video_out_path}")

    csv_file = open(args.csv_out_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "x", "y", "score"])

    frames = deque(maxlen=3)
    trace_points = deque(maxlen=args.trace)
    processed = 0
    detected = 0
    infer_ms = []
    total_t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frames.append(frame)
            point = None
            score = 0.0

            if len(frames) < 3:
                csv_writer.writerow([processed, "", "", ""])
            else:
                inp = make_input(frames[-1], frames[-2], frames[-3])
                tensor = torch.from_numpy(inp).float().to(device)

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = model(tensor)
                    heatmap_tensor = torch.sigmoid(logits)[0, 0]
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                heatmap = heatmap_tensor.detach().cpu().numpy()
                score = float(np.max(heatmap))
                x_pred, y_pred = postprocess_heatmap(
                    heatmap,
                    threshold=args.threshold,
                    scale_x=width / float(MODEL_WIDTH),
                    scale_y=height / float(MODEL_HEIGHT),
                    peak_window=args.peak_window,
                )

                infer_ms.append((t1 - t0) * 1000.0)
                if x_pred is not None:
                    detected += 1
                    point = (x_pred, y_pred)
                    csv_writer.writerow([processed, f"{x_pred:.2f}", f"{y_pred:.2f}", f"{score:.6f}"])
                else:
                    csv_writer.writerow([processed, "", "", f"{score:.6f}"])

            trace_points.append(point)
            writer.write(draw_trace(frame, trace_points))
            processed += 1

            if processed % args.print_interval == 0:
                avg_ms = float(np.mean(infer_ms[-args.print_interval:])) if infer_ms else 0.0
                print(
                    f"processed {processed}/{total}, detected={detected}, "
                    f"avg_infer_ms_recent={avg_ms:.2f}",
                    flush=True,
                )
    finally:
        csv_file.close()
        writer.release()
        cap.release()

    total_s = time.perf_counter() - total_t0
    avg_ms = float(np.mean(infer_ms)) if infer_ms else 0.0
    print(f"done: frames={processed}, detected={detected}, detection_rate={detected / max(1, processed - 2):.4f}")
    print(f"avg_model_infer_ms={avg_ms:.2f}, wall_time_sec={total_s:.2f}")
    print(f"video_out={args.video_out_path}")
    print(f"csv_out={args.csv_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream TrackNet Lite V2 inference on video.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--csv_out_path", type=str, required=True)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--peak_window", type=int, default=15)
    parser.add_argument("--trace", type=int, default=7)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--print_interval", type=int, default=50)
    infer_stream(parser.parse_args())
