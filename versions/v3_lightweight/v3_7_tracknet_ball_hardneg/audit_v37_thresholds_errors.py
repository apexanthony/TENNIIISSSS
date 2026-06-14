import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from datasets_v37_hardneg import COURT_SEGMENTS, TrackNetDatasetV37HardNeg
from general_v3 import postprocess_heatmap
from model_v3 import BallTrackerNetV3


KEY_COLUMNS = ["game", "clip", "frame_name"]


def build_thresholds():
    values = list(np.arange(0.50, 0.80, 0.01))
    values += list(np.arange(0.80, 0.951, 0.005))
    values += list(np.arange(0.96, 0.981, 0.01))
    return sorted({round(float(value), 3) for value in values})


def load_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        return state["model_state"]
    return state


def parse_boxes(value):
    return TrackNetDatasetV37HardNeg.parse_boxes(value)


def parse_court_points(row):
    points = {}
    if row is None:
        return points
    for index in range(1, 15):
        point = TrackNetDatasetV37HardNeg.parse_point(row.get(f"court_{index}", ""))
        if point is not None:
            points[index] = point
    return points


def point_segment_distance(point, start, end):
    point = np.asarray(point, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-9:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * segment)))


def context_at_prediction(pred, mapped_row, image):
    if pred is None:
        return "none"
    x, y = pred
    if mapped_row is not None:
        for xmin, ymin, xmax, ymax in parse_boxes(mapped_row.get("player_boxes", "")):
            margin = 8.0
            if xmin - margin <= x <= xmax + margin and ymin - margin <= y <= ymax + margin:
                lower_y = ymin + 0.58 * (ymax - ymin)
                return "shoe_or_lower_player" if y >= lower_y else "player_body"

        points = parse_court_points(mapped_row)
        for start_index, end_index in COURT_SEGMENTS:
            if start_index in points and end_index in points:
                if point_segment_distance(pred, points[start_index], points[end_index]) <= 8.0:
                    return "court_line"

    if image is not None:
        ix = int(round(x))
        iy = int(round(y))
        h, w = image.shape[:2]
        x0, x1 = max(0, ix - 3), min(w, ix + 4)
        y0, y1 = max(0, iy - 3), min(h, iy + 4)
        if x0 < x1 and y0 < y1:
            hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            bright_ratio = float(np.mean((hsv[:, :, 2] >= 190) & (hsv[:, :, 1] <= 90)))
            if bright_ratio >= 0.35:
                return "bright_or_white_line"
    return "other"


def add_temporal_label_residual(records):
    for record in records:
        record["label_temporal_residual"] = None
        record["trajectory_expected_x"] = None
        record["trajectory_expected_y"] = None

    by_clip = {}
    for index, record in enumerate(records):
        by_clip.setdefault(record["clip_key"], []).append((index, record))

    for rows in by_clip.values():
        rows.sort(key=lambda item: item[1]["frame_id"])
        for position in range(1, len(rows) - 1):
            index, current = rows[position]
            previous = rows[position - 1][1]
            following = rows[position + 1][1]
            if not (current["visible"] and previous["visible"] and following["visible"]):
                continue
            t0, t1, t2 = previous["frame_id"], current["frame_id"], following["frame_id"]
            if t2 <= t0 or t1 - t0 > 2 or t2 - t1 > 2:
                continue
            ratio = float(t1 - t0) / float(t2 - t0)
            expected_x = previous["gt_x"] + ratio * (following["gt_x"] - previous["gt_x"])
            expected_y = previous["gt_y"] + ratio * (following["gt_y"] - previous["gt_y"])
            residual = math.hypot(current["gt_x"] - expected_x, current["gt_y"] - expected_y)
            records[index]["label_temporal_residual"] = residual
            records[index]["trajectory_expected_x"] = expected_x
            records[index]["trajectory_expected_y"] = expected_y


def is_suspected_label_shift(record):
    if record["pred_x"] is None or record["label_temporal_residual"] is None:
        return False
    expected = (record["trajectory_expected_x"], record["trajectory_expected_y"])
    pred_to_expected = math.hypot(record["pred_x"] - expected[0], record["pred_y"] - expected[1])
    return (
        record["label_temporal_residual"] >= 15.0
        and pred_to_expected <= 10.0
        and record["distance"] - pred_to_expected >= 5.0
    )


def classify_record(record, context, low_threshold=0.50):
    pred_exists = record["pred_x"] is not None
    if record["visible"]:
        if not pred_exists:
            return "weak_response_filtered" if record["peak_confidence"] >= low_threshold else "true_miss"
        if record["distance"] < 5.0:
            return "correct_within_5px"
        if is_suspected_label_shift(record):
            return "suspected_label_shift"
        if record["distance"] < 15.0:
            return "localization_5_to_15px"
        return {
            "shoe_or_lower_player": "shoe_or_lower_player_false_positive",
            "player_body": "player_body_false_positive",
            "court_line": "court_line_false_positive",
            "bright_or_white_line": "bright_or_white_line_false_positive",
        }.get(context, "other_far_prediction")

    if not pred_exists:
        return "correct_no_ball"
    return {
        "shoe_or_lower_player": "shoe_or_lower_player_false_positive",
        "player_body": "player_body_false_positive",
        "court_line": "court_line_false_positive",
        "bright_or_white_line": "bright_or_white_line_false_positive",
    }.get(context, "other_no_ball_false_positive")


def metric_rows(records_by_threshold, thresholds, radii):
    rows = []
    eps = 1e-15
    for radius in radii:
        for threshold in thresholds:
            tp = fp = tn = fn = 0
            visible_count = 0
            for record in records_by_threshold[threshold]:
                if record["visible"]:
                    visible_count += 1
                if record["pred"] is None:
                    if record["visible"]:
                        fn += 1
                    else:
                        tn += 1
                elif record["visible"] and record["distance"] < radius:
                    tp += 1
                else:
                    fp += 1
            precision = tp / (tp + fp + eps)
            recall = tp / (visible_count + eps)
            f1 = 2.0 * precision * recall / (precision + recall + eps)
            rows.append(
                {
                    "radius": radius,
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }
            )
    return rows


def save_plots(metric_frame, selected_records, output_dir):
    plt.figure(figsize=(8, 6))
    for radius, group in metric_frame.groupby("radius"):
        ordered = group.sort_values("recall")
        plt.plot(ordered["recall"], ordered["precision"], marker=".", label=f"{int(radius)} px")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("V3.7 validation PR curves")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=180)
    plt.close()

    distances = [record["distance"] for record in selected_records if record["distance"] is not None]
    plt.figure(figsize=(9, 5))
    plt.hist(distances, bins=np.arange(0, 101, 2), color="#3377aa", edgecolor="white")
    for boundary, color in [(5, "#cc3333"), (8, "#dd8800"), (10, "#44aa55"), (15, "#8844aa")]:
        plt.axvline(boundary, color=color, linestyle="--", label=f"{boundary}px")
    plt.xlabel("Prediction-to-label distance (px)")
    plt.ylabel("Frames")
    plt.title("Prediction distance error distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "distance_error_histogram.png", dpi=180)
    plt.close()


def draw_visual(record, image, mapped_row, output_path):
    if image is None:
        return
    canvas = image.copy()
    if mapped_row is not None:
        for xmin, ymin, xmax, ymax in parse_boxes(mapped_row.get("player_boxes", "")):
            cv2.rectangle(canvas, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 180, 0), 2)
        points = parse_court_points(mapped_row)
        for start_index, end_index in COURT_SEGMENTS:
            if start_index in points and end_index in points:
                start = tuple(int(round(v)) for v in points[start_index])
                end = tuple(int(round(v)) for v in points[end_index])
                cv2.line(canvas, start, end, (0, 220, 220), 1)
    if record["visible"]:
        cv2.drawMarker(
            canvas,
            (int(round(record["gt_x"])), int(round(record["gt_y"]))),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            18,
            2,
        )
    if record["pred_x"] is not None:
        cv2.circle(canvas, (int(round(record["pred_x"])), int(round(record["pred_y"]))), 7, (0, 0, 255), 2)
    text = f"{record['category']} conf={record['peak_confidence']:.3f}"
    if record["distance"] is not None:
        text += f" dist={record['distance']:.1f}px"
    cv2.putText(canvas, text, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def main():
    parser = argparse.ArgumentParser(description="Audit V3.7 thresholds, distance sensitivity, and validation errors.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid-csv", default="datasets/tracknet_v37_clip_split/valid.csv")
    parser.add_argument(
        "--mapped-csv",
        default="datasets/tennis_all_v4i_mapped/annotations_hardneg_cleaned_strict.csv",
    )
    parser.add_argument("--output-dir", default="exps/v37_best_error_audit")
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--label-height", type=int, default=720)
    parser.add_argument("--label-width", type=int, default=1280)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--peak-window", type=int, default=9)
    parser.add_argument("--radii", default="5,8,10,15")
    parser.add_argument("--device", default=None)
    parser.add_argument("--visuals-per-category", type=int, default=20)
    parser.add_argument("--print-interval", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = build_thresholds()
    radii = [float(value.strip()) for value in args.radii.split(",") if value.strip()]

    dataset = TrackNetDatasetV37HardNeg(
        args.valid_csv,
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=5,
        heatmap_sigma=2.0,
        mapped_csv=args.mapped_csv,
        augment=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = BallTrackerNetV3(base_channels=args.base_channels).to(device)
    model.load_state_dict(load_state(args.checkpoint, device), strict=True)
    model.eval()

    scale_x = float(args.label_width) / float(args.input_width)
    scale_y = float(args.label_height) / float(args.input_height)
    records_by_threshold = {threshold: [] for threshold in thresholds}
    base_records = []
    manifest = dataset.data.reset_index(drop=True)
    sample_index = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            logits = model(batch["input"].float().to(device))
            heatmaps = torch.sigmoid(logits).cpu().numpy()
            for item_index in range(len(heatmaps)):
                row = manifest.iloc[sample_index]
                heatmap = heatmaps[item_index, 0]
                peak_confidence = float(np.max(heatmap))
                visible = int(batch["visibility"][item_index]) != 0
                gt_x = float(batch["x"][item_index])
                gt_y = float(batch["y"][item_index])
                base = {
                    "sample_index": sample_index,
                    "game": str(row["game"]),
                    "clip": str(row["clip"]),
                    "clip_key": str(row["clip_key"]),
                    "frame_name": str(row["frame_name"]),
                    "frame_id": int(row["frame_id"]),
                    "path1": str(row["path1"]),
                    "visible": visible,
                    "gt_x": gt_x,
                    "gt_y": gt_y,
                    "peak_confidence": peak_confidence,
                }
                base_records.append(base)
                for threshold in thresholds:
                    pred_x, pred_y = postprocess_heatmap(
                        heatmap,
                        threshold=threshold,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        peak_window=args.peak_window,
                    )
                    pred = None if pred_x is None else (float(pred_x), float(pred_y))
                    distance = None
                    if pred is not None and visible:
                        distance = float(math.hypot(pred[0] - gt_x, pred[1] - gt_y))
                    records_by_threshold[threshold].append(
                        {"pred": pred, "visible": visible, "distance": distance}
                    )
                sample_index += 1
            if args.print_interval > 0 and (batch_index % args.print_interval == 0 or batch_index + 1 == len(loader)):
                print(f"audit inference [{batch_index + 1}|{len(loader)}]", flush=True)

    metrics = metric_rows(records_by_threshold, thresholds, radii)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output_dir / "threshold_radius_scan.csv", index=False)
    best_rows = []
    for radius in radii:
        group = metric_frame[metric_frame["radius"] == radius]
        best_rows.append(group.sort_values(["f1", "recall"], ascending=False).iloc[0].to_dict())
    best_frame = pd.DataFrame(best_rows)
    best_frame.to_csv(output_dir / "best_by_radius.csv", index=False)

    selected_threshold = float(best_frame.loc[best_frame["radius"] == 5.0, "threshold"].iloc[0])
    selected_predictions = records_by_threshold[selected_threshold]
    selected_records = []
    for base, prediction in zip(base_records, selected_predictions):
        record = dict(base)
        pred = prediction["pred"]
        record["pred_x"] = None if pred is None else pred[0]
        record["pred_y"] = None if pred is None else pred[1]
        record["distance"] = prediction["distance"]
        selected_records.append(record)

    add_temporal_label_residual(selected_records)
    mapped_frame = pd.read_csv(args.mapped_csv)
    mapped = {
        tuple(str(row[column]) for column in KEY_COLUMNS): row
        for _, row in mapped_frame.iterrows()
    }
    tracknet_root = Path("datasets/trackNet")
    visual_counts = Counter()
    for record in selected_records:
        key = (record["game"], record["clip"], record["frame_name"])
        mapped_row = mapped.get(key)
        image = cv2.imread(str(tracknet_root / record["path1"]))
        pred = None if record["pred_x"] is None else (record["pred_x"], record["pred_y"])
        context = context_at_prediction(pred, mapped_row, image)
        record["prediction_context"] = context
        record["category"] = classify_record(record, context)
        category = record["category"]
        if category not in {"correct_within_5px", "correct_no_ball"} and visual_counts[category] < args.visuals_per_category:
            visual_counts[category] += 1
            filename = f"{record['game']}_{record['clip']}_{record['frame_id']:04d}.jpg"
            draw_visual(record, image, mapped_row, output_dir / "error_visuals" / category / filename)

    pd.DataFrame(selected_records).to_csv(output_dir / "selected_threshold_predictions.csv", index=False)
    counts = Counter(record["category"] for record in selected_records)
    summary_rows = [{"category": key, "count": value} for key, value in counts.most_common()]
    pd.DataFrame(summary_rows).to_csv(output_dir / "error_summary.csv", index=False)
    save_plots(metric_frame, selected_records, output_dir)

    summary = {
        "checkpoint": str(args.checkpoint),
        "validation_frames": len(selected_records),
        "threshold_count": len(thresholds),
        "selected_5px_threshold": selected_threshold,
        "best_by_radius": best_rows,
        "error_counts": dict(counts),
        "notes": {
            "suspected_label_shift": "Heuristic only: label breaks local trajectory while prediction follows it; manual review required.",
            "true_miss": "Visible label and peak confidence below 0.50.",
            "weak_response_filtered": "Visible label with peak confidence >=0.50 but below the selected threshold.",
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(best_frame.to_string(index=False), flush=True)
    print(json.dumps(summary["error_counts"], ensure_ascii=False, indent=2), flush=True)
    print(f"output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
