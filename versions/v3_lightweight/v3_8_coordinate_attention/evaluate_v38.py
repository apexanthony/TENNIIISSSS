import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset_v38 import TrackNetDatasetV38
from model_v38 import BallTrackerNetV38
from train_v38_ablation import VARIANTS, metrics_from_counts, new_counts


def threshold_grid(start, stop, step):
    values = []
    current = float(start)
    while current <= float(stop) + 1e-9:
        values.append(round(current, 6))
        current += float(step)
    return values


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    variant_name = config["variant"]
    variant = VARIANTS[variant_name]
    model = BallTrackerNetV38(
        base_channels=int(config["base_channels"]),
        use_ca=variant["use_ca"],
        ca_reduction=int(config["ca_reduction"]),
        ca_min_channels=int(config["ca_min_channels"]),
        initialization_seed=int(config["seed"]),
    )
    state = torch.load(run_dir / "model_best.pt", map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config


def make_dataset(csv_path, config):
    return TrackNetDatasetV38(
        csv_path,
        input_height=int(config["input_height"]),
        input_width=int(config["input_width"]),
        heatmap_radius=int(config["heatmap_radius"]),
        heatmap_sigma=float(config["heatmap_sigma"]),
        tracknet_root=config["tracknet_root"],
        mapped_csv=config["hardneg_mapped_csv"],
        aux_mapped_csv=config["aux_mapped_csv"],
        aux_heatmap_radius=int(config["aux_heatmap_radius"]),
        aux_heatmap_sigma=float(config["aux_heatmap_sigma"]),
        augment=False,
        rgb_input=True,
    )


def refine_peak(record, threshold):
    crop = record["peak_crop"]
    weights = np.maximum(crop - threshold, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return float(record["peak_x"]), float(record["peak_y"])
    xs = np.arange(record["crop_x0"], record["crop_x0"] + crop.shape[1], dtype=np.float32)
    ys = np.arange(record["crop_y0"], record["crop_y0"] + crop.shape[0], dtype=np.float32)
    return (
        float((weights * xs[None, :]).sum() / total),
        float((weights * ys[:, None]).sum() / total),
    )


def collect_predictions(model, dataset, device, batch_size, num_workers, peak_window, max_samples=0):
    if max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    records = []
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            inputs = batch["input"].float().to(device, non_blocking=True)
            probabilities = torch.sigmoid(model(inputs)).cpu().numpy()
            for index in range(len(probabilities)):
                heatmap = probabilities[index, 0]
                flat_index = int(np.argmax(heatmap))
                peak_y, peak_x = np.unravel_index(flat_index, heatmap.shape)
                window = max(3, int(peak_window))
                if window % 2 == 0:
                    window += 1
                radius = window // 2
                x0, x1 = max(0, peak_x - radius), min(heatmap.shape[1], peak_x + radius + 1)
                y0, y1 = max(0, peak_y - radius), min(heatmap.shape[0], peak_y + radius + 1)
                records.append(
                    {
                        "clip_key": str(batch["clip_key"][index]),
                        "frame_id": int(batch["frame_id"][index]),
                        "visibility": int(batch["visibility"][index]),
                        "gt_x": float(batch["x"][index]) * heatmap.shape[1] / float(batch["orig_width"][index]),
                        "gt_y": float(batch["y"][index]) * heatmap.shape[0] / float(batch["orig_height"][index]),
                        "peak_score": float(heatmap[peak_y, peak_x]),
                        "peak_x": int(peak_x),
                        "peak_y": int(peak_y),
                        "peak_crop": heatmap[y0:y1, x0:x1].copy(),
                        "crop_x0": int(x0),
                        "crop_y0": int(y0),
                    }
                )
            if step % 100 == 0:
                print(f"inference batch={step + 1}/{len(loader)}", flush=True)
    return records


def classify(record, threshold, match_radius):
    visible = record["visibility"] != 0
    if record["peak_score"] < threshold:
        return "missed_visible" if visible else "tn", None, None, None
    pred_x, pred_y = refine_peak(record, threshold)
    if not visible:
        return "fp_background", pred_x, pred_y, None
    distance = float(np.hypot(pred_x - record["gt_x"], pred_y - record["gt_y"]))
    return ("tp" if distance <= match_radius else "wrong_localization"), pred_x, pred_y, distance


def score(records, threshold, match_radius):
    overall = new_counts()
    grouped = defaultdict(new_counts)
    for record in records:
        outcome, _x, _y, _distance = classify(record, threshold, match_radius)
        overall[outcome] += 1
        grouped[record["clip_key"].split("/", 1)[0]][outcome] += 1
    return metrics_from_counts(overall), {key: metrics_from_counts(value) for key, value in grouped.items()}


def write_predictions(path, records, threshold, match_radius):
    fields = [
        "clip_key", "frame_id", "visibility", "gt_x", "gt_y", "peak_score",
        "pred_x", "pred_y", "distance", "outcome", "threshold", "match_radius",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            outcome, pred_x, pred_y, distance = classify(record, threshold, match_radius)
            writer.writerow(
                {
                    "clip_key": record["clip_key"],
                    "frame_id": record["frame_id"],
                    "visibility": record["visibility"],
                    "gt_x": record["gt_x"],
                    "gt_y": record["gt_y"],
                    "peak_score": record["peak_score"],
                    "pred_x": "" if pred_x is None else pred_x,
                    "pred_y": "" if pred_y is None else pred_y,
                    "distance": "" if distance is None else distance,
                    "outcome": outcome,
                    "threshold": threshold,
                    "match_radius": match_radius,
                }
            )


def discard_heatmaps(records):
    for record in records:
        record.pop("peak_crop", None)


def main():
    parser = argparse.ArgumentParser(description="Validation threshold selection and frozen V3.8 paper test.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--valid-csv", default="datasets/tracknet_v38_match_split/valid.csv")
    parser.add_argument("--test-csv", default="datasets/tracknet_v38_match_split/test.csv")
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--match-radius", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="Smoke-test limit; 0 evaluates the complete split.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_run(run_dir, device)
    valid_set = make_dataset(args.valid_csv, config)
    valid_records = collect_predictions(
        model, valid_set, device, args.batch_size, args.num_workers, config["peak_window"], args.max_samples
    )
    sweep_rows = []
    for threshold in threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step):
        metrics, _by_game = score(valid_records, threshold, args.match_radius)
        sweep_rows.append({"threshold": threshold, **metrics})
    best = max(sweep_rows, key=lambda row: (row["f1"], row["precision"], row["recall"], row["threshold"]))
    with (run_dir / "validation_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)
    write_predictions(run_dir / "validation_predictions.csv", valid_records, best["threshold"], args.match_radius)
    discard_heatmaps(valid_records)

    test_set = make_dataset(args.test_csv, config)
    test_records = collect_predictions(
        model, test_set, device, args.batch_size, args.num_workers, config["peak_window"], args.max_samples
    )
    test_metrics, test_by_game = score(test_records, best["threshold"], args.match_radius)
    write_predictions(run_dir / "test_predictions.csv", test_records, best["threshold"], args.match_radius)
    result = {
        "variant": config["variant"],
        "seed": config["seed"],
        "checkpoint": str((run_dir / "model_best.pt").resolve()),
        "selection_split": "validation: last clip of games1-7",
        "test_split": "all clips of games8-10",
        "input_space": f"{config['input_width']}x{config['input_height']}",
        "match_rule": f"Euclidean distance <= {args.match_radius} px in resized input space",
        "selected_threshold": best["threshold"],
        "validation": best,
        "test": test_metrics,
        "test_by_game": test_by_game,
    }
    (run_dir / "paper_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
