import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch
from scipy.spatial import distance

from datasets_v36_clean import TrackNetDatasetV36Clean
from general_v3 import heatmap_loss, postprocess_heatmap
from main_v36_clean_reference import hard_negative_loss, jump_stats, move_batch, parse_thresholds
from model_v3 import BallTrackerNetV3


def parse_distances(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        return state["model_state"]
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def evaluate_once(model, loader, device, args):
    meters = {"loss": [], "heatmap": [], "hardneg": []}
    by_threshold = {
        thr: {"records": [], "pred_rows": [], "visible": 0}
        for thr in args.thresholds
    }
    scale_x = float(args.label_width) / float(args.input_width)
    scale_y = float(args.label_height) / float(args.input_height)

    model.eval()
    with torch.no_grad():
        for iter_id, raw_batch in enumerate(loader):
            if args.max_val_batches > 0 and iter_id >= args.max_val_batches:
                break
            batch = move_batch(raw_batch, device)
            logits = model(batch["input"])
            hm_loss = heatmap_loss(logits, batch["ball_heatmap"], pos_weight=args.pos_weight, mse_weight=args.mse_weight)
            hn_loss = hard_negative_loss(logits, batch["ball_heatmap"], batch["hardneg_mask"])
            loss = hm_loss + args.hardneg_weight * hn_loss
            meters["loss"].append(float(loss.item()))
            meters["heatmap"].append(float(hm_loss.item()))
            meters["hardneg"].append(float(hn_loss.item()))

            heatmaps = torch.sigmoid(logits).detach().cpu().numpy()
            for i in range(len(heatmaps)):
                x_gt = float(batch["x"][i])
                y_gt = float(batch["y"][i])
                vis = int(batch["visibility"][i])
                clip_key = batch["clip_key"][i]
                frame_id = int(batch["frame_id"][i])
                for thr in args.thresholds:
                    x_pred, y_pred = postprocess_heatmap(
                        heatmaps[i, 0],
                        threshold=thr,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        peak_window=args.peak_window,
                    )
                    pred = None if x_pred is None else (x_pred, y_pred)
                    by_threshold[thr]["pred_rows"].append({"clip_key": clip_key, "frame_id": frame_id, "pred": pred})
                    if vis != 0:
                        by_threshold[thr]["visible"] += 1
                    dist = None
                    if pred is not None and vis != 0:
                        dist = distance.euclidean(pred, (x_gt, y_gt))
                    by_threshold[thr]["records"].append({"pred": pred, "visible": vis != 0, "dist": dist})

            is_last = iter_id == len(loader) - 1 or (args.max_val_batches > 0 and iter_id + 1 >= args.max_val_batches)
            if args.val_print_interval > 0 and (iter_id % args.val_print_interval == 0 or is_last):
                print(
                    "eval_v36 | iter=[{}|{}], loss={:.6f}".format(iter_id, len(loader), np.mean(meters["loss"])),
                    flush=True,
                )

    eps = 1e-15
    rows = []
    for min_dist in args.min_dists:
        for thr, stat in by_threshold.items():
            tp = fp = tn = fn = 0
            for record in stat["records"]:
                pred = record["pred"]
                visible = record["visible"]
                if pred is not None:
                    if visible and record["dist"] is not None and record["dist"] < min_dist:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if visible:
                        fn += 1
                    else:
                        tn += 1
            precision = tp / (tp + fp + eps)
            recall = tp / (stat["visible"] + eps)
            f1 = 2.0 * precision * recall / (precision + recall + eps)
            rows.append(
                {
                    "min_dist": min_dist,
                    "threshold": thr,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    **jump_stats(stat["pred_rows"]),
                    "val_loss": float(np.mean(meters["loss"])),
                }
            )
    best_by_dist = {}
    for min_dist in args.min_dists:
        candidates = [row for row in rows if row["min_dist"] == min_dist]
        best_by_dist[min_dist] = max(candidates, key=lambda row: (row["f1"], row["recall"], -row["jump120"]))
    return rows, best_by_dist


def main():
    parser = argparse.ArgumentParser(description="Evaluate V3.6 with multiple hit-distance radii.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid-csv", default="datasets/tennis_all_v4i_clip_split_cleaned/valid.csv")
    parser.add_argument("--output-csv", default="exps/v36_cleaned_sensitivity.csv")
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--label-height", type=int, default=720)
    parser.add_argument("--label-width", type=int, default=1280)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--heatmap-radius", type=int, default=8)
    parser.add_argument("--heatmap-sigma", type=float, default=3.0)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("0.30,0.40,0.50,0.60,0.70,0.80,0.90"))
    parser.add_argument("--min-dists", type=parse_distances, default=parse_distances("5,8,10,15"))
    parser.add_argument("--peak-window", type=int, default=15)
    parser.add_argument("--pos-weight", type=float, default=120.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--hardneg-weight", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--val-print-interval", type=int, default=100)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrackNetDatasetV36Clean(
        args.valid_csv,
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        augment=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    model = BallTrackerNetV3(base_channels=args.base_channels).to(device)
    model.load_state_dict(load_state(args.checkpoint, device), strict=False)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    fields = [
        "min_dist", "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn",
        "valid_ratio", "jump80", "jump120", "jump240", "max_jump", "val_loss",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        _, best_by_dist = evaluate_once(model, loader, device, args)
        for min_dist in args.min_dists:
            best = best_by_dist[min_dist]
            row = {key: best.get(key) for key in fields if key not in {"min_dist", "val_loss"}}
            row["min_dist"] = min_dist
            row["val_loss"] = best["val_loss"]
            writer.writerow(row)
            print(
                "min_dist={:.1f} thr={:.2f} precision={:.6f} recall={:.6f} f1={:.6f} jump120={}".format(
                    min_dist, best["threshold"], best["precision"], best["recall"], best["f1"], best["jump120"]
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
