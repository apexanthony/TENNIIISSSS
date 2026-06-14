import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial import distance

from datasets_v36_clean import TrackNetDatasetV36Clean
from general_v3 import heatmap_loss, postprocess_heatmap, save_state_dict_atomic
from model_v3 import BallTrackerNetV3


def parse_thresholds(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def save_checkpoint_atomic(checkpoint, path):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def hard_negative_loss(logits, targets, hardneg_mask):
    probs = torch.sigmoid(logits)
    active = hardneg_mask * (targets < 0.05).float()
    denom = torch.clamp(active.sum(), min=1.0)
    return ((probs * probs) * active).sum() / denom


def move_batch(batch, device):
    return {
        "input": batch["input"].float().to(device),
        "ball_heatmap": batch["ball_heatmap"].float().to(device),
        "hardneg_mask": batch["hardneg_mask"].float().to(device),
        "visibility": batch["visibility"].long(),
        "x": batch["x"].float(),
        "y": batch["y"].float(),
        "clip_key": batch["clip_key"],
        "frame_id": batch["frame_id"].long(),
    }


def train_epoch(model, loader, optimizer, device, epoch, args, scaler=None):
    model.train()
    meters = {"loss": [], "heatmap": [], "hardneg": []}
    for iter_id, raw_batch in enumerate(loader):
        if iter_id >= args.steps_per_epoch:
            break
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and str(device).startswith("cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(batch["input"])
            hm_loss = heatmap_loss(
                logits,
                batch["ball_heatmap"],
                pos_weight=args.pos_weight,
                mse_weight=args.mse_weight,
            )
            hn_loss = hard_negative_loss(logits, batch["ball_heatmap"], batch["hardneg_mask"])
            loss = hm_loss + args.hardneg_weight * hn_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        meters["loss"].append(float(loss.item()))
        meters["heatmap"].append(float(hm_loss.item()))
        meters["hardneg"].append(float(hn_loss.item()))
        if args.print_interval > 0 and (iter_id % args.print_interval == 0 or iter_id + 1 >= args.steps_per_epoch):
            print(
                "train_v36 | epoch={}, iter=[{}|{}], loss={:.6f}, heatmap={:.6f}, hardneg={:.6f}".format(
                    epoch,
                    iter_id,
                    args.steps_per_epoch,
                    np.mean(meters["loss"]),
                    np.mean(meters["heatmap"]),
                    np.mean(meters["hardneg"]),
                ),
                flush=True,
            )
    return {key: float(np.mean(value)) for key, value in meters.items()}


def jump_stats(pred_rows):
    by_clip = defaultdict(list)
    for row in pred_rows:
        by_clip[row["clip_key"]].append(row)

    jumps = []
    valid = 0
    total = 0
    for rows in by_clip.values():
        rows = sorted(rows, key=lambda item: item["frame_id"])
        prev = None
        for row in rows:
            total += 1
            point = row["pred"]
            if point is None:
                prev = None
                continue
            valid += 1
            if prev is not None:
                jumps.append(distance.euclidean(point, prev))
            prev = point
    return {
        "valid": valid,
        "total": total,
        "valid_ratio": valid / max(total, 1),
        "jump80": int(sum(j > 80 for j in jumps)),
        "jump120": int(sum(j > 120 for j in jumps)),
        "jump240": int(sum(j > 240 for j in jumps)),
        "max_jump": float(max(jumps) if jumps else 0.0),
    }


def validate(model, loader, device, epoch, args):
    meters = {"loss": [], "heatmap": [], "hardneg": []}
    stats = {thr: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "visible": 0, "pred_rows": []} for thr in args.thresholds}
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
                    stats[thr]["pred_rows"].append({"clip_key": clip_key, "frame_id": frame_id, "pred": pred})
                    if vis != 0:
                        stats[thr]["visible"] += 1
                    if x_pred is not None:
                        if vis != 0 and distance.euclidean((x_pred, y_pred), (x_gt, y_gt)) < args.min_dist:
                            stats[thr]["tp"] += 1
                        else:
                            stats[thr]["fp"] += 1
                    else:
                        if vis != 0:
                            stats[thr]["fn"] += 1
                        else:
                            stats[thr]["tn"] += 1

            is_last = iter_id == len(loader) - 1 or (args.max_val_batches > 0 and iter_id + 1 >= args.max_val_batches)
            if args.val_print_interval > 0 and (iter_id % args.val_print_interval == 0 or is_last):
                print(
                    "val_v36 | epoch={}, iter=[{}|{}], loss={:.6f}".format(
                        epoch,
                        iter_id,
                        len(loader),
                        np.mean(meters["loss"]),
                    ),
                    flush=True,
                )

    eps = 1e-15
    rows = []
    for thr in args.thresholds:
        tp = stats[thr]["tp"]
        fp = stats[thr]["fp"]
        tn = stats[thr]["tn"]
        fn = stats[thr]["fn"]
        precision = tp / (tp + fp + eps)
        recall = tp / (stats[thr]["visible"] + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        js = jump_stats(stats[thr]["pred_rows"])
        rows.append(
            {
                "threshold": thr,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                **js,
            }
        )
    best = max(rows, key=lambda row: (row["f1"], row["recall"], -row["jump120"]))
    return {**{key: float(np.mean(value)) for key, value in meters.items()}, "best": best, "rows": rows}


def append_metrics(path, epoch, train_m, val_m, lr):
    exists = os.path.exists(path)
    fields = [
        "epoch", "lr",
        "train_loss", "train_heatmap", "train_hardneg",
        "val_loss", "val_heatmap", "val_hardneg",
        "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn",
        "valid_ratio", "jump80", "jump120", "jump240", "max_jump",
    ]
    with open(path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        if not exists:
            writer.writeheader()
        best = val_m["best"]
        writer.writerow(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_m["loss"],
                "train_heatmap": train_m["heatmap"],
                "train_hardneg": train_m["hardneg"],
                "val_loss": val_m["loss"],
                "val_heatmap": val_m["heatmap"],
                "val_hardneg": val_m["hardneg"],
                "threshold": best["threshold"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"],
                "tp": best["tp"],
                "fp": best["fp"],
                "tn": best["tn"],
                "fn": best["fn"],
                "valid_ratio": best["valid_ratio"],
                "jump80": best["jump80"],
                "jump120": best["jump120"],
                "jump240": best["jump240"],
                "max_jump": best["max_jump"],
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Train V3.6 clean reference-style ball heatmap tracker.")
    parser.add_argument("--train-csv", default="datasets/tennis_all_v4i_clip_split/train.csv")
    parser.add_argument("--valid-csv", default="datasets/tennis_all_v4i_clip_split/valid.csv")
    parser.add_argument("--exp-id", default="lite_heatmap_v36_clean_360x640")
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--label-height", type=int, default=720)
    parser.add_argument("--label-width", type=int, default=1280)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--heatmap-radius", type=int, default=8)
    parser.add_argument("--heatmap-sigma", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=120)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--val-intervals", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pos-weight", type=float, default=120.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--hardneg-weight", type=float, default=0.05)
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("0.30,0.40,0.50,0.60,0.70,0.80,0.90"))
    parser.add_argument("--peak-window", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=5.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default="")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-interval", type=int, default=20)
    parser.add_argument("--val-print-interval", type=int, default=100)
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join("exps", args.exp_id), exist_ok=True)

    train_set = TrackNetDatasetV36Clean(
        args.train_csv,
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        augment=args.augment,
    )
    valid_set = TrackNetDatasetV36Clean(
        args.valid_csv,
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        augment=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    model = BallTrackerNetV3(base_channels=args.base_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.num_epochs))
    scaler = torch.amp.GradScaler("cuda") if args.amp and str(device).startswith("cuda") else None

    start_epoch = 0
    best_f1 = -1.0
    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state.get("model_state", state), strict=False)
        print(f"loaded pretrained={args.pretrained}", flush=True)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", -1.0))
        print(f"resumed={args.resume}, start_epoch={start_epoch}, best_f1={best_f1}", flush=True)

    metrics_path = os.path.join("exps", args.exp_id, "metrics.csv")
    for epoch in range(start_epoch, args.num_epochs):
        train_m = train_epoch(model, train_loader, optimizer, device, epoch, args, scaler)
        scheduler.step()

        should_validate = (epoch % args.val_intervals == args.val_intervals - 1) or (epoch + 1 == args.num_epochs)
        if should_validate:
            val_m = validate(model, valid_loader, device, epoch, args)
            best = val_m["best"]
            lr = optimizer.param_groups[0]["lr"]
            append_metrics(metrics_path, epoch, train_m, val_m, lr)
            print(
                "val best_ball | epoch={} thr={:.2f} precision={:.6f} recall={:.6f} f1={:.6f} valid_ratio={:.4f} jump120={} max_jump={:.2f}".format(
                    epoch,
                    best["threshold"],
                    best["precision"],
                    best["recall"],
                    best["f1"],
                    best["valid_ratio"],
                    best["jump120"],
                    best["max_jump"],
                ),
                flush=True,
            )
            if best["f1"] > best_f1:
                best_f1 = best["f1"]
                save_state_dict_atomic(model.state_dict(), os.path.join("exps", args.exp_id, "model_best_f1.pt"))
                print(f"saved best_f1={best_f1:.6f}", flush=True)

        save_checkpoint_atomic(
            {
                "epoch": epoch,
                "best_f1": best_f1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "args": vars(args),
            },
            os.path.join("exps", args.exp_id, "training_state.pt"),
        )
        save_state_dict_atomic(model.state_dict(), os.path.join("exps", args.exp_id, "model_last.pt"))


if __name__ == "__main__":
    main()
