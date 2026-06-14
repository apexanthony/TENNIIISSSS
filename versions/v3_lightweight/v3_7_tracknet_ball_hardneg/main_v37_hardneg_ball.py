import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial import distance

from datasets_v37_hardneg import TrackNetDatasetV37HardNeg
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


def set_learning_rate(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def warmup_learning_rate(epoch, warmup_epochs, start_lr, target_lr):
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return float(target_lr)
    progress = float(epoch + 1) / float(warmup_epochs)
    return float(start_lr + (target_lr - start_lr) * progress)


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
    total_steps = len(loader) if args.steps_per_epoch <= 0 else min(len(loader), args.steps_per_epoch)
    for iter_id, raw_batch in enumerate(loader):
        if args.steps_per_epoch > 0 and iter_id >= args.steps_per_epoch:
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
        if args.print_interval > 0 and (iter_id % args.print_interval == 0 or iter_id + 1 >= total_steps):
            print(
                "train_v37 | epoch={}, iter=[{}|{}], loss={:.6f}, heatmap={:.6f}, hardneg={:.6f}".format(
                    epoch,
                    iter_id,
                    total_steps,
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
                    "val_v37 | epoch={}, iter=[{}|{}], loss={:.6f}".format(
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
    parser = argparse.ArgumentParser(description="Train V3.7 TrackNet ball-only hard-negative heatmap tracker.")
    parser.add_argument("--train-csv", default="datasets/tracknet_v37_clip_split/train.csv")
    parser.add_argument("--valid-csv", default="datasets/tracknet_v37_clip_split/valid.csv")
    parser.add_argument("--exp-id", default="lite_heatmap_v37_tracknet_hardneg_ballonly_360x640")
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--label-height", type=int, default=720)
    parser.add_argument("--label-width", type=int, default=1280)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--heatmap-radius", type=int, default=5)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=0,
        help="0 traverses the complete training DataLoader; positive values are intended only for short searches.",
    )
    parser.add_argument("--val-intervals", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default="plateau")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-start-lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=40)
    parser.add_argument("--pos-weight", type=float, default=120.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--hardneg-weight", type=float, default=0.05)
    parser.add_argument(
        "--mapped-csv",
        default="datasets/tennis_all_v4i_mapped/annotations_hardneg_cleaned_strict.csv",
    )
    parser.add_argument("--hardneg-player-weight", type=float, default=1.0)
    parser.add_argument("--hardneg-shoe-weight", type=float, default=1.4)
    parser.add_argument("--hardneg-court-weight", type=float, default=0.85)
    parser.add_argument("--hardneg-bright-weight", type=float, default=0.35)
    parser.add_argument("--hardneg-edge-weight", type=float, default=0.18)
    parser.add_argument("--hardneg-ball-clear-radius", type=int, default=18)
    parser.add_argument("--augment-min-ball-contrast", type=float, default=4.0)
    parser.add_argument("--augment-min-contrast-ratio", type=float, default=0.55)
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("0.30,0.40,0.50,0.60,0.70,0.80,0.90"))
    parser.add_argument("--peak-window", type=int, default=9)
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
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--target-f1", type=float, default=0.95)
    args = parser.parse_args()

    if args.val_intervals != 1:
        print("V3.7 requires per-epoch validation; overriding val_intervals=1", flush=True)
        args.val_intervals = 1
    if args.num_epochs > 100:
        print("production training is capped at 100 epochs; overriding num_epochs=100", flush=True)
        args.num_epochs = 100

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join("exps", args.exp_id), exist_ok=True)

    train_set = TrackNetDatasetV37HardNeg(
        args.train_csv,
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        augment=args.augment,
        mapped_csv=args.mapped_csv,
        hardneg_player_weight=args.hardneg_player_weight,
        hardneg_shoe_weight=args.hardneg_shoe_weight,
        hardneg_court_weight=args.hardneg_court_weight,
        hardneg_bright_weight=args.hardneg_bright_weight,
        hardneg_edge_weight=args.hardneg_edge_weight,
        hardneg_ball_clear_radius=args.hardneg_ball_clear_radius,
        augment_min_ball_contrast=args.augment_min_ball_contrast,
        augment_min_contrast_ratio=args.augment_min_contrast_ratio,
    )
    valid_set = TrackNetDatasetV37HardNeg(
        args.valid_csv,
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        augment=False,
        mapped_csv=args.mapped_csv,
        hardneg_player_weight=args.hardneg_player_weight,
        hardneg_shoe_weight=args.hardneg_shoe_weight,
        hardneg_court_weight=args.hardneg_court_weight,
        hardneg_bright_weight=args.hardneg_bright_weight,
        hardneg_edge_weight=args.hardneg_edge_weight,
        hardneg_ball_clear_radius=args.hardneg_ball_clear_radius,
        augment_min_ball_contrast=args.augment_min_ball_contrast,
        augment_min_contrast_ratio=args.augment_min_contrast_ratio,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=str(device).startswith("cuda"),
    )

    model = BallTrackerNetV3(base_channels=args.base_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            threshold=args.scheduler_threshold,
            threshold_mode="abs",
            min_lr=args.min_lr,
        )
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.num_epochs - args.warmup_epochs),
            eta_min=args.min_lr,
        )
    else:
        scheduler = None
    scaler = torch.amp.GradScaler("cuda") if args.amp and str(device).startswith("cuda") else None

    start_epoch = 0
    best_f1 = -1.0
    no_improve_validations = 0
    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state.get("model_state", state), strict=False)
        print(f"loaded pretrained={args.pretrained}", flush=True)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        checkpoint_scheduler = ckpt.get("scheduler_name")
        if checkpoint_scheduler is None:
            checkpoint_scheduler = ckpt.get("args", {}).get("scheduler", "cosine")
        if (
            scheduler is not None
            and ckpt.get("scheduler_state") is not None
            and checkpoint_scheduler == args.scheduler
        ):
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except (KeyError, ValueError, TypeError) as error:
                print(f"warning: scheduler state was not restored: {error}", flush=True)
        elif checkpoint_scheduler != args.scheduler:
            print(
                f"warning: checkpoint scheduler={checkpoint_scheduler} differs from requested scheduler={args.scheduler}; "
                "starting a fresh scheduler",
                flush=True,
            )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", -1.0))
        no_improve_validations = int(ckpt.get("no_improve_validations", 0))
        print(f"resumed={args.resume}, start_epoch={start_epoch}, best_f1={best_f1}", flush=True)

    metrics_path = os.path.join("exps", args.exp_id, "metrics.csv")
    target_reached = False
    for epoch in range(start_epoch, args.num_epochs):
        if epoch < args.warmup_epochs:
            set_learning_rate(
                optimizer,
                warmup_learning_rate(epoch, args.warmup_epochs, args.warmup_start_lr, args.lr),
            )
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_m = train_epoch(model, train_loader, optimizer, device, epoch, args, scaler)

        should_validate = (epoch % args.val_intervals == args.val_intervals - 1) or (epoch + 1 == args.num_epochs)
        stop_early = False
        if should_validate:
            val_m = validate(model, valid_loader, device, epoch, args)
            best = val_m["best"]
            append_metrics(metrics_path, epoch, train_m, val_m, epoch_lr)
            print(
                "val best_ball | epoch={} lr={:.8f} thr={:.2f} precision={:.6f} recall={:.6f} f1={:.6f} valid_ratio={:.4f} jump120={} max_jump={:.2f}".format(
                    epoch,
                    epoch_lr,
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
                no_improve_validations = 0
                save_state_dict_atomic(model.state_dict(), os.path.join("exps", args.exp_id, "model_best_f1.pt"))
                print(f"saved best_f1={best_f1:.6f}", flush=True)
            else:
                no_improve_validations += 1

            if args.scheduler == "plateau" and scheduler is not None and epoch >= args.warmup_epochs:
                old_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(best["f1"])
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr != old_lr:
                    print(f"lr adjusted | old={old_lr:.8f} new={new_lr:.8f}", flush=True)

            stop_early = (
                args.early_stop_patience > 0
                and no_improve_validations >= args.early_stop_patience
            )
            target_reached = (
                best["precision"] >= args.target_precision
                and best["recall"] >= args.target_recall
                and best["f1"] >= args.target_f1
            )

        if args.scheduler == "cosine" and scheduler is not None and epoch >= args.warmup_epochs:
            scheduler.step()

        save_checkpoint_atomic(
            {
                "epoch": epoch,
                "best_f1": best_f1,
                "no_improve_validations": no_improve_validations,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scheduler_name": args.scheduler,
                "args": vars(args),
            },
            os.path.join("exps", args.exp_id, "training_state.pt"),
        )
        save_state_dict_atomic(model.state_dict(), os.path.join("exps", args.exp_id, "model_last.pt"))
        if target_reached:
            print(
                "target reached | precision={:.6f} recall={:.6f} f1={:.6f}".format(
                    best["precision"],
                    best["recall"],
                    best["f1"],
                ),
                flush=True,
            )
            break
        if stop_early:
            print(
                f"early stop | no F1 improvement for {no_improve_validations} validations, best_f1={best_f1:.6f}",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
