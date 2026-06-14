import argparse
import csv
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tensorboardX import SummaryWriter

from datasets_v34_multitask import TrackNetDatasetV34MultiTask
from general_v3 import heatmap_loss, postprocess_heatmap, save_state_dict_atomic
from model_v34 import BallTrackerNetV34MultiTask


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def save_checkpoint_atomic(checkpoint, path):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def parse_thresholds(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def compute_status_weights(train_csv, device):
    df = pd.read_csv(train_csv)
    counts = df["tracknet_status"].astype(int).value_counts().reindex([0, 1, 2], fill_value=1).astype(float)
    total = float(counts.sum())
    weights = total / (3.0 * counts)
    weights = np.minimum(weights.values, 12.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def dice_bce_loss(logits, target, pos_weight=3.0):
    pos = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos)
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return bce + dice


def multitask_loss(outputs, batch, args, status_weights):
    ball_target = batch["ball_heatmap"]
    player_target = batch["player_mask"]
    court_target = batch["court_heatmaps"]
    status_target = batch["status"]

    ball_loss = heatmap_loss(outputs["ball"], ball_target, pos_weight=args.ball_pos_weight, mse_weight=args.ball_mse_weight)
    player_loss = dice_bce_loss(outputs["player"], player_target, pos_weight=args.player_pos_weight)
    court_loss = heatmap_loss(outputs["court"], court_target, pos_weight=args.court_pos_weight, mse_weight=args.court_mse_weight)
    status_loss = F.cross_entropy(outputs["status"], status_target, weight=status_weights)
    total = (
        args.ball_weight * ball_loss
        + args.player_weight * player_loss
        + args.court_weight * court_loss
        + args.status_weight * status_loss
    )
    return total, {
        "ball": ball_loss,
        "player": player_loss,
        "court": court_loss,
        "status": status_loss,
    }


def move_batch(batch, device):
    return {
        "input": batch["input"].float().to(device),
        "ball_heatmap": batch["ball_heatmap"].float().to(device),
        "player_mask": batch["player_mask"].float().to(device),
        "court_heatmaps": batch["court_heatmaps"].float().to(device),
        "status": batch["status"].long().to(device),
        "visibility": batch["visibility"].long(),
        "x": batch["x"].float(),
        "y": batch["y"].float(),
    }


def train_epoch(model, loader, optimizer, device, epoch, args, status_weights, scaler=None):
    model.train()
    meters = {"loss": [], "ball": [], "player": [], "court": [], "status": []}
    for iter_id, raw_batch in enumerate(loader):
        if iter_id >= args.steps_per_epoch:
            break
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and str(device).startswith("cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(batch["input"])
            loss, losses = multitask_loss(outputs, batch, args, status_weights)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        meters["loss"].append(float(loss.item()))
        for key in ["ball", "player", "court", "status"]:
            meters[key].append(float(losses[key].item()))

        if args.print_interval > 0 and (iter_id % args.print_interval == 0 or iter_id + 1 >= args.steps_per_epoch):
            print(
                "train_v34 | epoch={}, iter=[{}|{}], loss={:.6f}, ball={:.6f}, player={:.6f}, court={:.6f}, status={:.6f}".format(
                    epoch,
                    iter_id,
                    args.steps_per_epoch,
                    np.mean(meters["loss"]),
                    np.mean(meters["ball"]),
                    np.mean(meters["player"]),
                    np.mean(meters["court"]),
                    np.mean(meters["status"]),
                ),
                flush=True,
            )
    return {key: float(np.mean(value)) for key, value in meters.items()}


def status_metrics(y_true, y_pred, num_classes=3):
    rows = []
    eps = 1e-15
    for cls in range(num_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        rows.append((precision, recall, f1))
    return {
        "acc": float(np.mean(y_true == y_pred)),
        "macro_precision": float(np.mean([row[0] for row in rows])),
        "macro_recall": float(np.mean([row[1] for row in rows])),
        "macro_f1": float(np.mean([row[2] for row in rows])),
    }


def validate(model, loader, device, epoch, args, status_weights):
    meters = {"loss": [], "ball": [], "player": [], "court": [], "status": []}
    stats = {thr: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "visible": 0} for thr in args.thresholds}
    y_true = []
    y_pred = []
    scale_x = float(args.label_width) / float(args.input_width)
    scale_y = float(args.label_height) / float(args.input_height)

    model.eval()
    with torch.no_grad():
        for iter_id, raw_batch in enumerate(loader):
            batch = move_batch(raw_batch, device)
            outputs = model(batch["input"])
            loss, losses = multitask_loss(outputs, batch, args, status_weights)
            meters["loss"].append(float(loss.item()))
            for key in ["ball", "player", "court", "status"]:
                meters[key].append(float(losses[key].item()))

            heatmaps = torch.sigmoid(outputs["ball"]).detach().cpu().numpy()
            for i in range(len(heatmaps)):
                x_gt = float(batch["x"][i])
                y_gt = float(batch["y"][i])
                vis = int(batch["visibility"][i])
                for thr in args.thresholds:
                    x_pred, y_pred_ball = postprocess_heatmap(
                        heatmaps[i, 0],
                        threshold=thr,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        peak_window=args.peak_window,
                    )
                    if vis != 0:
                        stats[thr]["visible"] += 1
                    if x_pred is not None:
                        if vis != 0 and distance.euclidean((x_pred, y_pred_ball), (x_gt, y_gt)) < args.min_dist:
                            stats[thr]["tp"] += 1
                        else:
                            stats[thr]["fp"] += 1
                    else:
                        if vis != 0:
                            stats[thr]["fn"] += 1
                        else:
                            stats[thr]["tn"] += 1

            status_pred = torch.argmax(outputs["status"], dim=1).detach().cpu().numpy()
            y_pred.extend(status_pred.tolist())
            y_true.extend(batch["status"].detach().cpu().numpy().tolist())

            if args.val_print_interval > 0 and (iter_id % args.val_print_interval == 0 or iter_id == len(loader) - 1):
                print(
                    "val_v34 | epoch={}, iter=[{}|{}], loss={:.6f}".format(
                        epoch,
                        iter_id,
                        len(loader),
                        np.mean(meters["loss"]),
                    ),
                    flush=True,
                )

    ball_rows = []
    eps = 1e-15
    for thr in args.thresholds:
        tp = stats[thr]["tp"]
        fp = stats[thr]["fp"]
        tn = stats[thr]["tn"]
        fn = stats[thr]["fn"]
        precision = tp / (tp + fp + eps)
        recall = tp / (stats[thr]["visible"] + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        ball_rows.append({"threshold": thr, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    best_ball = max(ball_rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    status = status_metrics(np.asarray(y_true), np.asarray(y_pred))
    return {
        **{key: float(np.mean(value)) for key, value in meters.items()},
        "best_ball": best_ball,
        "status": status,
    }


def append_metrics(path, epoch, train_m, val_m, lr):
    exists = os.path.exists(path)
    fields = [
        "epoch", "lr",
        "train_loss", "train_ball", "train_player", "train_court", "train_status",
        "val_loss", "val_ball", "val_player", "val_court", "val_status",
        "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn",
        "status_acc", "status_macro_precision", "status_macro_recall", "status_macro_f1",
    ]
    with open(path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        if not exists:
            writer.writeheader()
        best = val_m["best_ball"]
        status = val_m["status"]
        writer.writerow({
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_m["loss"],
            "train_ball": train_m["ball"],
            "train_player": train_m["player"],
            "train_court": train_m["court"],
            "train_status": train_m["status"],
            "val_loss": val_m["loss"],
            "val_ball": val_m["ball"],
            "val_player": val_m["player"],
            "val_court": val_m["court"],
            "val_status": val_m["status"],
            "threshold": best["threshold"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "tp": best["tp"],
            "fp": best["fp"],
            "tn": best["tn"],
            "fn": best["fn"],
            "status_acc": status["acc"],
            "status_macro_precision": status["macro_precision"],
            "status_macro_recall": status["macro_recall"],
            "status_macro_f1": status["macro_f1"],
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train V3.4 multi-task TrackNet from scratch.")
    parser.add_argument("--exp_id", default="lite_heatmap_v34_multitask_clip")
    parser.add_argument("--train_csv", default="datasets/tennis_all_v4i_clip_split/train.csv")
    parser.add_argument("--valid_csv", default="datasets/tennis_all_v4i_clip_split/valid.csv")
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_intervals", type=int, default=5)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--input_height", type=int, default=270)
    parser.add_argument("--input_width", type=int, default=480)
    parser.add_argument("--label_height", type=int, default=720)
    parser.add_argument("--label_width", type=int, default=1280)
    parser.add_argument("--heatmap_radius", type=int, default=6)
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)
    parser.add_argument("--court_radius", type=int, default=5)
    parser.add_argument("--court_sigma", type=float, default=2.0)
    parser.add_argument("--ball_weight", type=float, default=1.0)
    parser.add_argument("--player_weight", type=float, default=0.05)
    parser.add_argument("--court_weight", type=float, default=0.05)
    parser.add_argument("--status_weight", type=float, default=0.15)
    parser.add_argument("--ball_pos_weight", type=float, default=80.0)
    parser.add_argument("--ball_mse_weight", type=float, default=1.0)
    parser.add_argument("--player_pos_weight", type=float, default=3.0)
    parser.add_argument("--court_pos_weight", type=float, default=40.0)
    parser.add_argument("--court_mse_weight", type=float, default=1.0)
    parser.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95")
    parser.add_argument("--peak_window", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=8.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_interval", type=int, default=20)
    parser.add_argument("--val_print_interval", type=int, default=100)
    parser.add_argument("--snapshot_interval", type=int, default=25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--pretrained", default="", help="state_dict .pt to initialize model weights")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.thresholds = parse_thresholds(args.thresholds)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TrackNetDatasetV34MultiTask(
        args.train_csv,
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        court_radius=args.court_radius,
        court_sigma=args.court_sigma,
        augment=args.augment,
    )
    valid_dataset = TrackNetDatasetV34MultiTask(
        args.valid_csv,
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        court_radius=args.court_radius,
        court_sigma=args.court_sigma,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device == "cuda")
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device == "cuda")

    model = BallTrackerNetV34MultiTask(base_channels=args.base_channels).to(device)
    if args.pretrained:
        print(f"loading pretrained weights: {args.pretrained}", flush=True)
        state = torch.load(args.pretrained, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device == "cuda" else None
    status_weights = compute_status_weights(args.train_csv, device)
    print(f"status_weights = {status_weights.detach().cpu().numpy().tolist()}", flush=True)

    exp_dir = os.path.join("exps", args.exp_id)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    try:
        writer = SummaryWriter(os.path.join(exp_dir, "plots"))
    except Exception:
        writer = NullWriter()
    metrics_path = os.path.join(exp_dir, "metrics.csv")
    best_f1 = 0.0
    stale = 0

    print(
        "V3.4 config: train_csv={}, valid_csv={}, input={}x{}, batch={}, epochs={}, device={}, amp={}".format(
            args.train_csv, args.valid_csv, args.input_height, args.input_width, args.batch_size, args.num_epochs, device, scaler is not None
        ),
        flush=True,
    )

    for epoch in range(args.num_epochs):
        train_m = train_epoch(model, train_loader, optimizer, device, epoch, args, status_weights, scaler=scaler)
        writer.add_scalar("Train/loss", train_m["loss"], epoch)
        writer.add_scalar("Train/ball", train_m["ball"], epoch)
        writer.add_scalar("Train/status", train_m["status"], epoch)
        should_validate = (epoch + 1) % args.val_intervals == 0
        if should_validate:
            val_m = validate(model, valid_loader, device, epoch, args, status_weights)
            best = val_m["best_ball"]
            status = val_m["status"]
            print(
                "val best_ball | epoch={} thr={:.2f} precision={:.6f} recall={:.6f} f1={:.6f} status_macro_f1={:.6f}".format(
                    epoch, best["threshold"], best["precision"], best["recall"], best["f1"], status["macro_f1"]
                ),
                flush=True,
            )
            append_metrics(metrics_path, epoch, train_m, val_m, optimizer.param_groups[0]["lr"])
            writer.add_scalar("Val/f1", best["f1"], epoch)
            writer.add_scalar("Val/precision", best["precision"], epoch)
            writer.add_scalar("Val/recall", best["recall"], epoch)
            writer.add_scalar("Val/status_macro_f1", status["macro_f1"], epoch)

            if best["f1"] > best_f1:
                best_f1 = best["f1"]
                stale = 0
                save_state_dict_atomic(model.state_dict(), os.path.join(exp_dir, "model_best_f1.pt"))
                print(f"saved best_f1: {best_f1:.6f}", flush=True)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stop: no f1 improvement for {args.patience} validations", flush=True)
                    break

        save_state_dict_atomic(model.state_dict(), os.path.join(exp_dir, "model_last.pt"))
        save_checkpoint_atomic(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "args": vars(args),
            },
            os.path.join(exp_dir, "training_state.pt"),
        )
        if args.snapshot_interval > 0 and (epoch + 1) % args.snapshot_interval == 0:
            save_state_dict_atomic(model.state_dict(), os.path.join(exp_dir, f"model_epoch_{epoch + 1:03d}.pt"))
        scheduler.step()
        writer.flush()

    print(f"training finished, best_f1={best_f1:.6f}", flush=True)
    writer.close()
