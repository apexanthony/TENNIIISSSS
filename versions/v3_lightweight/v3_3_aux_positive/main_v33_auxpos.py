import argparse
import csv
import os

import numpy as np
import torch
import torch.optim as optim
from scipy.spatial import distance
from tensorboardX import SummaryWriter

from datasets_v33_auxpos import TrackNetDatasetV33AuxPos
from general_v3 import heatmap_loss, postprocess_heatmap, save_state_dict_atomic
from model_v3 import BallTrackerNetV3


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


def aux_positive_loss(logits, aux_target, aux_valid, pos_weight, mse_weight):
    valid = aux_valid.float().view(-1) > 0.5
    if not torch.any(valid):
        return logits.sum() * 0.0
    return heatmap_loss(
        logits[valid],
        aux_target[valid],
        pos_weight=pos_weight,
        mse_weight=mse_weight,
    )


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    max_iters,
    aux_weight,
    pos_weight,
    mse_weight,
    aux_pos_weight,
    aux_mse_weight,
    print_interval,
    scaler=None,
):
    model.train()
    total_losses = []
    main_losses = []
    aux_losses = []
    aux_coverages = []
    for iter_id, batch in enumerate(train_loader):
        if iter_id >= max_iters:
            break
        optimizer.zero_grad(set_to_none=True)
        inp = batch[0].float().to(device)
        target = batch[1].float().to(device)
        aux_target = batch[5].float().to(device)
        aux_valid = batch[6].float().to(device)
        use_amp = scaler is not None and str(device).startswith("cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(inp)
            main_loss = heatmap_loss(logits, target, pos_weight=pos_weight, mse_weight=mse_weight)
            aux_loss = aux_positive_loss(
                logits,
                aux_target,
                aux_valid,
                pos_weight=aux_pos_weight,
                mse_weight=aux_mse_weight,
            )
            loss = main_loss + aux_weight * aux_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_losses.append(float(loss.item()))
        main_losses.append(float(main_loss.item()))
        aux_losses.append(float(aux_loss.item()))
        aux_coverages.append(float(aux_valid.float().mean().item()))
        if print_interval > 0 and (iter_id % print_interval == 0 or iter_id + 1 >= max_iters):
            print(
                "train_v33 | epoch = {}, iter = [{}|{}], loss = {:.6f}, main = {:.6f}, aux = {:.6f}, aux_cov = {:.3f}, aux_w = {:.3f}".format(
                    epoch,
                    iter_id,
                    max_iters,
                    float(np.mean(total_losses)),
                    float(np.mean(main_losses)),
                    float(np.mean(aux_losses)),
                    float(np.mean(aux_coverages)),
                    aux_weight,
                ),
                flush=True,
            )

    return {
        "loss": float(np.mean(total_losses)),
        "main_loss": float(np.mean(main_losses)),
        "aux_loss": float(np.mean(aux_losses)),
        "aux_coverage": float(np.mean(aux_coverages)),
    }


def validate_sweep(
    model,
    val_loader,
    device,
    epoch,
    thresholds,
    input_width,
    input_height,
    label_width,
    label_height,
    min_dist,
    peak_window,
    pos_weight,
    mse_weight,
    aux_weight,
    aux_pos_weight,
    aux_mse_weight,
    print_interval,
):
    losses = []
    main_losses = []
    aux_losses = []
    stats = {thr: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "visible": 0} for thr in thresholds}
    scale_x = float(label_width) / float(input_width)
    scale_y = float(label_height) / float(input_height)

    model.eval()
    with torch.no_grad():
        for iter_id, batch in enumerate(val_loader):
            inp = batch[0].float().to(device)
            target = batch[1].float().to(device)
            aux_target = batch[5].float().to(device)
            aux_valid = batch[6].float().to(device)
            logits = model(inp)
            main_loss = heatmap_loss(logits, target, pos_weight=pos_weight, mse_weight=mse_weight)
            aux_loss = aux_positive_loss(
                logits,
                aux_target,
                aux_valid,
                pos_weight=aux_pos_weight,
                mse_weight=aux_mse_weight,
            )
            loss = main_loss + aux_weight * aux_loss
            losses.append(float(loss.item()))
            main_losses.append(float(main_loss.item()))
            aux_losses.append(float(aux_loss.item()))

            heatmaps = torch.sigmoid(logits).detach().cpu().numpy()
            for i in range(len(heatmaps)):
                x_gt = float(batch[2][i])
                y_gt = float(batch[3][i])
                vis = int(batch[4][i])
                for thr in thresholds:
                    x_pred, y_pred = postprocess_heatmap(
                        heatmaps[i, 0],
                        threshold=thr,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        peak_window=peak_window,
                    )
                    if vis != 0:
                        stats[thr]["visible"] += 1
                    if x_pred is not None:
                        if vis != 0:
                            dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                            if dst < min_dist:
                                stats[thr]["tp"] += 1
                            else:
                                stats[thr]["fp"] += 1
                        else:
                            stats[thr]["fp"] += 1
                    else:
                        if vis != 0:
                            stats[thr]["fn"] += 1
                        else:
                            stats[thr]["tn"] += 1

            if print_interval > 0 and (iter_id % print_interval == 0 or iter_id == len(val_loader) - 1):
                print(
                    "val_v33 | epoch = {}, iter = [{}|{}], loss = {:.6f}".format(
                        epoch,
                        iter_id,
                        len(val_loader),
                        float(np.mean(losses)),
                    ),
                    flush=True,
                )

    rows = []
    eps = 1e-15
    for thr in thresholds:
        tp = stats[thr]["tp"]
        fp = stats[thr]["fp"]
        tn = stats[thr]["tn"]
        fn = stats[thr]["fn"]
        precision = tp / (tp + fp + eps)
        recall = tp / (stats[thr]["visible"] + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        rows.append(
            {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    best_f1 = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    best_balanced = max(rows, key=lambda row: (min(row["precision"], row["recall"], row["f1"]), row["f1"]))
    best_recall94_candidates = [row for row in rows if row["precision"] >= 0.94]
    best_recall94 = (
        max(best_recall94_candidates, key=lambda row: (row["recall"], row["f1"]))
        if best_recall94_candidates
        else None
    )
    return {
        "loss": float(np.mean(losses)),
        "main_loss": float(np.mean(main_losses)),
        "aux_loss": float(np.mean(aux_losses)),
        "rows": rows,
        "best_f1": best_f1,
        "best_balanced": best_balanced,
        "best_recall94": best_recall94,
    }


def append_metrics(path, epoch, train_metrics, val_result, lr, aux_weight):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_main_loss",
            "train_aux_loss",
            "train_aux_coverage",
            "val_loss",
            "val_main_loss",
            "val_aux_loss",
            "threshold",
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "tn",
            "fn",
            "lr",
            "aux_weight",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        best = val_result["best_f1"]
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_main_loss": train_metrics["main_loss"],
                "train_aux_loss": train_metrics["aux_loss"],
                "train_aux_coverage": train_metrics["aux_coverage"],
                "val_loss": val_result["loss"],
                "val_main_loss": val_result["main_loss"],
                "val_aux_loss": val_result["aux_loss"],
                "threshold": best["threshold"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"],
                "tp": best["tp"],
                "fp": best["fp"],
                "tn": best["tn"],
                "fn": best["fn"],
                "lr": lr,
                "aux_weight": aux_weight,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrackNet V3.3 with Roboflow ball-box auxiliary positives.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--exp_id", type=str, default="lite_heatmap_v33_auxpos")
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--val_intervals", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--input_height", type=int, default=270)
    parser.add_argument("--input_width", type=int, default=480)
    parser.add_argument("--label_height", type=int, default=720)
    parser.add_argument("--label_width", type=int, default=1280)
    parser.add_argument("--heatmap_radius", type=int, default=6)
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)
    parser.add_argument("--aux_heatmap_radius", type=int, default=5)
    parser.add_argument("--aux_heatmap_sigma", type=float, default=1.8)
    parser.add_argument("--pos_weight", type=float, default=80.0)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=0.15)
    parser.add_argument("--aux_pos_weight", type=float, default=40.0)
    parser.add_argument("--aux_mse_weight", type=float, default=1.0)
    parser.add_argument("--thresholds", type=str, default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95")
    parser.add_argument("--peak_window", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=8.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--print_interval", type=int, default=20)
    parser.add_argument("--val_print_interval", type=int, default=100)
    parser.add_argument("--snapshot_interval", type=int, default=20)
    parser.add_argument("--mapped_csv", default="datasets/tennis_all_v4i_mapped/annotations_mapped.csv")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--target_precision", type=float, default=0.94)
    parser.add_argument("--target_recall", type=float, default=0.94)
    parser.add_argument("--target_f1", type=float, default=0.94)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = parse_thresholds(args.thresholds)

    train_dataset = TrackNetDatasetV33AuxPos(
        "train",
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        aux_heatmap_radius=args.aux_heatmap_radius,
        aux_heatmap_sigma=args.aux_heatmap_sigma,
        augment=args.augment,
        mapped_csv=args.mapped_csv,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    val_dataset = TrackNetDatasetV33AuxPos(
        "val",
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        aux_heatmap_radius=args.aux_heatmap_radius,
        aux_heatmap_sigma=args.aux_heatmap_sigma,
        mapped_csv=args.mapped_csv,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    model = BallTrackerNetV3(base_channels=args.base_channels).to(device)
    resume_checkpoint = None
    best_f1 = 0.0
    if args.resume:
        print(f"loading checkpoint: {args.resume}", flush=True)
        resume_checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        args.start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
        best_f1 = float(resume_checkpoint.get("best_f1", 0.0))
    elif args.pretrained:
        print(f"loading pretrained weights: {args.pretrained}", flush=True)
        state = torch.load(args.pretrained, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

    exps_path = os.path.join("./exps", args.exp_id)
    tb_path = os.path.join(exps_path, "plots")
    os.makedirs(tb_path, exist_ok=True)
    try:
        log_writer = SummaryWriter(tb_path)
    except Exception as exc:
        print(f"warning: tensorboard writer disabled: {exc}", flush=True)
        log_writer = NullWriter()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.num_epochs - args.start_epoch),
        eta_min=args.lr * 0.2,
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])

    scaler = torch.amp.GradScaler("cuda") if args.amp and device == "cuda" else None
    model_last_path = os.path.join(exps_path, "model_last.pt")
    model_best_f1_path = os.path.join(exps_path, "model_best_f1.pt")
    model_best_balanced_path = os.path.join(exps_path, "model_best_balanced.pt")
    checkpoint_path = os.path.join(exps_path, "training_state.pt")
    metrics_path = os.path.join(exps_path, "metrics.csv")
    best_balanced_score = 0.0

    print(
        "V3.3 config: input={}x{}, batch_size={}, device={}, amp={}, val_intervals={}, aux_weight={}".format(
            args.input_height,
            args.input_width,
            args.batch_size,
            device,
            scaler is not None,
            args.val_intervals,
            args.aux_weight,
        ),
        flush=True,
    )

    for epoch in range(args.start_epoch, args.num_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.steps_per_epoch,
            args.aux_weight,
            args.pos_weight,
            args.mse_weight,
            args.aux_pos_weight,
            args.aux_mse_weight,
            args.print_interval,
            scaler=scaler,
        )
        log_writer.add_scalar("Train/loss", train_metrics["loss"], epoch)
        log_writer.add_scalar("Train/main_loss", train_metrics["main_loss"], epoch)
        log_writer.add_scalar("Train/aux_loss", train_metrics["aux_loss"], epoch)
        log_writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

        should_validate = ((epoch + 1) % args.val_intervals == 0)
        if should_validate:
            val_result = validate_sweep(
                model,
                val_loader,
                device,
                epoch,
                thresholds,
                args.input_width,
                args.input_height,
                args.label_width,
                args.label_height,
                args.min_dist,
                args.peak_window,
                args.pos_weight,
                args.mse_weight,
                args.aux_weight,
                args.aux_pos_weight,
                args.aux_mse_weight,
                args.val_print_interval,
            )
            best = val_result["best_f1"]
            print(
                "val best_f1 | epoch={} thr={:.2f} precision={:.6f} recall={:.6f} f1={:.6f} val_loss={:.6f}".format(
                    epoch,
                    best["threshold"],
                    best["precision"],
                    best["recall"],
                    best["f1"],
                    val_result["loss"],
                ),
                flush=True,
            )
            append_metrics(metrics_path, epoch, train_metrics, val_result, optimizer.param_groups[0]["lr"], args.aux_weight)
            log_writer.add_scalar("Val/loss", val_result["loss"], epoch)
            log_writer.add_scalar("Val/precision", best["precision"], epoch)
            log_writer.add_scalar("Val/recall", best["recall"], epoch)
            log_writer.add_scalar("Val/f1", best["f1"], epoch)

            if best["f1"] > best_f1:
                best_f1 = best["f1"]
                save_state_dict_atomic(model.state_dict(), model_best_f1_path)
                print(f"saved best_f1: {best_f1:.6f}", flush=True)

            balanced = val_result["best_balanced"]
            balanced_score = min(balanced["precision"], balanced["recall"], balanced["f1"])
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                save_state_dict_atomic(model.state_dict(), model_best_balanced_path)
                print(f"saved best_balanced: score={best_balanced_score:.6f}", flush=True)

            if (
                best["precision"] >= args.target_precision
                and best["recall"] >= args.target_recall
                and best["f1"] >= args.target_f1
            ):
                print(
                    "target reached: precision={:.6f}, recall={:.6f}, f1={:.6f}".format(
                        best["precision"], best["recall"], best["f1"]
                    ),
                    flush=True,
                )
                save_state_dict_atomic(model.state_dict(), model_last_path)
                break

        save_state_dict_atomic(model.state_dict(), model_last_path)
        save_checkpoint_atomic(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "best_balanced_score": best_balanced_score,
                "args": vars(args),
            },
            checkpoint_path,
        )
        if args.snapshot_interval > 0 and (epoch + 1) % args.snapshot_interval == 0:
            snapshot_path = os.path.join(exps_path, f"model_epoch_{epoch + 1:03d}.pt")
            save_state_dict_atomic(model.state_dict(), snapshot_path)
        scheduler.step()
        log_writer.flush()

    print(f"training finished, best_f1={best_f1:.6f}", flush=True)
    log_writer.close()
