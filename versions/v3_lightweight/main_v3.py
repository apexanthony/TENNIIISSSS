import argparse
import os

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from datasets_v2 import trackNetDatasetV2
from general_v3 import save_state_dict_atomic, train_v3, validate_v3
from model_v3 import BallTrackerNetV3


def save_checkpoint_atomic(checkpoint, path):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrackNet V3 depthwise heatmap model.")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--exp_id", type=str, default="lite_heatmap_v3_180x320", help="path to saving results")
    parser.add_argument("--num_epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--val_intervals", type=int, default=5, help="number of epochs to run validation")
    parser.add_argument("--steps_per_epoch", type=int, default=200, help="number of steps per one epoch")
    parser.add_argument("--base_channels", type=int, default=24, help="V3 base channel count")
    parser.add_argument("--input_height", type=int, default=180)
    parser.add_argument("--input_width", type=int, default=320)
    parser.add_argument("--label_height", type=int, default=720)
    parser.add_argument("--label_width", type=int, default=1280)
    parser.add_argument("--heatmap_radius", type=int, default=4)
    parser.add_argument("--heatmap_sigma", type=float, default=1.5)
    parser.add_argument("--pos_weight", type=float, default=80.0, help="positive weight for BCE heatmap loss")
    parser.add_argument("--mse_weight", type=float, default=1.0, help="MSE auxiliary loss weight")
    parser.add_argument("--threshold", type=float, default=0.50, help="validation peak threshold")
    parser.add_argument("--peak_window", type=int, default=15, help="validation peak refinement window")
    parser.add_argument("--min_dist", type=float, default=8.0, help="validation hit distance in label pixels")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; use 0 on restricted Windows")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint for continuing training")
    parser.add_argument("--start_epoch", type=int, default=0, help="first epoch index when resuming")
    parser.add_argument("--best_metric", type=float, default=0.0, help="existing best F1 when resuming")
    parser.add_argument("--print_interval", type=int, default=20, help="training log interval")
    parser.add_argument("--val_print_interval", type=int, default=100, help="validation log interval")
    parser.add_argument("--snapshot_interval", type=int, default=25, help="save model_epoch_XXX.pt every N epochs")
    parser.add_argument("--augment", action="store_true", help="enable train-time brightness/blur/compression augmentation")
    parser.add_argument("--amp", action="store_true", help="enable CUDA mixed precision training")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu; default auto")
    parser.add_argument("--target_precision", type=float, default=0.0, help="stop once validation precision reaches this")
    parser.add_argument("--target_recall", type=float, default=0.0, help="stop once validation recall reaches this")
    parser.add_argument("--target_f1", type=float, default=0.0, help="stop once validation F1 reaches this")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = trackNetDatasetV2(
        "train",
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        augment=args.augment,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    val_dataset = trackNetDatasetV2(
        "val",
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
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
    if args.resume:
        print("loading checkpoint: {}".format(args.resume), flush=True)
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            resume_checkpoint = checkpoint
            if args.start_epoch == 0:
                args.start_epoch = int(checkpoint.get("epoch", -1)) + 1
            if args.best_metric == 0.0:
                args.best_metric = float(checkpoint.get("best_f1", 0.0))
        else:
            model.load_state_dict(checkpoint)

    exps_path = "./exps/{}".format(args.exp_id)
    tb_path = os.path.join(exps_path, "plots")
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    try:
        log_writer = SummaryWriter(tb_path)
    except Exception as exc:
        print("warning: tensorboard writer disabled: {}".format(exc), flush=True)
        log_writer = NullWriter()

    model_last_path = os.path.join(exps_path, "model_last.pt")
    model_best_path = os.path.join(exps_path, "model_best.pt")
    checkpoint_path = os.path.join(exps_path, "training_state.pt")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.num_epochs - args.start_epoch),
        eta_min=args.lr * 0.05,
    )
    if resume_checkpoint is not None:
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    scaler = torch.amp.GradScaler("cuda") if args.amp and device == "cuda" else None
    val_best_metric = args.best_metric

    print(
        "V3 config: input={}x{}, base_channels={}, batch_size={}, device={}, amp={}".format(
            args.input_height,
            args.input_width,
            args.base_channels,
            args.batch_size,
            device,
            scaler is not None,
        ),
        flush=True,
    )

    for epoch in range(args.start_epoch, args.num_epochs):
        train_loss = train_v3(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.steps_per_epoch,
            pos_weight=args.pos_weight,
            mse_weight=args.mse_weight,
            print_interval=args.print_interval,
            scaler=scaler,
        )
        print("train loss = {}".format(train_loss), flush=True)
        log_writer.add_scalar("Train/training_loss", train_loss, epoch)
        log_writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

        should_validate = (epoch > 0) and (epoch % args.val_intervals == 0)
        if should_validate:
            val_loss, precision, recall, f1 = validate_v3(
                model,
                val_loader,
                device,
                epoch,
                input_width=args.input_width,
                input_height=args.input_height,
                label_width=args.label_width,
                label_height=args.label_height,
                min_dist=args.min_dist,
                threshold=args.threshold,
                peak_window=args.peak_window,
                pos_weight=args.pos_weight,
                mse_weight=args.mse_weight,
                print_interval=args.val_print_interval,
            )
            print("val loss = {}".format(val_loss), flush=True)
            log_writer.add_scalar("Val/loss", val_loss, epoch)
            log_writer.add_scalar("Val/precision", precision, epoch)
            log_writer.add_scalar("Val/recall", recall, epoch)
            log_writer.add_scalar("Val/f1", f1, epoch)
            if f1 > val_best_metric:
                val_best_metric = f1
                save_state_dict_atomic(model.state_dict(), model_best_path)
                print("saved new best: f1 = {}".format(f1), flush=True)

            if (
                args.target_precision > 0.0
                and args.target_recall > 0.0
                and args.target_f1 > 0.0
                and precision >= args.target_precision
                and recall >= args.target_recall
                and f1 >= args.target_f1
            ):
                print(
                    "target reached: precision = {}, recall = {}, f1 = {}".format(
                        precision, recall, f1
                    ),
                    flush=True,
                )
                save_state_dict_atomic(model.state_dict(), model_last_path)
                save_checkpoint_atomic(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_f1": val_best_metric,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
                log_writer.flush()
                log_writer.close()
                raise SystemExit(0)

        save_state_dict_atomic(model.state_dict(), model_last_path)
        save_checkpoint_atomic(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": val_best_metric,
                "args": vars(args),
            },
            checkpoint_path,
        )
        if args.snapshot_interval > 0 and (epoch > 0) and (epoch % args.snapshot_interval == 0):
            snapshot_path = os.path.join(exps_path, "model_epoch_{:03d}.pt".format(epoch))
            save_state_dict_atomic(model.state_dict(), snapshot_path)

        scheduler.step()
        log_writer.flush()

    print("training finished, best_f1 = {}".format(val_best_metric), flush=True)
    log_writer.close()
