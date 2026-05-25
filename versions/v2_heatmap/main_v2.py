import argparse
import os

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from datasets_v2 import trackNetDatasetV2
from general_v2 import train_v2, validate_v2
from model_v2 import BallTrackerNetLite


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrackNet Lite V2 with 1-channel heatmap output.")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--exp_id", type=str, default="lite_heatmap_v2", help="path to saving results")
    parser.add_argument("--num_epochs", type=int, default=300, help="total training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--val_intervals", type=int, default=5, help="number of epochs to run validation")
    parser.add_argument("--steps_per_epoch", type=int, default=200, help="number of steps per one epoch")
    parser.add_argument("--base_channels", type=int, default=32, help="lite model base channel count")
    parser.add_argument("--pos_weight", type=float, default=80.0, help="positive weight for BCE heatmap loss")
    parser.add_argument("--mse_weight", type=float, default=1.0, help="MSE auxiliary loss weight")
    parser.add_argument("--threshold", type=float, default=0.35, help="validation peak threshold")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; use 0 on restricted Windows")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint for continuing training")
    parser.add_argument("--start_epoch", type=int, default=0, help="first epoch index when resuming")
    parser.add_argument("--best_metric", type=float, default=0.0, help="existing best F1 when resuming")
    parser.add_argument("--print_interval", type=int, default=20, help="training log interval")
    parser.add_argument("--val_print_interval", type=int, default=100, help="validation log interval")
    parser.add_argument("--augment", action="store_true", help="enable train-time brightness/blur/compression augmentation")
    parser.add_argument("--amp", action="store_true", help="enable CUDA mixed precision training")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu; default auto")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = trackNetDatasetV2("train", augment=args.augment)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    val_dataset = trackNetDatasetV2("val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    model = BallTrackerNetLite(base_channels=args.base_channels).to(device)
    if args.resume:
        print("loading checkpoint: {}".format(args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    exps_path = "./exps/{}".format(args.exp_id)
    tb_path = os.path.join(exps_path, "plots")
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    try:
        log_writer = SummaryWriter(tb_path)
    except Exception as exc:
        print("warning: tensorboard writer disabled: {}".format(exc))
        log_writer = NullWriter()
    model_last_path = os.path.join(exps_path, "model_last.pt")
    model_best_path = os.path.join(exps_path, "model_best.pt")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device == "cuda" else None
    val_best_metric = args.best_metric

    for epoch in range(args.start_epoch, args.num_epochs):
        train_loss = train_v2(
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
        print("train loss = {}".format(train_loss))
        log_writer.add_scalar("Train/training_loss", train_loss, epoch)
        log_writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

        if (epoch > 0) and (epoch % args.val_intervals == 0):
            val_loss, precision, recall, f1 = validate_v2(
                model,
                val_loader,
                device,
                epoch,
                threshold=args.threshold,
                pos_weight=args.pos_weight,
                mse_weight=args.mse_weight,
                print_interval=args.val_print_interval,
            )
            print("val loss = {}".format(val_loss))
            log_writer.add_scalar("Val/loss", val_loss, epoch)
            log_writer.add_scalar("Val/precision", precision, epoch)
            log_writer.add_scalar("Val/recall", recall, epoch)
            log_writer.add_scalar("Val/f1", f1, epoch)
            if f1 > val_best_metric:
                val_best_metric = f1
                torch.save(model.state_dict(), model_best_path)
            torch.save(model.state_dict(), model_last_path)
        else:
            torch.save(model.state_dict(), model_last_path)
