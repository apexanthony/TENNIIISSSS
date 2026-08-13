import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


V38_DIR = Path(__file__).resolve().parent
V37_DIR = V38_DIR.parent / "v3_7_tracknet_ball_hardneg"
if str(V37_DIR) not in sys.path:
    sys.path.insert(0, str(V37_DIR))

from general_v3 import heatmap_loss, postprocess_heatmap  # noqa: E402

from dataset_v38 import TrackNetDatasetV38
from model_v38 import BallTrackerNetV38


VARIANTS = {
    "baseline": {"use_ca": False, "use_hardneg": False, "use_aux": False},
    "ca": {"use_ca": True, "use_hardneg": False, "use_aux": False},
    "hardneg": {"use_ca": False, "use_hardneg": True, "use_aux": False},
    "aux": {"use_ca": False, "use_hardneg": False, "use_aux": True},
    "full": {"use_ca": True, "use_hardneg": True, "use_aux": False},
}


def parse_thresholds(value):
    if isinstance(value, list):
        return value
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def capture_rng_state(generator):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, generator):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_torch_atomic(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def hard_negative_loss(logits, targets, hardneg_mask):
    probability = torch.sigmoid(logits)
    active = hardneg_mask * (targets < 0.05).to(logits.dtype)
    denominator = torch.clamp(active.sum(), min=1.0)
    return ((probability.square()) * active).sum() / denominator


def auxiliary_positive_loss(logits, target, valid, pos_weight, mse_weight):
    valid = valid.to(logits.dtype).view(-1, 1, 1, 1)
    denominator = torch.clamp(valid.sum(), min=1.0)
    positive_weight = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight, reduction="none")
    mse = F.mse_loss(torch.sigmoid(logits), target, reduction="none")
    per_sample = (bce + mse_weight * mse).mean(dim=(1, 2, 3), keepdim=True)
    return (per_sample * valid).sum() / denominator


def move_batch(raw, device):
    return {
        "input": raw["input"].float().to(device, non_blocking=True),
        "target": raw["ball_heatmap"].float().to(device, non_blocking=True),
        "hardneg": raw["hardneg_mask"].float().to(device, non_blocking=True),
        "aux": raw["aux_heatmap"].float().to(device, non_blocking=True),
        "aux_valid": raw["aux_valid"].float().to(device, non_blocking=True),
        "visibility": raw["visibility"],
        "x": raw["x"],
        "y": raw["y"],
        "orig_width": raw["orig_width"],
        "orig_height": raw["orig_height"],
        "clip_key": raw["clip_key"],
        "frame_id": raw["frame_id"],
    }


def compute_losses(logits, batch, args, variant):
    heatmap = heatmap_loss(logits, batch["target"], pos_weight=args.pos_weight, mse_weight=args.mse_weight)
    hardneg = logits.new_zeros(())
    auxiliary = logits.new_zeros(())
    total = heatmap
    if variant["use_hardneg"]:
        hardneg = hard_negative_loss(logits, batch["target"], batch["hardneg"])
        total = total + args.hardneg_weight * hardneg
    if variant["use_aux"]:
        auxiliary = auxiliary_positive_loss(
            logits,
            batch["aux"],
            batch["aux_valid"],
            args.aux_pos_weight,
            args.aux_mse_weight,
        )
        total = total + args.aux_weight * auxiliary
    return total, heatmap, hardneg, auxiliary


def train_epoch(model, loader, optimizer, scaler, device, args, variant, epoch):
    model.train()
    totals = {name: 0.0 for name in ("loss", "heatmap", "hardneg", "aux")}
    seen = 0
    max_steps = len(loader) if args.steps_per_epoch <= 0 else min(len(loader), args.steps_per_epoch)
    for step, raw in enumerate(loader):
        if step >= max_steps:
            break
        batch = move_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(batch["input"])
            loss, heatmap, hardneg, auxiliary = compute_losses(logits, batch, args, variant)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = int(batch["input"].shape[0])
        seen += batch_size
        for name, value in (("loss", loss), ("heatmap", heatmap), ("hardneg", hardneg), ("aux", auxiliary)):
            totals[name] += float(value.detach().item()) * batch_size
        if args.print_interval > 0 and (step % args.print_interval == 0 or step + 1 == max_steps):
            print(
                f"train | epoch={epoch} step={step + 1}/{max_steps} "
                f"loss={totals['loss']/seen:.6f} hm={totals['heatmap']/seen:.6f} "
                f"hn={totals['hardneg']/seen:.6f} aux={totals['aux']/seen:.6f}",
                flush=True,
            )
    return {name: value / max(seen, 1) for name, value in totals.items()}


def new_counts():
    return {"tp": 0, "wrong_localization": 0, "missed_visible": 0, "fp_background": 0, "tn": 0}


def metrics_from_counts(counts):
    tp = counts["tp"]
    wrong = counts["wrong_localization"]
    missed = counts["missed_visible"]
    fp_background = counts["fp_background"]
    tn = counts["tn"]
    total = tp + wrong + missed + fp_background + tn
    visible = tp + wrong + missed
    fp = wrong + fp_background
    precision = tp / max(tp + fp, 1)
    recall = tp / max(visible, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(total, 1)
    return {
        **counts,
        "total": total,
        "visible": visible,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def validate(model, loader, device, args, variant, epoch):
    model.eval()
    thresholds = args.thresholds
    counts = {threshold: new_counts() for threshold in thresholds}
    loss_sums = {name: 0.0 for name in ("loss", "heatmap", "hardneg", "aux")}
    seen = 0
    with torch.inference_mode():
        for step, raw in enumerate(loader):
            if args.max_val_batches > 0 and step >= args.max_val_batches:
                break
            batch = move_batch(raw, device)
            logits = model(batch["input"])
            loss, heatmap, hardneg, auxiliary = compute_losses(logits, batch, args, variant)
            batch_size = int(batch["input"].shape[0])
            seen += batch_size
            for name, value in (("loss", loss), ("heatmap", heatmap), ("hardneg", hardneg), ("aux", auxiliary)):
                loss_sums[name] += float(value.item()) * batch_size
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for index in range(batch_size):
                visible = int(batch["visibility"][index]) != 0
                gt_x = float(batch["x"][index]) * args.input_width / float(batch["orig_width"][index])
                gt_y = float(batch["y"][index]) * args.input_height / float(batch["orig_height"][index])
                for threshold in thresholds:
                    pred_x, pred_y = postprocess_heatmap(
                        probabilities[index, 0],
                        threshold=threshold,
                        scale_x=1.0,
                        scale_y=1.0,
                        peak_window=args.peak_window,
                    )
                    item = counts[threshold]
                    if pred_x is None:
                        item["missed_visible" if visible else "tn"] += 1
                    elif not visible:
                        item["fp_background"] += 1
                    else:
                        distance = float(np.hypot(pred_x - gt_x, pred_y - gt_y))
                        item["tp" if distance <= args.match_radius else "wrong_localization"] += 1
            if args.val_print_interval > 0 and step % args.val_print_interval == 0:
                print(f"valid | epoch={epoch} step={step + 1}/{len(loader)}", flush=True)
    rows = []
    for threshold in thresholds:
        rows.append({"threshold": threshold, **metrics_from_counts(counts[threshold])})
    best = max(rows, key=lambda row: (row["f1"], row["precision"], row["recall"], row["threshold"]))
    losses = {name: value / max(seen, 1) for name, value in loss_sums.items()}
    return {"losses": losses, "rows": rows, "best": best}


def append_epoch_metrics(path, epoch, learning_rate, train_metrics, validation):
    path = Path(path)
    fields = [
        "epoch", "lr", "train_loss", "train_heatmap", "train_hardneg", "train_aux",
        "val_loss", "val_heatmap", "val_hardneg", "val_aux", "threshold",
        "accuracy", "precision", "recall", "f1", "tp", "wrong_localization",
        "missed_visible", "fp_background", "tn", "total", "visible",
    ]
    exists = path.exists()
    best = validation["best"]
    row = {
        "epoch": epoch,
        "lr": learning_rate,
        **{f"train_{name}": value for name, value in train_metrics.items()},
        **{f"val_{name}": value for name, value in validation["losses"].items()},
        **best,
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def make_dataset(csv_path, args, augment):
    return TrackNetDatasetV38(
        csv_path,
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        tracknet_root=args.tracknet_root,
        mapped_csv=args.hardneg_mapped_csv,
        aux_mapped_csv=args.aux_mapped_csv,
        aux_heatmap_radius=args.aux_heatmap_radius,
        aux_heatmap_sigma=args.aux_heatmap_sigma,
        augment=augment,
        rgb_input=True,
        hardneg_player_weight=args.hardneg_player_weight,
        hardneg_shoe_weight=args.hardneg_shoe_weight,
        hardneg_court_weight=args.hardneg_court_weight,
        hardneg_bright_weight=args.hardneg_bright_weight,
        hardneg_edge_weight=args.hardneg_edge_weight,
        hardneg_ball_clear_radius=args.hardneg_ball_clear_radius,
        augment_min_ball_contrast=args.augment_min_ball_contrast,
        augment_min_contrast_ratio=args.augment_min_contrast_ratio,
    )


def main():
    parser = argparse.ArgumentParser(description="Unified one-seed V3.8 paper ablation trainer.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--train-csv", default="datasets/tracknet_v38_match_split/train.csv")
    parser.add_argument("--valid-csv", default="datasets/tracknet_v38_match_split/valid.csv")
    parser.add_argument("--tracknet-root", default="datasets/trackNet")
    parser.add_argument("--output-root", default="exps/v38_essay_ablation")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", default="", help="training_state.pt to resume; empty starts from scratch")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--ca-reduction", type=int, default=32)
    parser.add_argument("--ca-min-channels", type=int, default=8)
    parser.add_argument("--heatmap-radius", type=int, default=5)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--aux-heatmap-radius", type=int, default=5)
    parser.add_argument("--aux-heatmap-sigma", type=float, default=1.8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-start-lr", type=float, default=1e-5)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--pos-weight", type=float, default=120.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--hardneg-weight", type=float, default=0.08)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--aux-pos-weight", type=float, default=40.0)
    parser.add_argument("--aux-mse-weight", type=float, default=1.0)
    parser.add_argument("--hardneg-mapped-csv", default="datasets/tennis_all_v4i_mapped/annotations_hardneg_cleaned_strict.csv")
    parser.add_argument("--aux-mapped-csv", default="datasets/tennis_all_v4i_mapped/annotations_hardneg_cleaned_strict.csv")
    parser.add_argument("--hardneg-player-weight", type=float, default=1.0)
    parser.add_argument("--hardneg-shoe-weight", type=float, default=1.4)
    parser.add_argument("--hardneg-court-weight", type=float, default=0.85)
    parser.add_argument("--hardneg-bright-weight", type=float, default=0.35)
    parser.add_argument("--hardneg-edge-weight", type=float, default=0.18)
    parser.add_argument("--hardneg-ball-clear-radius", type=int, default=18)
    parser.add_argument("--augment-min-ball-contrast", type=float, default=4.0)
    parser.add_argument("--augment-min-contrast-ratio", type=float, default=0.55)
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95"))
    parser.add_argument("--peak-window", type=int, default=9)
    parser.add_argument("--match-radius", type=float, default=4.0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pin DataLoader batches in host memory; disabled by default for low-memory Windows hosts.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--print-interval", type=int, default=100)
    parser.add_argument("--val-print-interval", type=int, default=100)
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    seed_everything(args.seed, args.deterministic)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_name = args.run_name or f"{args.variant}_seed{args.seed}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "thresholds": args.thresholds, "variant_flags": variant, "device": str(device)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_set = make_dataset(args.train_csv, args, augment=args.augment)
    valid_set = make_dataset(args.valid_csv, args, augment=False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.pin_memory and device.type == "cuda",
    )
    model = BallTrackerNetV38(
        base_channels=args.base_channels,
        use_ca=variant["use_ca"],
        ca_reduction=args.ca_reduction,
        ca_min_channels=args.ca_min_channels,
        initialization_seed=args.seed,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"variant={args.variant} flags={variant} parameters={parameter_count} device={device}", flush=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    best_f1 = -1.0
    no_improvement = 0
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_variant = checkpoint.get("args", {}).get("variant")
        if saved_variant and saved_variant != args.variant:
            raise ValueError(f"resume variant mismatch: checkpoint={saved_variant}, requested={args.variant}")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler is not None and checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint.get("rng_state"), generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("best_f1", -1.0))
        no_improvement = int(checkpoint.get("no_improvement", 0))
        print(f"resumed={args.resume} start_epoch={start_epoch} best_f1={best_f1:.6f}", flush=True)
    metrics_path = run_dir / "metrics.csv"
    for epoch in range(start_epoch, args.epochs):
        if epoch < args.warmup_epochs:
            ratio = float(epoch + 1) / max(args.warmup_epochs, 1)
            learning_rate = args.warmup_start_lr + (args.lr - args.warmup_start_lr) * ratio
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, args, variant, epoch)
        validation = validate(model, valid_loader, device, args, variant, epoch)
        append_epoch_metrics(metrics_path, epoch, learning_rate, train_metrics, validation)
        best = validation["best"]
        print(
            f"valid best | epoch={epoch} threshold={best['threshold']:.2f} "
            f"acc={best['accuracy']:.6f} p={best['precision']:.6f} "
            f"r={best['recall']:.6f} f1={best['f1']:.6f}",
            flush=True,
        )
        if best["f1"] > best_f1:
            best_f1 = best["f1"]
            no_improvement = 0
            save_torch_atomic(model.state_dict(), run_dir / "model_best.pt")
            (run_dir / "best_validation.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        else:
            no_improvement += 1
        if epoch >= args.warmup_epochs:
            scheduler.step(best["f1"])
        save_torch_atomic(
            {
                "epoch": epoch,
                "best_f1": best_f1,
                "no_improvement": no_improvement,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "rng_state": capture_rng_state(generator),
                "args": config,
            },
            run_dir / "training_state.pt",
        )
        if args.early_stop_patience > 0 and no_improvement >= args.early_stop_patience:
            print(f"early stop after {no_improvement} epochs without improvement", flush=True)
            break


if __name__ == "__main__":
    main()
