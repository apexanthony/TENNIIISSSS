import argparse

import torch
from scipy.spatial import distance

from datasets_v2 import trackNetDatasetV2
from general_v3 import postprocess_heatmap
from model_v3 import BallTrackerNetV3


def parse_thresholds(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Sweep peak thresholds for TrackNet V3 validation metrics.")
    parser.add_argument("--model-path", default="./exps/lite_heatmap_v3_180x320/model_best.pt")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-height", type=int, default=180)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--label-height", type=int, default=720)
    parser.add_argument("--label-width", type=int, default=1280)
    parser.add_argument("--heatmap-radius", type=int, default=4)
    parser.add_argument("--heatmap-sigma", type=float, default=1.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95")
    parser.add_argument("--min-dist", type=float, default=8.0)
    parser.add_argument("--peak-window", type=int, default=15)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = parse_thresholds(args.thresholds)
    scale_x = float(args.label_width) / float(args.input_width)
    scale_y = float(args.label_height) / float(args.input_height)

    dataset = trackNetDatasetV2(
        "val",
        input_height=args.input_height,
        input_width=args.input_width,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = BallTrackerNetV3(base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    stats = {thr: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "visible": 0} for thr in thresholds}
    with torch.no_grad():
        for iter_id, batch in enumerate(loader):
            logits = model(batch[0].float().to(device))
            heatmaps = torch.sigmoid(logits).detach().cpu().numpy()
            for i in range(heatmaps.shape[0]):
                x_gt = float(batch[2][i])
                y_gt = float(batch[3][i])
                vis = int(batch[4][i])
                for thr in thresholds:
                    x_pred, y_pred = postprocess_heatmap(
                        heatmaps[i, 0],
                        threshold=thr,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        peak_window=args.peak_window,
                    )
                    s = stats[thr]
                    if vis != 0:
                        s["visible"] += 1
                    if x_pred is not None:
                        if vis != 0:
                            dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                            if dst < args.min_dist:
                                s["tp"] += 1
                            else:
                                s["fp"] += 1
                        else:
                            s["fp"] += 1
                    else:
                        if vis != 0:
                            s["fn"] += 1
                        else:
                            s["tn"] += 1
            if iter_id % 100 == 0:
                print("processed batch {}/{}".format(iter_id, len(loader)), flush=True)

    print("threshold,precision,recall,f1,tp,fp,tn,fn")
    best = None
    for thr in thresholds:
        s = stats[thr]
        eps = 1e-15
        precision = s["tp"] / (s["tp"] + s["fp"] + eps)
        recall = s["tp"] / (s["visible"] + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        print(
            "{:.3f},{:.6f},{:.6f},{:.6f},{},{},{},{}".format(
                thr, precision, recall, f1, s["tp"], s["fp"], s["tn"], s["fn"]
            )
        )
        if best is None or f1 > best[3]:
            best = (thr, precision, recall, f1)
    print("best_threshold={:.3f}, precision={:.6f}, recall={:.6f}, f1={:.6f}".format(*best))


if __name__ == "__main__":
    main()
