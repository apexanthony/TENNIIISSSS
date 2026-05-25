import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import distance


def heatmap_loss(logits, targets, pos_weight=80.0, mse_weight=1.0):
    pos = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos)
    mse = F.mse_loss(torch.sigmoid(logits), targets)
    return bce + mse_weight * mse


def train_v2(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    max_iters=200,
    pos_weight=80.0,
    mse_weight=1.0,
    print_interval=20,
    scaler=None,
):
    start_time = time.time()
    losses = []
    for iter_id, batch in enumerate(train_loader):
        if iter_id >= max_iters:
            break
        optimizer.zero_grad()
        model.train()
        inp = batch[0].float().to(device)
        gt = batch[1].float().to(device)
        use_amp = scaler is not None and str(device).startswith("cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            out = model(inp)
            loss = heatmap_loss(out, gt, pos_weight=pos_weight, mse_weight=mse_weight)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        end_time = time.time()
        duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        losses.append(loss.item())
        if print_interval > 0 and (iter_id % print_interval == 0 or iter_id + 1 >= max_iters):
            print(
                "train_v2 | epoch = {}, iter = [{}|{}], loss = {}, avg_loss = {}, time = {}".format(
                    epoch, iter_id, max_iters, round(loss.item(), 6), round(float(np.mean(losses)), 6), duration
                )
            )

    return np.mean(losses)


def validate_v2(
    model,
    val_loader,
    device,
    epoch,
    min_dist=5,
    threshold=0.35,
    pos_weight=80.0,
    mse_weight=1.0,
    print_interval=100,
):
    losses = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    model.eval()
    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            inp = batch[0].float().to(device)
            gt = batch[1].float().to(device)
            out = model(inp)
            loss = heatmap_loss(out, gt, pos_weight=pos_weight, mse_weight=mse_weight)
            losses.append(loss.item())

            heatmaps = torch.sigmoid(out).detach().cpu().numpy()
            for i in range(len(heatmaps)):
                x_pred, y_pred = postprocess_heatmap(heatmaps[i, 0], threshold=threshold)
                x_gt = float(batch[2][i])
                y_gt = float(batch[3][i])
                vis = int(batch[4][i])

                if x_pred is not None:
                    if vis != 0:
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < min_dist:
                            tp[vis] += 1
                        else:
                            fp[vis] += 1
                    else:
                        fp[vis] += 1
                else:
                    if vis != 0:
                        fn[vis] += 1
                    else:
                        tn[vis] += 1

            if print_interval > 0 and (iter_id % print_interval == 0 or iter_id == len(val_loader) - 1):
                print(
                    "val_v2 | epoch = {}, iter = [{}|{}], loss = {}, tp = {}, tn = {}, fp = {}, fn = {} ".format(
                        epoch,
                        iter_id,
                        len(val_loader),
                        round(np.mean(losses), 6),
                        sum(tp),
                        sum(tn),
                        sum(fp),
                        sum(fn),
                    )
                )

    eps = 1e-15
    precision = sum(tp) / (sum(tp) + sum(fp) + eps)
    vc1 = tp[1] + fp[1] + tn[1] + fn[1]
    vc2 = tp[2] + fp[2] + tn[2] + fn[2]
    vc3 = tp[3] + fp[3] + tn[3] + fn[3]
    recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    print("precision = {}".format(precision))
    print("recall = {}".format(recall))
    print("f1 = {}".format(f1))

    return np.mean(losses), precision, recall, f1


def postprocess_heatmap(heatmap, threshold=0.35, scale_x=2.0, scale_y=2.0, peak_window=15):
    if heatmap.ndim != 2:
        heatmap = heatmap.reshape((360, 640))

    heatmap = np.asarray(heatmap, dtype=np.float32)
    _, max_value, _, max_loc = cv2.minMaxLoc(heatmap)
    if max_value < threshold:
        return None, None

    peak_window = max(3, int(peak_window))
    if peak_window % 2 == 0:
        peak_window += 1
    radius = peak_window // 2
    peak_x, peak_y = max_loc
    x0 = max(0, peak_x - radius)
    x1 = min(heatmap.shape[1], peak_x + radius + 1)
    y0 = max(0, peak_y - radius)
    y1 = min(heatmap.shape[0], peak_y + radius + 1)

    crop = heatmap[y0:y1, x0:x1]
    weights = np.maximum(crop - threshold, 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return float(peak_x) * scale_x, float(peak_y) * scale_y

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    x_model = float(np.sum(weights * xs[None, :]) / weight_sum)
    y_model = float(np.sum(weights * ys[:, None]) / weight_sum)
    return x_model * scale_x, y_model * scale_y
