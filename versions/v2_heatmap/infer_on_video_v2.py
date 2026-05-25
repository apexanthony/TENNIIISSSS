import argparse
from itertools import groupby

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from general_v2 import postprocess_heatmap
from model_v2 import BallTrackerNetLite


MODEL_HEIGHT = 360
MODEL_WIDTH = 640


def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps


def infer_model(frames, model, device, threshold=0.35, peak_window=15):
    dists = [-1] * 2
    ball_track = [(None, None)] * 2
    for num in tqdm(range(2, len(frames))):
        height, width = frames[num].shape[:2]
        img = cv2.resize(frames[num], (MODEL_WIDTH, MODEL_HEIGHT))
        img_prev = cv2.resize(frames[num - 1], (MODEL_WIDTH, MODEL_HEIGHT))
        img_preprev = cv2.resize(frames[num - 2], (MODEL_WIDTH, MODEL_HEIGHT))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        with torch.no_grad():
            logits = model(torch.from_numpy(inp).float().to(device))
            heatmap = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        x_pred, y_pred = postprocess_heatmap(
            heatmap,
            threshold=threshold,
            scale_x=width / float(MODEL_WIDTH),
            scale_y=height / float(MODEL_HEIGHT),
            peak_window=peak_window,
        )
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] is not None and ball_track[-2][0] is not None:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)
    return ball_track, dists


def remove_outliers(ball_track, dists, max_dist=100):
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if i + 1 < len(dists) and ((dists[i + 1] > max_dist) or (dists[i + 1] == -1)):
            ball_track[i] = (None, None)
        elif i - 1 >= 0 and dists[i - 1] == -1:
            ball_track[i - 1] = (None, None)
    return ball_track


def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    list_det = [0 if x[0] is not None else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, length) in enumerate(groups):
        if (k == 1) and (i > 0) and (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + length])
            if (length >= max_gap) or (dist / length > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + length - 1
        cursor += length
    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])
    return result


def interpolation(coords):
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])

    return [*zip(x, y)]


def write_track(frames, ball_track, path_output_video, fps, trace=7):
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height))
    for num in range(len(frames)):
        frame = frames[num]
        for i in range(trace):
            if num - i > 0:
                if ball_track[num - i][0] is not None:
                    x = int(ball_track[num - i][0])
                    y = int(ball_track[num - i][1])
                    frame = cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=10 - i)
                else:
                    break
        out.write(frame)
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TrackNet Lite V2 heatmap inference on video.")
    parser.add_argument("--model_path", type=str, required=True, help="path to model")
    parser.add_argument("--video_path", type=str, required=True, help="path to input video")
    parser.add_argument("--video_out_path", type=str, required=True, help="path to output video")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--peak_window", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--extrapolation", action="store_true", help="whether to use ball track extrapolation")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BallTrackerNetLite(base_channels=args.base_channels)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    frames, fps = read_video(args.video_path)
    ball_track, dists = infer_model(frames, model, device, threshold=args.threshold, peak_window=args.peak_window)
    ball_track = remove_outliers(ball_track, dists)

    if args.extrapolation:
        subtracks = split_track(ball_track)
        for r in subtracks:
            ball_subtrack = ball_track[r[0] : r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_track[r[0] : r[1]] = ball_subtrack

    write_track(frames, ball_track, args.video_out_path, fps)
