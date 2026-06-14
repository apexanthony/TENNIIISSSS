import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TrackNetDatasetV34MultiTask(Dataset):
    def __init__(
        self,
        manifest_csv,
        input_height=270,
        input_width=480,
        heatmap_radius=6,
        heatmap_sigma=2.0,
        court_radius=5,
        court_sigma=2.0,
        tracknet_root="datasets/trackNet/images",
        augment=False,
    ):
        self.data = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.height = input_height
        self.width = input_width
        self.heatmap_radius = heatmap_radius
        self.heatmap_sigma = heatmap_sigma
        self.court_radius = court_radius
        self.court_sigma = court_sigma
        self.tracknet_root = Path(tracknet_root)
        self.augment = augment
        print(f"manifest = {manifest_csv}, samples = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        game = row["game"]
        clip = row["clip"]
        frame_id = int(row["frame_id"])
        current_path = self.tracknet_path(game, clip, frame_id)
        prev1_path = self.tracknet_path(game, clip, frame_id - 1)
        prev2_path = self.tracknet_path(game, clip, frame_id - 2)

        inputs, orig_width, orig_height = self.get_input(
            current_path,
            prev1_path,
            prev2_path,
            fallback_path=str(row["image_path"]),
        )

        visibility = int(row["tracknet_visibility"])
        status = int(row["tracknet_status"])
        ball_x = self.to_float(row["tracknet_x"])
        ball_y = self.to_float(row["tracknet_y"])
        if math.isnan(ball_x) or visibility == 0:
            ball_x = -1.0
            ball_y = -1.0

        ball_heatmap = self.make_heatmap(
            ball_x,
            ball_y,
            visibility,
            orig_width,
            orig_height,
            self.heatmap_radius,
            self.heatmap_sigma,
        )
        player_mask = self.make_player_mask(row.get("player_boxes", ""), orig_width, orig_height)
        court_heatmaps = self.make_court_heatmaps(row, orig_width, orig_height)

        return {
            "input": inputs,
            "ball_heatmap": ball_heatmap,
            "player_mask": player_mask,
            "court_heatmaps": court_heatmaps,
            "status": np.int64(status),
            "visibility": np.int64(visibility),
            "x": np.float32(ball_x),
            "y": np.float32(ball_y),
        }

    def tracknet_path(self, game, clip, frame_id):
        frame_id = max(0, int(frame_id))
        return self.tracknet_root / str(game) / str(clip) / f"{frame_id:04d}.jpg"

    @staticmethod
    def to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def get_input(self, current_path, prev1_path, prev2_path, fallback_path):
        current = self.read_image(current_path)
        if current is None:
            current = self.read_image(fallback_path)
        if current is None:
            raise RuntimeError(f"failed to read current frame: {current_path}")

        prev1 = self.read_image(prev1_path)
        prev2 = self.read_image(prev2_path)
        if prev1 is None:
            prev1 = current
        if prev2 is None:
            prev2 = prev1

        orig_height, orig_width = current.shape[:2]
        frames = [
            cv2.resize(current, (self.width, self.height)),
            cv2.resize(prev1, (self.width, self.height)),
            cv2.resize(prev2, (self.width, self.height)),
        ]
        if self.augment:
            frames = self.apply_augmentations(frames)

        stacked = np.empty((9, self.height, self.width), dtype=np.float16)
        for idx, frame in enumerate(frames):
            chw = frame.transpose(2, 0, 1).astype(np.float16)
            chw *= np.float16(1.0 / 255.0)
            stacked[idx * 3 : (idx + 1) * 3] = chw
        return stacked, orig_width, orig_height

    @staticmethod
    def read_image(path):
        return cv2.imread(str(path)) if path else None

    def make_heatmap(self, x, y, visibility, orig_width, orig_height, radius, sigma):
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        if visibility == 0 or x < 0 or y < 0:
            return heatmap[None, :, :]

        x_model = x * self.width / float(orig_width)
        y_model = y * self.height / float(orig_height)
        sigma2 = 2.0 * sigma * sigma
        cx = int(round(x_model))
        cy = int(round(y_model))
        x0 = max(0, cx - radius)
        x1 = min(self.width, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(self.height, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return heatmap[None, :, :]

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        patch = np.exp(-((xs[None, :] - x_model) ** 2 + (ys[:, None] - y_model) ** 2) / sigma2)
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], patch)
        return heatmap[None, :, :]

    def make_player_mask(self, player_boxes, orig_width, orig_height):
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        if not isinstance(player_boxes, str) or not player_boxes:
            return mask[None, :, :]
        sx = self.width / float(orig_width)
        sy = self.height / float(orig_height)
        for item in player_boxes.split("|"):
            parts = item.split(",")
            if len(parts) != 4:
                continue
            try:
                xmin, ymin, xmax, ymax = [float(v) for v in parts]
            except ValueError:
                continue
            x0 = max(0, min(self.width, int(math.floor(xmin * sx))))
            x1 = max(0, min(self.width, int(math.ceil(xmax * sx))))
            y0 = max(0, min(self.height, int(math.floor(ymin * sy))))
            y1 = max(0, min(self.height, int(math.ceil(ymax * sy))))
            if x0 < x1 and y0 < y1:
                mask[y0:y1, x0:x1] = 1.0
        return mask[None, :, :]

    def make_court_heatmaps(self, row, orig_width, orig_height):
        heatmaps = np.zeros((14, self.height, self.width), dtype=np.float32)
        for idx in range(1, 15):
            value = row.get(f"court_{idx}", "")
            if not isinstance(value, str) or "," not in value:
                continue
            try:
                x, y = [float(v) for v in value.split(",", 1)]
            except ValueError:
                continue
            heatmaps[idx - 1] = self.make_heatmap(
                x,
                y,
                1,
                orig_width,
                orig_height,
                self.court_radius,
                self.court_sigma,
            )[0]
        return heatmaps

    def apply_augmentations(self, frames):
        if np.random.rand() < 0.75:
            alpha = np.random.uniform(0.75, 1.35)
            beta = np.random.uniform(-25.0, 25.0)
            frames = [np.clip(frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8) for frame in frames]
        if np.random.rand() < 0.35:
            ksize = int(np.random.choice([3, 5, 7]))
            angle = np.random.uniform(0, np.pi)
            kernel = self.motion_blur_kernel(ksize, angle)
            frames = [cv2.filter2D(frame, -1, kernel) for frame in frames]
        if np.random.rand() < 0.35:
            quality = int(np.random.randint(35, 90))
            frames = [self.jpeg_compress(frame, quality) for frame in frames]
        if np.random.rand() < 0.20:
            sigma = np.random.uniform(2.0, 8.0)
            frames = [
                np.clip(frame.astype(np.float32) + np.random.normal(0.0, sigma, frame.shape), 0, 255).astype(np.uint8)
                for frame in frames
            ]
        return frames

    @staticmethod
    def motion_blur_kernel(ksize, angle):
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        for i in range(ksize):
            offset = i - center
            x = int(round(center + offset * cos_a))
            y = int(round(center + offset * sin_a))
            if 0 <= x < ksize and 0 <= y < ksize:
                kernel[y, x] = 1.0
        total = float(kernel.sum())
        if total <= 0:
            kernel[center, :] = 1.0
            total = float(kernel.sum())
        return kernel / total

    @staticmethod
    def jpeg_compress(frame, quality):
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return frame
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return frame if decoded is None else decoded
