import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TrackNetDatasetV36Clean(Dataset):
    """Clean reference-style ball heatmap dataset.

    This dataset intentionally trains only the ball heatmap head. Player boxes
    are used only as optional hard-negative masks, not as supervised model heads.
    """

    def __init__(
        self,
        manifest_csv,
        input_height=360,
        input_width=640,
        heatmap_radius=8,
        heatmap_sigma=3.0,
        tracknet_root="datasets/trackNet/images",
        augment=False,
        hardneg_from_players=True,
    ):
        self.data = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.height = int(input_height)
        self.width = int(input_width)
        self.heatmap_radius = int(heatmap_radius)
        self.heatmap_sigma = float(heatmap_sigma)
        self.tracknet_root = Path(tracknet_root)
        self.augment = bool(augment)
        self.hardneg_from_players = bool(hardneg_from_players)
        print(f"manifest={manifest_csv}, samples={len(self.data)}, size={self.width}x{self.height}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        game = str(row["game"])
        clip = str(row["clip"])
        frame_id = int(row["frame_id"])

        current_path = self.tracknet_path(game, clip, frame_id)
        prev1_path = self.tracknet_path(game, clip, frame_id - 1)
        prev2_path = self.tracknet_path(game, clip, frame_id - 2)
        fallback_path = str(row.get("image_path", ""))
        inputs, orig_width, orig_height = self.get_input(current_path, prev1_path, prev2_path, fallback_path)

        visibility = int(float(row.get("tracknet_visibility", 0) or 0))
        x = self.to_float(row.get("tracknet_x", ""))
        y = self.to_float(row.get("tracknet_y", ""))
        if math.isnan(x) or math.isnan(y) or visibility == 0:
            x = self.to_float(row.get("ball_cx", ""))
            y = self.to_float(row.get("ball_cy", ""))
            if not math.isnan(x) and not math.isnan(y):
                visibility = 1
        if math.isnan(x) or math.isnan(y):
            x = -1.0
            y = -1.0
            visibility = 0

        heatmap = self.make_heatmap(x, y, visibility, orig_width, orig_height)
        hardneg_mask = self.make_hardneg_mask(row.get("player_boxes", ""), x, y, visibility, orig_width, orig_height)

        return {
            "input": inputs,
            "ball_heatmap": heatmap,
            "hardneg_mask": hardneg_mask,
            "visibility": np.int64(visibility),
            "x": np.float32(x),
            "y": np.float32(y),
            "clip_key": f"{game}/{clip}",
            "frame_id": np.int64(frame_id),
        }

    def tracknet_path(self, game, clip, frame_id):
        frame_id = max(0, int(frame_id))
        return self.tracknet_root / game / clip / f"{frame_id:04d}.jpg"

    @staticmethod
    def to_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return value

    @staticmethod
    def read_image(path):
        return cv2.imread(str(path)) if path else None

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

    def make_heatmap(self, x, y, visibility, orig_width, orig_height):
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        if visibility == 0 or x < 0 or y < 0:
            return heatmap[None, :, :]

        x_model = x * self.width / float(orig_width)
        y_model = y * self.height / float(orig_height)
        sigma2 = 2.0 * self.heatmap_sigma * self.heatmap_sigma
        cx = int(round(x_model))
        cy = int(round(y_model))
        x0 = max(0, cx - self.heatmap_radius)
        x1 = min(self.width, cx + self.heatmap_radius + 1)
        y0 = max(0, cy - self.heatmap_radius)
        y1 = min(self.height, cy + self.heatmap_radius + 1)
        if x0 >= x1 or y0 >= y1:
            return heatmap[None, :, :]

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        patch = np.exp(-((xs[None, :] - x_model) ** 2 + (ys[:, None] - y_model) ** 2) / sigma2)
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], patch)
        return heatmap[None, :, :]

    def make_hardneg_mask(self, player_boxes, ball_x, ball_y, visibility, orig_width, orig_height):
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        if not self.hardneg_from_players or not isinstance(player_boxes, str) or not player_boxes:
            return mask[None, :, :]

        sx = self.width / float(orig_width)
        sy = self.height / float(orig_height)
        for item in player_boxes.split("|"):
            parts = [p.strip() for p in item.split(",") if p.strip()]
            if len(parts) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in parts]
            except ValueError:
                continue
            px0 = max(0, min(self.width, int(math.floor(x1 * sx))))
            px1 = max(0, min(self.width, int(math.ceil(x2 * sx))))
            py0 = max(0, min(self.height, int(math.floor(y1 * sy))))
            py1 = max(0, min(self.height, int(math.ceil(y2 * sy))))
            if px0 < px1 and py0 < py1:
                mask[py0:py1, px0:px1] = 1.0

        if visibility != 0 and ball_x >= 0 and ball_y >= 0:
            bx = int(round(ball_x * self.width / float(orig_width)))
            by = int(round(ball_y * self.height / float(orig_height)))
            r = max(self.heatmap_radius * 2, 12)
            x0 = max(0, bx - r)
            x1 = min(self.width, bx + r + 1)
            y0 = max(0, by - r)
            y1 = min(self.height, by + r + 1)
            mask[y0:y1, x0:x1] = 0.0
        return mask[None, :, :]

    def apply_augmentations(self, frames):
        if np.random.rand() < 0.80:
            alpha = np.random.uniform(0.70, 1.40)
            beta = np.random.uniform(-30.0, 30.0)
            frames = [np.clip(frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8) for frame in frames]

        if np.random.rand() < 0.45:
            ksize = int(np.random.choice([3, 5, 7, 9]))
            angle = np.random.uniform(0, np.pi)
            kernel = self.motion_blur_kernel(ksize, angle)
            frames = [cv2.filter2D(frame, -1, kernel) for frame in frames]

        if np.random.rand() < 0.40:
            quality = int(np.random.randint(30, 90))
            frames = [self.jpeg_compress(frame, quality) for frame in frames]

        if np.random.rand() < 0.25:
            sigma = np.random.uniform(2.0, 10.0)
            frames = [
                np.clip(frame.astype(np.float32) + np.random.normal(0.0, sigma, frame.shape), 0, 255).astype(np.uint8)
                for frame in frames
            ]

        if np.random.rand() < 0.20:
            frames = self.random_occlusion(frames)
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

    @staticmethod
    def random_occlusion(frames):
        h, w = frames[0].shape[:2]
        occ_w = int(np.random.uniform(0.03, 0.10) * w)
        occ_h = int(np.random.uniform(0.03, 0.10) * h)
        x0 = int(np.random.randint(0, max(1, w - occ_w)))
        y0 = int(np.random.randint(0, max(1, h - occ_h)))
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        out = []
        for frame in frames:
            copied = frame.copy()
            copied[y0 : y0 + occ_h, x0 : x0 + occ_w] = color
            out.append(copied)
        return out
