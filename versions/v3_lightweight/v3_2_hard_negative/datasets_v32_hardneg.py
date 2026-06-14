import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class TrackNetDatasetV32HardNeg(Dataset):
    def __init__(
        self,
        mode,
        input_height=270,
        input_width=480,
        heatmap_radius=6,
        heatmap_sigma=2.0,
        hardneg_player_weight=1.0,
        hardneg_court_weight=0.7,
        court_radius=8,
        augment=False,
        mapped_csv="datasets/tennis_all_v4i_mapped/annotations_mapped.csv",
    ):
        self.path_dataset = "./datasets/trackNet"
        assert mode in ["train", "val"], "incorrect mode"
        labels_path = os.path.join(self.path_dataset, f"labels_{mode}.csv")
        with open(labels_path, newline="", encoding="utf-8-sig") as f:
            self.data = list(csv.DictReader(f))
        print(f"mode = {mode}, samples = {len(self.data)}")

        self.height = input_height
        self.width = input_width
        self.heatmap_radius = heatmap_radius
        self.heatmap_sigma = heatmap_sigma
        self.hardneg_player_weight = hardneg_player_weight
        self.hardneg_court_weight = hardneg_court_weight
        self.court_radius = court_radius
        self.augment = augment and mode == "train"
        self.mapped = self.load_mapped_annotations(mapped_csv)
        print(f"mapped hardneg frames = {len(self.mapped)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        path = row.get("path1") or row.get("path")
        path_prev = row.get("path2") or row.get("path_prev")
        path_preprev = row.get("path3") or row.get("path_preprev")
        x = self._to_float(row.get("x-coordinate", row.get("x", "")))
        y = self._to_float(row.get("y-coordinate", row.get("y", "")))
        vis = int(float(row.get("visibility", row.get("vis", 0)) or 0))

        full_path = os.path.join(self.path_dataset, path)
        full_path_prev = os.path.join(self.path_dataset, path_prev)
        full_path_preprev = os.path.join(self.path_dataset, path_preprev)
        if math.isnan(x):
            x = -1
            y = -1

        inputs, orig_width, orig_height = self.get_input(full_path, full_path_prev, full_path_preprev)
        heatmap = self.make_heatmap(x, y, vis, orig_width, orig_height)
        hardneg = self.make_hardneg_mask(path, orig_width, orig_height, heatmap[0])

        return inputs, heatmap, x, y, vis, hardneg

    @staticmethod
    def _to_float(value):
        if value is None or value == "":
            return float("nan")
        return float(value)

    @staticmethod
    def parse_frame_path(path):
        parts = Path(path).parts
        # images/game9/Clip5/0090.jpg
        if len(parts) < 4:
            return None
        return parts[-3], parts[-2], parts[-1]

    @classmethod
    def load_mapped_annotations(cls, mapped_csv):
        mapped = {}
        mapped_path = Path(mapped_csv)
        if not mapped_path.exists():
            print(f"warning: mapped csv not found: {mapped_csv}")
            return mapped
        with mapped_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                key = (row["game"], row["clip"], row["frame_name"])
                mapped[key] = row
        return mapped

    def get_input(self, path, path_prev, path_preprev):
        img = cv2.imread(path)
        img_prev = cv2.imread(path_prev)
        img_preprev = cv2.imread(path_preprev)
        if img is None or img_prev is None or img_preprev is None:
            raise RuntimeError(f"failed to read input frames: {path}")

        orig_height, orig_width = img.shape[:2]
        img = cv2.resize(img, (self.width, self.height))
        img_prev = cv2.resize(img_prev, (self.width, self.height))
        img_preprev = cv2.resize(img_preprev, (self.width, self.height))

        if self.augment:
            img, img_prev, img_preprev = self.apply_augmentations([img, img_prev, img_preprev])

        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        return imgs, orig_width, orig_height

    def make_heatmap(self, x, y, vis, orig_width, orig_height):
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        if vis == 0 or x < 0 or y < 0:
            return heatmap[None, :, :]

        x_model = x * self.width / float(orig_width)
        y_model = y * self.height / float(orig_height)
        radius = self.heatmap_radius
        sigma2 = 2.0 * self.heatmap_sigma * self.heatmap_sigma
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

    def make_hardneg_mask(self, path, orig_width, orig_height, target_heatmap):
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        parsed = self.parse_frame_path(path)
        if parsed is None:
            return mask[None, :, :]
        row = self.mapped.get(parsed)
        if not row:
            return mask[None, :, :]

        sx = self.width / float(orig_width)
        sy = self.height / float(orig_height)

        for box in self.parse_player_boxes(row.get("player_boxes", "")):
            xmin, ymin, xmax, ymax = box
            # Lower body/shoes are the strongest distractors, but keep part of the full body.
            self.fill_box(mask, xmin * sx, ymin * sy, xmax * sx, ymax * sy, self.hardneg_player_weight * 0.35)
            lower_y = ymin + (ymax - ymin) * 0.58
            self.fill_box(mask, xmin * sx, lower_y * sy, xmax * sx, ymax * sy, self.hardneg_player_weight)

        for idx in range(1, 15):
            point = row.get(f"court_{idx}", "")
            if not point:
                continue
            try:
                px, py = [float(v) for v in point.split(",", 1)]
            except ValueError:
                continue
            self.fill_disk(mask, px * sx, py * sy, self.court_radius, self.hardneg_court_weight)

        # Never penalize the positive ball heatmap support.
        mask[target_heatmap > 0.05] = 0.0
        return mask[None, :, :]

    @staticmethod
    def parse_player_boxes(value):
        boxes = []
        if not value:
            return boxes
        for item in value.split("|"):
            parts = item.split(",")
            if len(parts) != 4:
                continue
            try:
                boxes.append(tuple(float(v) for v in parts))
            except ValueError:
                continue
        return boxes

    @staticmethod
    def fill_box(mask, xmin, ymin, xmax, ymax, value):
        h, w = mask.shape
        x0 = max(0, min(w, int(math.floor(xmin))))
        x1 = max(0, min(w, int(math.ceil(xmax))))
        y0 = max(0, min(h, int(math.floor(ymin))))
        y1 = max(0, min(h, int(math.ceil(ymax))))
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], value)

    @staticmethod
    def fill_disk(mask, cx, cy, radius, value):
        h, w = mask.shape
        x0 = max(0, int(math.floor(cx - radius)))
        x1 = min(w, int(math.ceil(cx + radius + 1)))
        y0 = max(0, int(math.floor(cy - radius)))
        y1 = min(h, int(math.ceil(cy + radius + 1)))
        if x0 >= x1 or y0 >= y1:
            return
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        disk = ((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2) <= radius * radius
        patch = mask[y0:y1, x0:x1]
        patch[disk] = np.maximum(patch[disk], value)

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
            noisy = []
            for frame in frames:
                noise = np.random.normal(0.0, sigma, frame.shape).astype(np.float32)
                noisy.append(np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8))
            frames = noisy

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
        s = float(kernel.sum())
        if s <= 0:
            kernel[center, :] = 1.0
            s = float(kernel.sum())
        return kernel / s

    @staticmethod
    def jpeg_compress(frame, quality):
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return frame
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return frame if decoded is None else decoded
