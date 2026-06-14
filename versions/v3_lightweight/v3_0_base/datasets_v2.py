from torch.utils.data import Dataset
import csv
import os
import cv2
import math
import numpy as np


class trackNetDatasetV2(Dataset):
    def __init__(
        self,
        mode,
        input_height=360,
        input_width=640,
        heatmap_radius=8,
        heatmap_sigma=3.0,
        augment=False,
    ):
        self.path_dataset = "./datasets/trackNet"
        assert mode in ["train", "val"], "incorrect mode"
        labels_path = os.path.join(self.path_dataset, "labels_{}.csv".format(mode))
        with open(labels_path, newline="") as f:
            self.data = list(csv.DictReader(f))
        print("mode = {}, samples = {}".format(mode, len(self.data)))
        self.height = input_height
        self.width = input_width
        self.heatmap_radius = heatmap_radius
        self.heatmap_sigma = heatmap_sigma
        self.augment = augment and mode == "train"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        path = row.get("path1") or row.get("path")
        path_prev = row.get("path2") or row.get("path_prev")
        path_preprev = row.get("path3") or row.get("path_preprev")
        path_gt = row.get("gt_path", "")
        x = self._to_float(row.get("x-coordinate", row.get("x", "")))
        y = self._to_float(row.get("y-coordinate", row.get("y", "")))
        vis = int(float(row.get("visibility", row.get("vis", 0)) or 0))

        path = os.path.join(self.path_dataset, path)
        path_prev = os.path.join(self.path_dataset, path_prev)
        path_preprev = os.path.join(self.path_dataset, path_preprev)
        path_gt = os.path.join(self.path_dataset, path_gt) if path_gt else ""
        if math.isnan(x):
            x = -1
            y = -1

        inputs, orig_width, orig_height = self.get_input(path, path_prev, path_preprev)
        if path_gt:
            outputs = self.get_output(path_gt)
        else:
            outputs = self.make_heatmap(x, y, vis, orig_width, orig_height)

        return inputs, outputs, x, y, vis

    @staticmethod
    def _to_float(value):
        if value is None or value == "":
            return float("nan")
        return float(value)

    def get_output(self, path_gt):
        img = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("failed to read gt image: {}".format(path_gt))
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32) / 255.0
        return img[None, :, :]

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

    def get_input(self, path, path_prev, path_preprev):
        img = cv2.imread(path)
        img_prev = cv2.imread(path_prev)
        img_preprev = cv2.imread(path_preprev)
        if img is None or img_prev is None or img_preprev is None:
            raise RuntimeError("failed to read input frames")

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
