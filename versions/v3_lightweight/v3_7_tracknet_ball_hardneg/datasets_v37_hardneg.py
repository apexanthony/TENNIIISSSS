import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


COURT_SEGMENTS = [
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7),
    (8, 9), (9, 10),
    (11, 12), (12, 13), (13, 14),
    (1, 11), (2, 12), (3, 13), (4, 14),
    (5, 8), (6, 9), (7, 10),
]


class TrackNetDatasetV37HardNeg(Dataset):
    """V3.7 ball-only dataset with multi-source hard-negative masks.

    Inputs and labels come from the original TrackNet dataset. Roboflow mapped
    annotations are used only to build negative masks for player body/shoes and
    court lines; they are not extra output heads.
    """

    def __init__(
        self,
        manifest_csv,
        input_height=360,
        input_width=640,
        heatmap_radius=8,
        heatmap_sigma=3.0,
        tracknet_root="datasets/trackNet",
        mapped_csv="datasets/tennis_all_v4i_mapped/annotations_hardneg_cleaned_strict.csv",
        augment=False,
        hardneg_player_weight=1.0,
        hardneg_shoe_weight=1.4,
        hardneg_court_weight=0.85,
        hardneg_bright_weight=0.35,
        hardneg_edge_weight=0.18,
        hardneg_ball_clear_radius=18,
        bright_threshold=190,
        augment_min_ball_contrast=4.0,
        augment_min_contrast_ratio=0.55,
    ):
        self.data = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.height = int(input_height)
        self.width = int(input_width)
        self.heatmap_radius = int(heatmap_radius)
        self.heatmap_sigma = float(heatmap_sigma)
        self.tracknet_root = Path(tracknet_root)
        self.augment = bool(augment)
        self.hardneg_player_weight = float(hardneg_player_weight)
        self.hardneg_shoe_weight = float(hardneg_shoe_weight)
        self.hardneg_court_weight = float(hardneg_court_weight)
        self.hardneg_bright_weight = float(hardneg_bright_weight)
        self.hardneg_edge_weight = float(hardneg_edge_weight)
        self.hardneg_ball_clear_radius = int(hardneg_ball_clear_radius)
        self.bright_threshold = int(bright_threshold)
        self.augment_min_ball_contrast = float(augment_min_ball_contrast)
        self.augment_min_contrast_ratio = float(augment_min_contrast_ratio)
        self.mapped = self.load_mapped_annotations(mapped_csv)
        print(
            f"manifest={manifest_csv}, samples={len(self.data)}, size={self.width}x{self.height}, mapped={len(self.mapped)}",
            flush=True,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path1 = str(row["path1"])
        path2 = str(row["path2"])
        path3 = str(row["path3"])
        x = self.to_float(row.get("x-coordinate", row.get("x", "")))
        y = self.to_float(row.get("y-coordinate", row.get("y", "")))
        visibility = int(float(row.get("visibility", 0) or 0))
        if math.isnan(x) or math.isnan(y) or visibility == 0:
            x = -1.0
            y = -1.0
            visibility = 0

        current_path = self.tracknet_root / path1
        prev1_path = self.tracknet_root / path2
        prev2_path = self.tracknet_root / path3
        inputs, current, orig_width, orig_height = self.get_input(
            current_path, prev1_path, prev2_path, x, y, visibility
        )
        heatmap = self.make_heatmap(x, y, visibility, orig_width, orig_height)
        mapped_row = self.mapped.get((str(row["game"]), str(row["clip"]), str(row["frame_name"])))
        hardneg_mask = self.make_hardneg_mask(current, mapped_row, x, y, visibility, orig_width, orig_height, heatmap[0])

        return {
            "input": inputs,
            "ball_heatmap": heatmap,
            "hardneg_mask": hardneg_mask,
            "visibility": np.int64(visibility),
            "x": np.float32(x),
            "y": np.float32(y),
            "orig_width": np.float32(orig_width),
            "orig_height": np.float32(orig_height),
            "clip_key": str(row["clip_key"]),
            "frame_id": np.int64(row["frame_id"]),
        }

    @staticmethod
    def to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def read_image(path):
        return cv2.imread(str(path)) if path else None

    @classmethod
    def load_mapped_annotations(cls, mapped_csv):
        mapped = {}
        if not mapped_csv:
            print("mapped hard-negative annotations disabled", flush=True)
            return mapped
        path = Path(mapped_csv)
        if not path.exists():
            print(f"warning: mapped csv not found: {mapped_csv}", flush=True)
            return mapped
        df = pd.read_csv(path)
        duplicate_count = int(df.duplicated(["game", "clip", "frame_name"]).sum())
        if duplicate_count:
            raise ValueError(f"mapped csv contains {duplicate_count} duplicate frame keys: {mapped_csv}")
        for _, row in df.iterrows():
            mapped[(str(row["game"]), str(row["clip"]), str(row["frame_name"]))] = row
        return mapped

    def get_input(self, current_path, prev1_path, prev2_path, ball_x, ball_y, visibility):
        current = self.read_image(current_path)
        prev1 = self.read_image(prev1_path)
        prev2 = self.read_image(prev2_path)
        if current is None:
            raise RuntimeError(f"failed to read current frame: {current_path}")
        if prev1 is None:
            prev1 = current
        if prev2 is None:
            prev2 = prev1

        orig_height, orig_width = current.shape[:2]
        resized = [
            cv2.resize(current, (self.width, self.height)),
            cv2.resize(prev1, (self.width, self.height)),
            cv2.resize(prev2, (self.width, self.height)),
        ]
        if self.augment:
            ball_point = None
            if visibility != 0 and ball_x >= 0 and ball_y >= 0:
                ball_point = (
                    ball_x * self.width / float(orig_width),
                    ball_y * self.height / float(orig_height),
                )
            resized = self.apply_augmentations(resized, ball_point)

        stacked = np.empty((9, self.height, self.width), dtype=np.float16)
        for frame_idx, frame in enumerate(resized):
            chw = frame.transpose(2, 0, 1).astype(np.float16)
            chw *= np.float16(1.0 / 255.0)
            stacked[frame_idx * 3 : (frame_idx + 1) * 3] = chw
        return stacked, resized[0], orig_width, orig_height

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

    def make_hardneg_mask(self, current_frame, mapped_row, ball_x, ball_y, visibility, orig_width, orig_height, target_heatmap):
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        if mapped_row is not None:
            self.add_player_hardneg(mask, mapped_row, orig_width, orig_height)
            self.add_court_line_hardneg(mask, mapped_row, orig_width, orig_height)
        self.add_image_hardneg(mask, current_frame)

        mask[target_heatmap > 0.03] = 0.0
        if visibility != 0 and ball_x >= 0 and ball_y >= 0:
            bx = int(round(ball_x * self.width / float(orig_width)))
            by = int(round(ball_y * self.height / float(orig_height)))
            self.clear_disk(mask, bx, by, max(self.hardneg_ball_clear_radius, self.heatmap_radius * 2))
        return mask[None, :, :]

    def add_player_hardneg(self, mask, row, orig_width, orig_height):
        sx = self.width / float(orig_width)
        sy = self.height / float(orig_height)
        boxes = self.parse_boxes(row.get("player_boxes", ""))
        for x1, y1, x2, y2 in boxes:
            self.fill_box(mask, x1 * sx, y1 * sy, x2 * sx, y2 * sy, self.hardneg_player_weight * 0.38)
            lower_y = y1 + (y2 - y1) * 0.58
            self.fill_box(mask, x1 * sx, lower_y * sy, x2 * sx, y2 * sy, self.hardneg_shoe_weight)

    def add_court_line_hardneg(self, mask, row, orig_width, orig_height):
        sx = self.width / float(orig_width)
        sy = self.height / float(orig_height)
        points = {}
        for idx in range(1, 15):
            point = self.parse_point(row.get(f"court_{idx}", ""))
            if point is not None:
                points[idx] = (point[0] * sx, point[1] * sy)
                self.fill_disk(mask, points[idx][0], points[idx][1], 5, self.hardneg_court_weight)
        for a, b in COURT_SEGMENTS:
            if a in points and b in points:
                self.draw_line(mask, points[a], points[b], self.hardneg_court_weight, thickness=5)

    def add_image_hardneg(self, mask, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bright = ((hsv[:, :, 2] >= self.bright_threshold) & (hsv[:, :, 1] <= 90)).astype(np.uint8) * 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        bright = cv2.dilate(bright, np.ones((3, 3), np.uint8), iterations=1)
        mask[bright > 0] = np.maximum(mask[bright > 0], self.hardneg_bright_weight)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 120, 220)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        # Upper/lateral edge-heavy areas often contain ad boards, score bugs, and audience.
        edge_region = np.zeros_like(edges, dtype=bool)
        h, w = edges.shape
        edge_region[: int(h * 0.22), :] = True
        edge_region[:, : int(w * 0.08)] = True
        edge_region[:, int(w * 0.92) :] = True
        active = (edges > 0) & edge_region
        mask[active] = np.maximum(mask[active], self.hardneg_edge_weight)

    @staticmethod
    def parse_boxes(value):
        boxes = []
        if not isinstance(value, str) or not value:
            return boxes
        for item in value.split("|"):
            parts = [p.strip() for p in item.split(",") if p.strip()]
            if len(parts) != 4:
                continue
            try:
                boxes.append(tuple(float(v) for v in parts))
            except ValueError:
                continue
        return boxes

    @staticmethod
    def parse_point(value):
        if not isinstance(value, str) or not value:
            return None
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

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

    @staticmethod
    def clear_disk(mask, cx, cy, radius):
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
        mask[y0:y1, x0:x1][disk] = 0.0

    @staticmethod
    def draw_line(mask, p0, p1, value, thickness=5):
        layer = np.zeros_like(mask, dtype=np.float32)
        cv2.line(layer, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), value, thickness)
        np.maximum(mask, layer, out=mask)

    def apply_augmentations(self, frames, ball_point=None):
        if np.random.rand() < 0.80:
            alpha = np.random.uniform(0.70, 1.40)
            beta = np.random.uniform(-30.0, 30.0)
            frames = [np.clip(frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8) for frame in frames]

        if np.random.rand() < 0.45:
            ksize = int(np.random.choice([3, 5, 7, 9]))
            angle = np.random.uniform(0, np.pi)
            kernel = self.motion_blur_kernel(ksize, angle)
            candidate = [cv2.filter2D(frame, -1, kernel) for frame in frames]
            frames = self.keep_if_ball_visible(frames, candidate, ball_point)

        if np.random.rand() < 0.40:
            quality = int(np.random.randint(30, 90))
            candidate = [self.jpeg_compress(frame, quality) for frame in frames]
            frames = self.keep_if_ball_visible(frames, candidate, ball_point)

        if np.random.rand() < 0.25:
            sigma = np.random.uniform(2.0, 10.0)
            candidate = [
                np.clip(frame.astype(np.float32) + np.random.normal(0.0, sigma, frame.shape), 0, 255).astype(np.uint8)
                for frame in frames
            ]
            frames = self.keep_if_ball_visible(frames, candidate, ball_point)
        return frames

    def keep_if_ball_visible(self, previous, candidate, ball_point):
        if ball_point is None:
            return candidate
        previous_contrast = self.local_ball_contrast(previous[0], ball_point)
        candidate_contrast = self.local_ball_contrast(candidate[0], ball_point)
        minimum = max(
            self.augment_min_ball_contrast,
            previous_contrast * self.augment_min_contrast_ratio,
        )
        return candidate if candidate_contrast >= minimum else previous

    @staticmethod
    def local_ball_contrast(frame, ball_point, inner_radius=2, outer_radius=7):
        cx = int(round(ball_point[0]))
        cy = int(round(ball_point[1]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        x0 = max(0, cx - outer_radius)
        x1 = min(w, cx + outer_radius + 1)
        y0 = max(0, cy - outer_radius)
        y1 = min(h, cy + outer_radius + 1)
        if x0 >= x1 or y0 >= y1:
            return 0.0

        ys, xs = np.ogrid[y0:y1, x0:x1]
        dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
        inner = dist2 <= inner_radius * inner_radius
        ring = (dist2 > (inner_radius + 1) ** 2) & (dist2 <= outer_radius * outer_radius)
        patch = gray[y0:y1, x0:x1]
        if not np.any(inner) or not np.any(ring):
            return 0.0
        center_values = patch[inner]
        ring_values = patch[ring]
        return float(abs(center_values.mean() - ring_values.mean()) + 0.5 * center_values.std())

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
