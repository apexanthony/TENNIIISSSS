import math
import sys
from pathlib import Path

import cv2
import numpy as np


V38_DIR = Path(__file__).resolve().parent
V37_DIR = V38_DIR.parent / "v3_7_tracknet_ball_hardneg"
if str(V37_DIR) not in sys.path:
    sys.path.insert(0, str(V37_DIR))

from datasets_v37_hardneg import TrackNetDatasetV37HardNeg  # noqa: E402


class TrackNetDatasetV38(TrackNetDatasetV37HardNeg):
    """Unified V3.8 dataset for baseline/CA/HN/Aux/Full ablations."""

    def __init__(
        self,
        *args,
        aux_mapped_csv="datasets/tennis_all_v4i_mapped/annotations_hardneg_cleaned_strict.csv",
        aux_heatmap_radius=5,
        aux_heatmap_sigma=1.8,
        rgb_input=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aux_mapped = self.load_aux_annotations(aux_mapped_csv)
        self.aux_heatmap_radius = int(aux_heatmap_radius)
        self.aux_heatmap_sigma = float(aux_heatmap_sigma)
        self.rgb_input = bool(rgb_input)
        print(f"aux_mapped={len(self.aux_mapped)}, rgb_input={self.rgb_input}", flush=True)

    @staticmethod
    def load_aux_annotations(path_value):
        import pandas as pd

        if not path_value:
            return {}
        path = Path(path_value)
        if not path.exists():
            print(f"warning: auxiliary mapping not found: {path}", flush=True)
            return {}
        frame = pd.read_csv(path)
        if "matched_tracknet" in frame.columns:
            frame = frame[pd.to_numeric(frame["matched_tracknet"], errors="coerce").fillna(0) == 1]
        if "tracknet_visibility" in frame.columns:
            frame = frame[pd.to_numeric(frame["tracknet_visibility"], errors="coerce").fillna(0) != 0]
        frame["ball_cx"] = pd.to_numeric(frame.get("ball_cx"), errors="coerce")
        frame["ball_cy"] = pd.to_numeric(frame.get("ball_cy"), errors="coerce")
        frame = frame.dropna(subset=["ball_cx", "ball_cy"])
        sort_column = "selected_error" if "selected_error" in frame.columns else None
        if sort_column:
            frame[sort_column] = pd.to_numeric(frame[sort_column], errors="coerce").fillna(float("inf"))
            frame = frame.sort_values(sort_column)
        frame = frame.drop_duplicates(["game", "clip", "frame_name"], keep="first")
        mapped = {}
        for _, row in frame.iterrows():
            key = (str(row["game"]), str(row["clip"]), str(row["frame_name"]))
            mapped[key] = (float(row["ball_cx"]), float(row["ball_cy"]))
        return mapped

    def __getitem__(self, idx):
        output = super().__getitem__(idx)
        row = self.data.iloc[idx]
        key = (str(row["game"]), str(row["clip"]), str(row["frame_name"]))
        point = self.aux_mapped.get(key)
        if point is None:
            output["aux_heatmap"] = np.zeros_like(output["ball_heatmap"], dtype=np.float32)
            output["aux_valid"] = np.float32(0.0)
        else:
            output["aux_heatmap"] = self.make_gaussian(
                point[0],
                point[1],
                float(output["orig_width"]),
                float(output["orig_height"]),
                self.aux_heatmap_radius,
                self.aux_heatmap_sigma,
            )
            output["aux_valid"] = np.float32(1.0)
        return output

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
        for frame_index, frame in enumerate(resized):
            network_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.rgb_input else frame
            chw = network_frame.transpose(2, 0, 1).astype(np.float16) * np.float16(1.0 / 255.0)
            stacked[frame_index * 3 : (frame_index + 1) * 3] = chw
        return stacked, resized[0], orig_width, orig_height

    def make_gaussian(self, x, y, orig_width, orig_height, radius, sigma):
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        if not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0:
            return heatmap[None]
        x_model = x * self.width / float(orig_width)
        y_model = y * self.height / float(orig_height)
        cx, cy = int(round(x_model)), int(round(y_model))
        x0, x1 = max(0, cx - radius), min(self.width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(self.height, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return heatmap[None]
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        patch = np.exp(-((xs[None] - x_model) ** 2 + (ys[:, None] - y_model) ** 2) / (2.0 * sigma * sigma))
        heatmap[y0:y1, x0:x1] = patch
        return heatmap[None]
