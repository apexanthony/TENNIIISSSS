import csv
import math
import os
from pathlib import Path

import numpy as np

from datasets_v2 import trackNetDatasetV2


class TrackNetDatasetV33AuxPos(trackNetDatasetV2):
    def __init__(
        self,
        mode,
        input_height=270,
        input_width=480,
        heatmap_radius=6,
        heatmap_sigma=2.0,
        aux_heatmap_radius=5,
        aux_heatmap_sigma=1.8,
        augment=False,
        mapped_csv="datasets/tennis_all_v4i_mapped/annotations_mapped.csv",
    ):
        super().__init__(
            mode,
            input_height=input_height,
            input_width=input_width,
            heatmap_radius=heatmap_radius,
            heatmap_sigma=heatmap_sigma,
            augment=augment,
        )
        self.aux_heatmap_radius = aux_heatmap_radius
        self.aux_heatmap_sigma = aux_heatmap_sigma
        self.mapped = self.load_mapped_annotations(mapped_csv)
        print(f"mapped aux-positive frames = {len(self.mapped)}")

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
        aux_heatmap, aux_valid = self.make_aux_heatmap(path, orig_width, orig_height)

        return inputs, heatmap, x, y, vis, aux_heatmap, aux_valid

    @staticmethod
    def parse_frame_path(path):
        parts = Path(path).parts
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

    def make_aux_heatmap(self, path, orig_width, orig_height):
        parsed = self.parse_frame_path(path)
        if parsed is None:
            return np.zeros((1, self.height, self.width), dtype=np.float32), np.float32(0.0)
        row = self.mapped.get(parsed)
        if not row:
            return np.zeros((1, self.height, self.width), dtype=np.float32), np.float32(0.0)
        try:
            x = float(row.get("ball_cx", ""))
            y = float(row.get("ball_cy", ""))
        except ValueError:
            return np.zeros((1, self.height, self.width), dtype=np.float32), np.float32(0.0)
        if x < 0 or y < 0:
            return np.zeros((1, self.height, self.width), dtype=np.float32), np.float32(0.0)

        old_radius = self.heatmap_radius
        old_sigma = self.heatmap_sigma
        self.heatmap_radius = self.aux_heatmap_radius
        self.heatmap_sigma = self.aux_heatmap_sigma
        heatmap = self.make_heatmap(x, y, 1, orig_width, orig_height)
        self.heatmap_radius = old_radius
        self.heatmap_sigma = old_sigma
        return heatmap, np.float32(1.0)
