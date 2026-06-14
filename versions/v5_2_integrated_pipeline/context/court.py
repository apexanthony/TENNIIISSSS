import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
Polygon = List[Point]


class CourtMapSequence(dict):
    def __init__(self):
        super().__init__()
        self.qualities: Dict[int, float] = {}


def homography_image_corners(matrix) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    canonical = np.asarray([[[0.0, 0.0], [1000.0, 0.0], [0.0, 2168.0], [1000.0, 2168.0]]], dtype=np.float32)
    try:
        inverse = np.linalg.inv(matrix)
        return cv2.perspectiveTransform(canonical, inverse)[0]
    except (cv2.error, np.linalg.LinAlgError):
        return None


def homography_from_image_corners(corners) -> Optional[np.ndarray]:
    if corners is None or len(corners) != 4:
        return None
    src = np.asarray(corners, dtype=np.float32)
    if abs(cv2.contourArea(src[[0, 1, 3, 2]])) < 1.0:
        return None
    dst = np.asarray([[0.0, 0.0], [1000.0, 0.0], [0.0, 2168.0], [1000.0, 2168.0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def stabilize_homography_update(previous_corners, matrix, width, height, alpha=0.22):
    corners = homography_image_corners(matrix)
    if corners is None:
        return previous_corners, None, 0.0

    frame_diag = max(float(np.hypot(width, height)), 1.0)
    polygon = corners[[0, 1, 3, 2]]
    area = abs(float(cv2.contourArea(polygon)))
    if area < width * height * 0.06 or area > width * height * 1.8:
        return previous_corners, None, 0.0
    if np.any(corners[:, 0] < -0.6 * width) or np.any(corners[:, 0] > 1.6 * width):
        return previous_corners, None, 0.0
    if np.any(corners[:, 1] < -0.6 * height) or np.any(corners[:, 1] > 1.6 * height):
        return previous_corners, None, 0.0

    if previous_corners is None:
        return corners, homography_from_image_corners(corners), 0.75

    displacement = np.linalg.norm(corners - previous_corners, axis=1)
    median_shift = float(np.median(displacement))
    max_shift = float(np.max(displacement))
    if median_shift > 0.08 * frame_diag or max_shift > 0.16 * frame_diag:
        return previous_corners, None, 0.0

    quality = max(0.15, 1.0 - median_shift / (0.08 * frame_diag))
    adaptive_alpha = max(0.08, min(alpha, alpha * quality + 0.05))
    smoothed = (1.0 - adaptive_alpha) * previous_corners + adaptive_alpha * corners
    return smoothed, homography_from_image_corners(smoothed), quality


def court_map_quality(court_maps, frame_id: int, default: float = 1.0) -> float:
    qualities = getattr(court_maps, "qualities", None)
    if qualities is None:
        return default if frame_id in court_maps else 0.0
    if frame_id in qualities:
        return float(qualities[frame_id])
    return 0.0


def load_playable_polygons(path: str) -> List[Polygon]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    polygons = data.get("playable_ground") or data.get("regions") or []
    out: List[Polygon] = []
    for poly in polygons:
        out.append([(float(p[0]), float(p[1])) for p in poly])
    return out


def point_in_polygons(point: Point, polygons: Sequence[Polygon]) -> bool:
    if not polygons:
        return True
    pt = (float(point[0]), float(point[1]))
    for poly in polygons:
        arr = np.asarray(poly, dtype=np.float32)
        if cv2.pointPolygonTest(arr, pt, False) >= 0:
            return True
    return False


def draw_polygons(frame, polygons: Sequence[Polygon]):
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (80, 255, 80), 2)
    return frame


def parse_court_point(value: str) -> Optional[Point]:
    if value is None:
        return None
    value = str(value).strip().strip('"')
    if not value or value.lower() == "nan":
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        return None
    return float(parts[0]), float(parts[1])


def homography_from_court_points(points: Sequence[Optional[Point]]):
    """Build image->mini-court homography from outer court corners.

    The current Roboflow court keypoints follow the common 14-point layout.
    We use points 1, 4, 11, 14 as the outer quadrilateral.
    """
    if len(points) < 14:
        return None
    tl = points[0]
    tr = points[3]
    bl = points[10]
    br = points[13]
    if tl is None or tr is None or bl is None or br is None:
        return None
    src = np.asarray([tl, tr, bl, br], dtype=np.float32)
    dst = np.asarray([[0.0, 0.0], [1000.0, 0.0], [0.0, 2168.0], [1000.0, 2168.0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def load_annotation_court_maps(path: str, game: str = "", clip: str = "") -> Dict[int, np.ndarray]:
    if not path:
        return {}
    maps: Dict[int, np.ndarray] = {}
    with open(path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if game and row.get("game") != game:
                continue
            if clip and row.get("clip") != clip:
                continue
            frame = int(float(row.get("frame_id") or row.get("frame") or 0))
            points = [parse_court_point(row.get(f"court_{idx}", "")) for idx in range(1, 15)]
            matrix = homography_from_court_points(points)
            if matrix is not None:
                maps[frame] = matrix
    return maps


def transform_point(matrix, point: Optional[Point]) -> Optional[Point]:
    if matrix is None or point is None:
        return None
    src = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, matrix)[0][0]
    return float(dst[0]), float(dst[1])


def draw_mini_court(frame, court_points: Sequence[Optional[Point]]):
    if len(court_points) < 14:
        return frame
    valid = [p for p in court_points if p is not None]
    for point in valid:
        cv2.circle(frame, (int(point[0]), int(point[1])), 4, (120, 255, 120), -1)
    return frame


def _line_intersection(line_a, line_b):
    x1, y1, x2, y2, *_ = line_a
    x3, y3, x4, y4, *_ = line_b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return float(px), float(py)


def _line_angle(line):
    x1, y1, x2, y2, *_ = line
    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
    if angle > 90.0:
        angle = 180.0 - angle
    return float(angle)


def _score_court_lines(lines, width, height):
    scored = []
    margin_x = width * 0.35
    margin_y = height * 0.35
    for i, line in enumerate(lines):
        count = 0
        for j, other in enumerate(lines):
            if i == j:
                continue
            pt = _line_intersection(line, other)
            if pt is None:
                continue
            x, y = pt
            if -margin_x <= x <= width + margin_x and -margin_y <= y <= height + margin_y:
                count += 1
        length = line[4]
        scored.append((*line, float(count), float(length + 25.0 * count)))
    return sorted(scored, key=lambda item: item[-1], reverse=True)


def _quad_area(points):
    arr = np.asarray(points, dtype=np.float32)
    return float(abs(cv2.contourArea(arr)))


def _build_homography_from_lines(horizontal, vertical, width, height):
    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    top = min(horizontal, key=lambda item: (item[1] + item[3]) * 0.5)
    bottom = max(horizontal, key=lambda item: (item[1] + item[3]) * 0.5)
    left = min(vertical, key=lambda item: (item[0] + item[2]) * 0.5)
    right = max(vertical, key=lambda item: (item[0] + item[2]) * 0.5)

    tl = _line_intersection(top, left)
    tr = _line_intersection(top, right)
    bl = _line_intersection(bottom, left)
    br = _line_intersection(bottom, right)
    if tl is None or tr is None or bl is None or br is None:
        return None

    src = np.asarray([tl, tr, bl, br], dtype=np.float32)
    if np.any(src[:, 0] < -width) or np.any(src[:, 0] > 2 * width):
        return None
    if np.any(src[:, 1] < -height) or np.any(src[:, 1] > 2 * height):
        return None
    if _quad_area(src) < width * height * 0.08:
        return None

    dst = np.asarray([[0.0, 0.0], [1000.0, 0.0], [0.0, 2168.0], [1000.0, 2168.0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def detect_court_homography_from_frame(frame):
    """Best-effort automatic court homography from bright court lines.

    This is a lightweight fallback inspired by the reference project. It is not
    meant to replace annotated court keypoints, but it can provide a first-pass
    mapping for fixed-camera videos with visible white court lines.
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    bright = cv2.dilate(bright, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.Canny(bright, 80, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=max(60, width // 12), maxLineGap=20)
    if lines is None:
        return None

    candidates = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < max(60, width * 0.08):
            continue
        candidates.append((x1, y1, x2, y2, length))

    if len(candidates) < 4:
        return None

    scored = _score_court_lines(candidates[:80], width, height)[:24]
    horizontal = [line for line in scored if _line_angle(line) < 28.0]
    vertical = [line for line in scored if _line_angle(line) > 48.0]
    matrix = _build_homography_from_lines(horizontal, vertical, width, height)
    if matrix is not None:
        return matrix

    # Fallback to a simple long-line split for side-view cases where one family
    # is not close to vertical.
    ranked = sorted(scored, key=lambda item: item[4], reverse=True)[:16]
    if len(ranked) < 4:
        return None
    angles = np.asarray([_line_angle(line) for line in ranked], dtype=np.float32)
    median = float(np.median(angles))
    family_a = [line for line in ranked if _line_angle(line) <= median]
    family_b = [line for line in ranked if _line_angle(line) > median]
    return _build_homography_from_lines(family_a, family_b, width, height)
